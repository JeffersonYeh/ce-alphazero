import datetime
import os
import pickle
import sys
import time
from functools import partial
from typing import Any, Optional
import pathlib
import flashbax as fbx  # type: ignore
import haiku as hk
import jax
import jax.numpy as jnp
import omegaconf
import optax  # type: ignore
import pgx  # type: ignore
import wandb
from pgx.experimental import auto_reset  # type: ignore
from flashbax.vault import Vault
import matplotlib.pyplot as plt
import cemctx as emctx

from config import Config, setup_config
from context import Context, get_epistemic_recurrent_fn, get_forward_fn
from envs.deep_sea import DeepSea
from envs.safety_freeway import SafetyMinAtarFreeway
from envs.safety_grid import SafetyGrid
from envs.subleq import Subleq, SubleqTask
from evaluate import evaluate
from reanalyze import reanalyze
from selfplay import selfplay, uniformrandomplay
from train import train

from type_aliases import PRNGKey, Array, Model


class TimeoutTerminationWrapper(pgx.Env):
    """
    A wrapper to induce termination on timeout.
    Automatically supports all methods from the original environment.
    """

    def __init__(self, env: pgx.Env, timelimit: int):
        super().__init__()
        self.env = env
        self.timelimit = timelimit
        self.step_counter = 0

    def init(self, key: PRNGKey) -> pgx.State:
        state = self.env.init(key)
        self.step_counter = 0
        return state

    def step(
        self,
        state: pgx.State,
        action: Array,
        key: Optional[Array] = None,
    ) -> pgx.State:
        new_state = self.env.step(state, action, key)
        self.step_counter += 1
        if self.step_counter >= self.timelimit:
            new_state.terminated = True

        return new_state

    # Delegate abstract methods to the underlying environment
    def _init(self, *args, **kwargs):
        return self.env._init(*args, **kwargs)

    def _observe(self, *args, **kwargs):
        return self.env._observe(*args, **kwargs)

    def _step(self, *args, **kwargs):
        return self.env._step(*args, **kwargs)

    @property
    def id(self):
        return self.env.id

    @property
    def num_players(self):
        return self.env.num_players

    @property
    def version(self):
        return self.env.version

    # Delegate other methods/attributes
    def __getattr__(self, name):
        """
        Delegate all other method calls to the wrapped environment.
        """
        return getattr(self.env, name)


@partial(jax.pmap, static_broadcasted_argnums=[2])
def debug_deep_sea(all_states_batch, model, context):
    model_params, model_state = model

    (_exploitation_logits, _exploration_logits, _value, value_epistemic_variance, reward_epistemic_variance), _ = (
        context.forward.apply(model_params, model_state, all_states_batch, is_training=True)
    )
    ube_predictions = jnp.reshape(
        value_epistemic_variance, shape=(all_states_batch.shape[-1], all_states_batch.shape[-1])
    )
    unseen_states = jnp.reshape(
        reward_epistemic_variance, shape=(all_states_batch.shape[-1], all_states_batch.shape[-1])
    )
    return ube_predictions, unseen_states


def main() -> None:
    # Get configuration from CLI.
    config_dict = omegaconf.OmegaConf.from_cli()
    config: Config = Config(**config_dict)  # type: ignore
    config: Config = setup_config(config)

    if config.env_class == "custom" and "deep_sea" in config.env_id:
        s = config.env_id.removeprefix("deep_sea-")
        size_of_grid = int(s) if s.isnumeric() else 4
        all_states_batch = jnp.zeros([size_of_grid * size_of_grid, size_of_grid, size_of_grid], dtype=jnp.bool)
        # all_states_batch = jnp.zeros([config.selfplay_batch_size, size_of_grid, size_of_grid], dtype=jnp.bool)
        # for each row
        for i in range(size_of_grid):
            # for each column
            for j in range(size_of_grid):
                all_states_batch = all_states_batch.at[i * size_of_grid + j, i, j].set(True)
        all_states_batch = jnp.expand_dims(all_states_batch, axis=0)

    # Make the results directory
    pathlib.Path(f"{config.results_path}/{config.wandb_run_name}").mkdir(parents=True, exist_ok=True)
    complete_results_path = f"{config.results_path}/{config.wandb_run_name}"

    print(f"Printing the config:\n{config}", flush=True)

    # Initialize Weights & Biases.
    if config.track:
        wandb.init(
            project=config.wandb_project,
            config=config.model_dump(),
            name=config.wandb_run_name,
            entity=config.wandb_team_name,
        )

    # Identify devices.
    devices = jax.local_devices()
    num_devices = len(devices)

    # Initialize RNG key.
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey_for_env, subkey_for_dummy_state, subkey_for_init = jax.random.split(rng_key, 4)

    # Make the environment.
    env: pgx.Env
    match (config.env_class, config.env_id):
        case ("safety", str(s)) if s.startswith("safety-minatar-freeway"):
            env = TimeoutTerminationWrapper(SafetyMinAtarFreeway(), timelimit=config.max_episode_length)
        case ("safety", str(s)) if s.startswith("safety-grid"):
            env = TimeoutTerminationWrapper(SafetyGrid(), timelimit=config.max_episode_length)
        case ("custom", str(s)) if s.startswith("deep_sea"):
            # E.g. For DeepSea size 16, use "deep_sea-16".
            s = s.removeprefix("deep_sea-")
            size_of_grid = int(s) if s.isnumeric() else 4
            env = DeepSea(size_of_grid=size_of_grid, action_map_key=subkey_for_env)
        case ("custom", str(s)) if s.startswith("subleq"):
            s = s.removeprefix("subleq-")
            word_size = int(s) if s.isnumeric() else 256
            # TODO: Make tasks configurable with command line args
            # TODO: Make reward function configurable with command line args
            tasks = []
            for task in config.subleq_tasks:
                match task:
                    case "NEGATION_POSITIVE":
                        tasks.append(SubleqTask.NEGATION_POSITIVE)
                    case "NEGATION":
                        tasks.append(SubleqTask.NEGATION)
                    case "IDENTITY":
                        tasks.append(SubleqTask.IDENTITY)
                    case "MAXIMUM":
                        tasks.append(SubleqTask.MAXIMUM)
                    case "MINIMUM":
                        tasks.append(SubleqTask.MINIMUM)
                    case "SUBTRACTION":
                        tasks.append(SubleqTask.SUBTRACTION)
                    case "ADDITION":
                        tasks.append(SubleqTask.ADDITION)
                    case "COMPARISON":
                        tasks.append(SubleqTask.COMPARISON)
                    case "SORT_2":
                        tasks.append(SubleqTask.SORT_2)
                    case "SORT_3":
                        tasks.append(SubleqTask.SORT_3)
                    case "SORT_4":
                        tasks.append(SubleqTask.SORT_4)
                    case "MULTIPLICATION":
                        tasks.append(SubleqTask.MULTIPLICATION)
                    case "DIVISION":
                        tasks.append(SubleqTask.DIVISION)
            print(f"len(tasks) = {len(tasks)}, tasks = {tasks}")
            env = TimeoutTerminationWrapper(
                Subleq(tasks=tasks, word_size=word_size, use_binary_encoding=config.use_binary_encoding),
                timelimit=config.max_episode_length,
            )
        case ("pgx", env_id) if env_id in pgx.available_envs():
            if "minatar" in env_id:
                env = TimeoutTerminationWrapper(pgx.make(env_id), timelimit=config.max_episode_length)
            else:
                env = pgx.make(env_id)

        case (cl, id):
            assert False, f"Invalid environment settings: {cl}, {id}."
    # selfplay_env, planner_env, eval_env = make_envs(config.env_class, config.env_id)
    # baseline = pgx.make_baseline_model(config.env_id + "_v0")  # type: ignore

    # Initialize model.
    forward = get_forward_fn(env, config)
    dummy_state = jax.vmap(env.init)(jax.random.split(subkey_for_dummy_state, 2))
    dummy_input = dummy_state.observation
    model: tuple[hk.MutableParams, hk.MutableState] = forward.init(subkey_for_init, dummy_input)
    # Initialize optimizer.
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state: optax.OptState = optimizer.init(params=model[0])
    # Replicate to all devices.
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Initialize replay buffer.
    # FIXME: UserWarning: Setting max_size dynamically sets the `max_length_time_axis` to be `max_size`//`add_batch_size = 3125`.
    # This allows one to control exactly how many timesteps are stored in the buffer.Note that this overrides the `max_length_time_axis` argument.
    buffer_fn = fbx.make_prioritised_flat_buffer(
        max_length=config.max_replay_buffer_length,
        min_length=max(config.min_replay_buffer_length, config.reanalyze_batch_size),
        sample_batch_size=config.reanalyze_batch_size,
        add_sequences=True,
        add_batch_size=config.selfplay_batch_size,
        priority_exponent=config.priority_exponent,
    )
    buffer_state = buffer_fn.init(jax.tree.map(lambda x: x[0], dummy_state))

    if config.save_replay_buffer:
        parts = config.replay_buffer_path.split("/")
        vault_uid = parts[-1]
        vault_name = parts[-2]
        rel_dir = "/".join(parts[:-2]) + "/" if len(parts) > 2 else ""
        rb_vault = Vault(
            vault_uid=vault_uid, vault_name=vault_name, experience_structure=buffer_state.experience, rel_dir=rel_dir
        )

    # Prepare checkpoint directory.
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize logging information and dictionary.
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    set_bits: int = 0
    log: dict[str, Any] = {"iteration": iteration, "hours": hours, "frames": frames}
    last_states = None
    start_time = time.time()

    mean_returns_list = []
    mean_costs_list = []
    frames_at_mean_returns_list = []
    rewards_not_yet_observed_flag = True
    costs_not_yet_observed_flag = True

    context = Context(
        env=env,
        devices=devices,
        forward=forward,
        selfplay_recurrent_fn=get_epistemic_recurrent_fn(
            env=env,
            forward=forward,
            batch_size=config.selfplay_batch_size,
            exploration=config.directed_exploration,
            discount=config.discount,
            two_players_game=config.two_players_game,
        ),
        reanalyze_recurrent_fn=get_epistemic_recurrent_fn(
            env=env,
            forward=forward,
            batch_size=config.reanalyze_batch_size,
            exploration=False,
            discount=config.discount,
            two_players_game=config.two_players_game,
        ),
        evaluation_recurrent_fn=get_epistemic_recurrent_fn(
            env=env,
            forward=forward,
            batch_size=config.num_eval_episodes,
            exploration=False,
            discount=config.discount,
            two_players_game=config.two_players_game,
        ),
        optimizer=optimizer,
        scale_uncertainty_losses=config.scale_uncertainty_losses,
        hash_path=config.hash_path,  # TODO: Automatically figure this out
        exploration_beta=config.exploration_beta_v,
        max_ube=config.max_ube,
        weigh_losses=config.weigh_losses,
        loss_weighting_temperature=config.loss_weighting_temperature,
        directed_exploration=config.directed_exploration,
    )

    # Training loop
    while True:
        print(log)
        if config.track:
            wandb.log(log)
        log = {}

        if iteration % config.eval_interval == 0:
            # Evaluate network.
            mean_return = None
            mean_cost = None
            # TODO: This can be used to test different beta_c configs
            if config.exploitation_beta_v < 0:
                # Do regular evaluation
                original_exploitation_beta = config.exploitation_beta_v
                config.exploitation_beta_v = 0.0
                rng_key, subkey = jax.random.split(rng_key)
                mean_return, mean_cost = evaluate(model, config, context, jax.random.split(subkey, num_devices))
                log.update({"regular mean_return": mean_return.item()})
                log.update({"regular mean_cost": mean_cost.item()})
                # And then do pessim_evaluation
                config.exploitation_beta_v = original_exploitation_beta
                rng_key, subkey = jax.random.split(rng_key)
                mean_return, mean_cost = evaluate(model, config, context, jax.random.split(subkey, num_devices))
                log.update({"pessimistic_evaluation mean_return": mean_return.item()})
                log.update({"pessimistic_evaluation mean_cost": mean_cost.item()})
            else:
                rng_key, subkey = jax.random.split(rng_key)
                mean_return, mean_cost = evaluate(model, config, context, jax.random.split(subkey, num_devices))
                log.update({"mean_return": mean_return.item()})
                log.update({"mean_cost": mean_cost.item()})

            if env.id == "safety_grid":
                # Evaluate value, cost, and uncertainties for all grid states
                grid_size = context.env.grid_size

                # Create all possible pgx states for the grid
                all_states = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Create a blank state
                        state = env.init(jax.random.PRNGKey(0))
                        obs = jnp.zeros_like(state.observation)
                        obs = obs.at[i, j].set(1)
                        state = state.replace(observation=obs)
                        all_states.append(state)
                all_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *all_states)

                # Forward pass
                model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
                model_params, model_state = model_0
                network_output, _ = context.forward.apply(
                    model_params, model_state, all_states.observation, is_training=False
                )

                value = network_output.value.reshape(grid_size, grid_size)
                value_uncertainty = network_output.value_epistemic_variance.reshape(grid_size, grid_size)
                cost_value = network_output.cost_value.reshape(grid_size, grid_size)
                cost_uncertainty = network_output.cost_value_epistemic_variance.reshape(grid_size, grid_size)

                # Get policies from policy networks
                exploration_action = (
                    jnp.argmax(network_output.exploration_logits, axis=-1)
                    .astype(jnp.int32)
                    .reshape(grid_size, grid_size)
                )
                exploitation_action = (
                    jnp.argmax(network_output.exploitation_logits, axis=-1)
                    .astype(jnp.int32)
                    .reshape(grid_size, grid_size)
                )

                # Get policies from tree root
                rng_key, subkey = jax.random.split(rng_key)
                safety_grid_recurrent_fn = get_epistemic_recurrent_fn(
                    env=env,
                    forward=forward,
                    batch_size=network_output.exploitation_logits.shape[0],
                    exploration=False,
                    discount=config.discount,
                    two_players_game=config.two_players_game,
                )

                root = emctx.EpistemicRootFnOutput(
                    prior_logits=network_output.exploration_logits,  # type: ignore
                    value=network_output.value,  # type: ignore
                    value_epistemic_variance=network_output.value_epistemic_variance,  # type: ignore
                    embedding=all_states,  # type: ignore
                    beta_v=config.exploitation_beta_v * jnp.ones_like(network_output.value),  # type: ignore
                    beta_c=config.exploitation_beta_c * jnp.ones_like(network_output.value),  # type: ignore
                    cost_value=network_output.cost_value,
                    cost_value_epistemic_variance=network_output.cost_value_epistemic_variance,
                    cost_threshold=network_output.cost_epistemic_variance,
                )
                policy_output = emctx.epistemic_gumbel_muzero_policy(
                    params=model_0,
                    rng_key=subkey,
                    root=root,
                    recurrent_fn=safety_grid_recurrent_fn,
                    num_simulations=config.selfplay_simulations_per_step,
                    invalid_actions=jnp.zeros_like(network_output.exploration_logits),
                    qtransform=emctx.epistemic_qtransform_completed_by_mix_value,  # type: ignore
                    gumbel_scale=0.0,
                )

                emcts_action = policy_output.action.reshape(grid_size, grid_size)

                fig, axs = plt.subplots(1, 5, figsize=(20, 5))

                # Value Uncertainty as color, Value as text
                im0 = axs[0].imshow(value_uncertainty, cmap="plasma")
                axs[0].set_title("Value (text), Value Uncertainty (color)")
                plt.colorbar(im0, ax=axs[0])
                for row in range(grid_size):
                    for col in range(grid_size):
                        val = value[row, col]
                        # Green if reward, else default
                        color = "green" if env.reward_map[row, col] > 0 else "black"
                        axs[0].text(
                            col,
                            row,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=12,
                            fontweight="bold",
                        )

                # Cost Uncertainty as color, Cost as text
                im1 = axs[1].imshow(cost_uncertainty, cmap="plasma")
                axs[1].set_title("Cost (text), Cost Uncertainty (color)")
                plt.colorbar(im1, ax=axs[1])
                for row in range(grid_size):
                    for col in range(grid_size):
                        cval = cost_value[row, col]
                        # Red if cost, else default
                        color = "red" if env.cost_map[row, col] > 0 else "black"
                        axs[1].text(
                            col,
                            row,
                            f"{cval:.2f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=12,
                            fontweight="bold",
                        )

                # --- Action visualization helper ---
                def plot_action_arrows(ax, action_grid, title):
                    # Map: 0=U, 1=D, 2=L, 3=R
                    dx = jnp.zeros_like(action_grid, dtype=float)
                    dy = jnp.zeros_like(action_grid, dtype=float)
                    dx = dx.at[action_grid == 2].set(-1)  # Left
                    dx = dx.at[action_grid == 3].set(1)  # Right
                    dy = dy.at[action_grid == 0].set(-1)  # Up
                    dy = dy.at[action_grid == 1].set(1)  # Down
                    X, Y = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size))
                    ax.imshow(jnp.zeros_like(action_grid), cmap="gray", vmin=0, vmax=1, zorder=0)  # blank grid
                    ax.set_facecolor("white")
                    # Set minor ticks after imshow
                    ax.set_xticks(jnp.arange(-0.5, grid_size, 1), minor=True)
                    ax.set_yticks(jnp.arange(-0.5, grid_size, 1), minor=True)
                    # Draw grid lines on top
                    ax.grid(which="minor", color="#222", linestyle="-", linewidth=2, zorder=2)
                    # Draw arrows
                    ax.quiver(
                        X,
                        Y,
                        dx,
                        dy,
                        angles="xy",
                        scale_units="xy",
                        scale=2.5,
                        color="blue",
                        width=0.008,
                        zorder=3,
                        pivot="middle",
                    )
                    ax.set_title(title)
                    ax.set_xticks(jnp.arange(grid_size))
                    ax.set_yticks(jnp.arange(grid_size))
                    ax.set_xlim(-0.5, grid_size - 0.5)
                    ax.set_ylim(-0.5, grid_size - 0.5)
                    ax.invert_yaxis()

                # Plot exploration actions
                plot_action_arrows(axs[2], exploration_action, "Exploration Action (arrows)")
                # Plot exploitation actions
                plot_action_arrows(axs[3], exploitation_action, "Exploitation Action (arrows)")
                # Plot emcts actions
                plot_action_arrows(axs[4], emcts_action, "EMCTS Action (arrows)")

                plt.tight_layout()
                try:
                    wandb.log({"grid/value_cost_uncertainty_and_actions": wandb.Image(fig)})
                except Exception as e:
                    print(f"[Visualization] wandb logging failed: {e}")
                plt.close(fig)

            mean_returns_list.append(mean_return)
            mean_costs_list.append(mean_cost)
            frames_at_mean_returns_list.append(frames)
            jnp.save(file=complete_results_path + "/mean_returns_list.npy", arr=mean_returns_list)
            jnp.save(file=complete_results_path + "/mean_costs_list.npy", arr=mean_costs_list)
            jnp.save(file=complete_results_path + "/frames_at_mean_returns_list.npy", arr=frames_at_mean_returns_list)
            sys.stdout.flush()

        if iteration % config.checkpoint_interval == 0:
            # Save checkpoint.
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            with open(os.path.join(checkpoint_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        if iteration >= config.maximum_number_of_iterations:
            break
        iteration += 1

        # Selfplay.
        rng_key, subkey = jax.random.split(rng_key)
        if frames < config.learning_starts:
            last_states, states = uniformrandomplay(config, context, jax.random.split(subkey, num_devices))
        else:
            last_states, selfplay_output = selfplay(
                model, config, context, last_states, jax.random.split(subkey, num_devices)
            )

            (
                states,
                root_values,
                root_epistemic_stds,
                root_cost_values,
                root_cost_epistemic_stds,
                raw_values,
                cost_values,
                value_prediction_epistemic_variances,
                q_value_variances,
                q_cost_variances,
            ) = selfplay_output

            all_rewards = states.rewards.sum().item()
            if rewards_not_yet_observed_flag and all_rewards > 0:
                frames_to_first_reward = frames + config.selfplay_batch_size * config.selfplay_steps
                print(f"Observed first reward after frames: {frames_to_first_reward}")
                rewards_not_yet_observed_flag = False
                log.update({"frames_to_first_reward": frames_to_first_reward})

            if "safety" in config.env_class:
                all_costs = states.costs.sum().item()
                if costs_not_yet_observed_flag and all_costs > 0:
                    frames_to_first_cost = frames + config.selfplay_batch_size * config.selfplay_steps
                    print(f"Observed first cost after frames: {frames_to_first_cost}")
                    costs_not_yet_observed_flag = False
                    log.update({"frames_to_first_cost": frames_to_first_cost})

            if "safety" in config.env_class:
                log.update(
                    {
                        "mean_raw_value": raw_values.mean().item(),
                        "mean_root_value": root_values.mean().item(),
                        "mean_raw_cost_value": cost_values.mean().item(),
                        "mean_root_cost_value": root_cost_values.mean().item(),
                        "mean_ube": value_prediction_epistemic_variances.mean().item(),
                        "mean_root_epistemic_std": root_epistemic_stds.mean().item(),
                        "mean_root_cost_epistemic_std": root_cost_epistemic_stds.mean().item(),
                        "mean_root_max_child_epistemic_variance": q_value_variances.max(axis=-1).mean().item(),
                        "mean_root_max_child_cost_epistemic_variance": q_cost_variances.max(axis=-1).mean().item(),
                        "observed_rewards": all_rewards,
                    }
                )
            else:
                log.update(
                    {
                        "mean_raw_value": raw_values.mean().item(),
                        "mean_root_value": root_values.mean().item(),
                        "mean_ube": ube_predictions.mean().item(),
                        "mean_root_epistemic_std": root_epistemic_stds.mean().item(),
                        "mean_root_max_child_epistemic_variance": q_value_variances.max(axis=-1).mean().item(),
                        "observed_rewards": all_rewards,
                    }
                )

            if "deep_sea" in config.env_id:
                ube_predictions, unseen_states = debug_deep_sea(all_states_batch, model, context)  # type: ignore
                print(f"ube_predictions = \n" f"{ube_predictions}")
                print(f"unseen_states = \n" f"{unseen_states}")
                print(f"Number of seen states: = {(1 - unseen_states).sum().item()}")

        frame_diff = config.selfplay_batch_size * config.selfplay_steps
        frames += frame_diff

        # Add to buffer.
        # TODO: Check truncation (?)
        buffer_state = buffer_fn.add(
            buffer_state, jax.tree.map(lambda x: jnp.concatenate(jnp.swapaxes(x, 1, 2)), states)
        )
        if config.save_replay_buffer:
            rb_vault.write(buffer_state)

        value_loss_list = []
        cost_value_loss_list = []
        ube_loss_list = []
        exploitation_policy_loss_list = []
        exploration_policy_loss_list = []
        exploitation_policy_entropy_list = []
        exploration_policy_entropy_list = []
        batch_novelty_list = []

        # If the buffer doesn't have enough interactions yet, keep interacting, or learning shouldn't start yet
        if not buffer_fn.can_sample(buffer_state) or frames < config.learning_starts:
            log.update(
                {
                    "iteration": iteration,
                    "hours": (time.time() - start_time) / 3600,
                    "frames": frames,
                    "executing random moves until": max(config.learning_starts, config.min_replay_buffer_length),
                }
            )
            continue
        else:
            for _ in range(config.reanalyze_loops_per_selfplay):  # type: ignore
                assert buffer_fn.can_sample(buffer_state)

                # Reanalyze.
                rng_key, subkey = jax.random.split(rng_key)
                batch = buffer_fn.sample(buffer_state, subkey)
                reanalyze_output = reanalyze(
                    model,
                    config,
                    context,
                    jax.tree.map(lambda x: jnp.array(jnp.split(x, num_devices)), batch.experience),
                    jax.random.split(subkey, num_devices),
                )

                (
                    _observation,
                    _next_observation,
                    value_target,
                    cost_value_target,
                    ube_target,
                    _exploration_policy_target,
                    _exploitation_policy_target,
                ) = reanalyze_output
                log.update(
                    {
                        "train/mean_value_target": value_target.mean().item(),
                        "train/mean_ube_target": ube_target.mean().item(),
                        "train/mean_cost_value_target": cost_value_target.mean().item(),
                    }
                )

                # Make sure that gradients don't pass through the targets
                reanalyze_output = jax.tree_util.tree_map(jax.lax.stop_gradient, reanalyze_output)
                if config.debug and "deep-sea" in config.env_id:
                    print(
                        f"Number of value_targets = 1.0 in reanalyze output: "
                        f"{jnp.where(reanalyze_output.value_target == 1.0, reanalyze_output.value_target, 0.0).sum().item()}"
                    )

                (
                    model,
                    opt_state,
                    priority_score,
                    exploitation_policy_entropy,
                    exploration_policy_entropy,
                    value_loss,
                    cost_value_loss,
                    ube_loss,
                    exploitation_policy_loss,
                    exploration_policy_loss,
                    batch_novelty,
                ) = train(model, opt_state, context, reanalyze_output)
                priority_score = jnp.concatenate(priority_score)

                # Adjust priorities.
                buffer_state = buffer_fn.set_priorities(buffer_state, batch.indices, priority_score)

                # Keep track of losses for logging.
                # `.mean()` because we get a separate loss per device.
                value_loss_list.append(value_loss.mean().item())
                cost_value_loss_list.append(cost_value_loss.mean().item())
                ube_loss_list.append(ube_loss.mean().item())
                exploitation_policy_loss_list.append(exploitation_policy_loss.mean().item())
                exploration_policy_loss_list.append(exploration_policy_loss.mean().item())
                exploitation_policy_entropy_list.append(exploitation_policy_entropy.mean().item())
                exploration_policy_entropy_list.append(exploration_policy_entropy.mean().item())
                batch_novelty_list.append(batch_novelty.mean().item())

            # Calculate "replay buffer uniqueness" (according to hash set).
            _model_params, model_state = model
            binary_set = model_state[context.hash_path]["binary_set"][0]  # index 0 because of device dimension
            new_set_bits = jax.lax.population_count(binary_set).sum().item()  # i.e. "seen" states
            set_bit_diff = new_set_bits - set_bits
            set_bits = new_set_bits

            log.update(
                {
                    "iteration": iteration,
                    "hours": (time.time() - start_time) / 3600,
                    "frames": frames,
                    "train/value_loss": sum(value_loss_list) / len(value_loss_list),
                    "train/cost_value_loss": sum(cost_value_loss_list) / len(cost_value_loss_list),
                    "train/ube_loss": sum(ube_loss_list) / len(ube_loss_list),
                    "train/exploitation_policy_loss": sum(exploitation_policy_loss_list)
                    / len(exploitation_policy_loss_list),
                    "train/exploration_policy_loss": sum(exploration_policy_loss_list)
                    / len(exploration_policy_loss_list),
                    "train/mean_exploitation_policy_entropy": sum(exploitation_policy_entropy_list)
                    / len(exploitation_policy_entropy_list),
                    "train/mean_exploration_policy_entropy": sum(exploration_policy_entropy_list)
                    / len(exploration_policy_entropy_list),
                    "hash/set_bits": set_bits,
                    "hash/new_bits_ratio": set_bit_diff / frame_diff,
                    "hash/batch_novelty": sum(batch_novelty_list) / len(batch_novelty_list),
                }
            )


if __name__ == "__main__":
    main()
