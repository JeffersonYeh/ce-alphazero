from typing import NamedTuple, Optional, Type

import chex

import haiku as hk
import jax
import jax.numpy as jnp
import optax  # type: ignore
import pgx  # type: ignore

import cemctx as emctx
from config import Config
from network.c_minatar import ConstraintEpistemicMinatarAZNet
from network.fully_connected import EpistemicFullyConnectedAZNet
from network.hashes import LCGHash, SimHash, XXHash
from network.minatar import EpistemicMinatarAZNet
from network.resnet import EpistemicResidualAZNet
from type_aliases import Array, ForwardFn, Model, NetworkOutput, Observation, PRNGKey


class Context(NamedTuple):
    """Context which stays the same throughout training."""

    env: pgx.Env
    devices: list[jax.Device]
    forward: ForwardFn
    selfplay_recurrent_fn: emctx.EpistemicRecurrentFn
    reanalyze_recurrent_fn: emctx.EpistemicRecurrentFn
    evaluation_recurrent_fn: emctx.EpistemicRecurrentFn
    optimizer: optax.GradientTransformation
    scale_uncertainty_losses: float
    hash_path: str
    exploration_beta: float
    max_ube: float
    max_ube_cost: float
    weigh_losses: bool
    loss_weighting_temperature: float
    directed_exploration: bool

    # HACK: Should be fine since there will only ever be one `Context`.
    def __hash__(self):
        return 1


def get_network(env: pgx.Env, config: Config) -> hk.Module:
    hash_class: Type
    match config.hash_class:
        case "LCGHash":
            hash_class = LCGHash
        case "SimHash":
            hash_class = SimHash
        case "XXHash":
            hash_class = XXHash
    if "safety" in config.env_class:
        return ConstraintEpistemicMinatarAZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            max_ube=config.max_ube,
            max_ube_cost=config.max_ube_cost,
            discount=config.discount,
            hidden_layers_size=config.linear_layer_size,
            hash_class=hash_class,
            max_epistemic_variance_reward=config.max_epistemic_variance_reward,
        )
    elif "minatar" in config.env_id:
        return EpistemicMinatarAZNet(
            num_actions=env.num_actions,
            num_channels=config.num_channels,
            max_ube=config.max_ube,
            discount=config.discount,
            hidden_layers_size=config.linear_layer_size,
            hash_class=hash_class,
            max_epistemic_variance_reward=config.max_epistemic_variance_reward,
        )
    elif "deep_sea" in config.env_id or "subleq" in config.env_id:
        if str(config.env_id).startswith("subleq"):
            s = str(config.env_id).removeprefix("subleq-")
            word_size = int(s) if s.isnumeric() else 256
            subleq = True
        else:
            word_size = 0
            subleq = False
        return EpistemicFullyConnectedAZNet(
            num_actions=env.num_actions,
            discount=config.discount,
            hash_class=hash_class,
            word_size=word_size,
            hash_io=subleq and config.subleq_hash_only_io,
            layer_size=config.linear_layer_size,
        )
    else:
        # TODO: Add missing hyper-params to config (e.g. hash_bits, hidden_layers, etc.)
        # TODO: Set the hyperparameters here correctly
        return EpistemicResidualAZNet(
            num_actions=env.num_actions,
            hash_class=hash_class,
        )


# Set up the training model and optimizer.
def get_forward_fn(env: pgx.Env, config: Config) -> ForwardFn:
    def forward_fn(x: Observation, is_training: bool = True, update_hash: bool = False) -> NetworkOutput:
        if "safety" in config.env_class:
            net = get_network(env, config)
            network_output = net(
                x, is_training=is_training, test_local_stats=False, update_hash=update_hash
            )  # type: ignore
            return network_output
        else:
            net = get_network(env, config)
            network_output = net(
                x, is_training=is_training, test_local_stats=False, update_hash=update_hash
            )  # type: ignore
            return network_output

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def get_epistemic_recurrent_fn(
    env: pgx.Env,
    forward: ForwardFn,
    batch_size: int,
    exploration: bool,
    discount: float,
    two_players_game: bool,
) -> emctx.EpistemicRecurrentFn:
    def epistemic_recurrent_fn(
        model: Model,
        rng_key: PRNGKey,
        action: Array,
        state: pgx.State,
    ) -> tuple[emctx.EpistemicRecurrentFnOutput, pgx.State]:
        model_params, model_state = model

        current_player = state.current_player
        keys = jax.random.split(rng_key, batch_size)
        state = jax.vmap(env.step)(state, action, keys)
        value: Array
        network_output, _ = forward.apply(model_params, model_state, state.observation, is_training=False)

        exploitation_logits = network_output.exploitation_logits
        exploration_logits = network_output.exploration_logits
        value = network_output.value
        value_epistemic_variance = network_output.value_epistemic_variance
        _reward_epistemic_variance = network_output.reward_epistemic_variance
        cost_value = network_output.cost_value
        cost_value_epistemic_variance = network_output.cost_value_epistemic_variance
        _cost_epistemic_variance = network_output.cost_epistemic_variance

        logits = jax.lax.cond(exploration, lambda: exploration_logits, lambda: exploitation_logits)

        # Subtract max from logits to improve numerical stability.
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        # Mask invalid actions with minimum float.
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)  # type: ignore

        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        value_epistemic_variance = jnp.where(state.terminated, 0.0, value_epistemic_variance)
        batched_discount = jnp.ones_like(value) * discount  # For two player games -1.0 *
        batched_discount = jax.lax.cond(two_players_game, lambda: batched_discount * -1.0, lambda: batched_discount)  # type: ignore
        batched_discount = jnp.where(state.terminated, 0.0, batched_discount)

        cost = state.costs[jnp.arange(state.costs.shape[0]), current_player]
        cost_value = jnp.where(state.terminated, 0.0, cost_value)
        cost_value_epistemic_variance = jnp.where(state.terminated, 0.0, cost_value_epistemic_variance)

        epistemic_recurrent_fn_output = emctx.EpistemicRecurrentFnOutput(
            reward=reward,  # type: ignore
            # NOTE: We have a known reward model, so we pass 0 reward uncertainty.
            reward_epistemic_variance=jnp.zeros_like(reward),  # type: ignore
            discount=batched_discount,  # type: ignore
            prior_logits=logits,  # type: ignore
            value=value,  # type: ignore
            value_epistemic_variance=value_epistemic_variance,  # type: ignore
            cost=cost,
            cost_epistemic_variance=jnp.zeros_like(
                cost
            ),  # NOTE: We have a known cost model, so we pass 0 cost uncertainty.
            cost_value=cost_value,
            cost_value_epistemic_variance=cost_value_epistemic_variance,
        )
        return epistemic_recurrent_fn_output, state

    return epistemic_recurrent_fn  # type: ignore
