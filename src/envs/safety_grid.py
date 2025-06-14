import chex
import jax
import jax.numpy as jnp
import pgx  # type: ignore
import pgx.core as core
from pgx._src.struct import dataclass  # type: ignore

from type_aliases import Array, PRNGKey
from typing import Literal, Optional

ENV_ID = "safety_grid"

# MAP_SIZE = jnp.int32(6)  # MAP_SIZE**2 must be divisible by 4 due to hashing function used
# REWARD_MAP = jnp.float32(
#     [
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#     ]
# )

# REWARD_LOC = jnp.square(MAP_SIZE) - 1

# COST_MAP = jnp.float32(
#     [
#         [0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0],
#     ]
# )

MAP_SIZE = jnp.int32(4)  # MAP_SIZE**2 must be divisible by 4 due to hashing function used
REWARD_MAP = jnp.float32([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

REWARD_LOCS = jnp.argwhere(REWARD_MAP > 0)

COST_MAP = jnp.float32([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

COST_THRESHOLD = jnp.float32(0.0)


@dataclass
class SafetyGridState(pgx.State):

    current_player: Array = jnp.int32(0)  # single player
    observation: Array = jnp.zeros(
        0, dtype=jnp.bool
    )  # The shape depends on the size of the grid. (only true at position of agent)

    rewards: Array = jnp.zeros(1, dtype=jnp.float32)
    terminated: Array = jnp.bool(False)
    truncated: Array = jnp.bool(False)
    legal_action_mask: Array = jnp.ones(4, dtype=jnp.bool)  # actions are U=0, D=1, L=2, R=3
    costs: Array = jnp.zeros(1, dtype=jnp.float32)

    _step_count: Array = jnp.int32(0)  # how many steps have been taken
    _position: Array = jnp.int32([0, 0])  # shape=(2,) - (x, y) start at (0, 0)
    _cum_costs: Array = jnp.zeros(1, dtype=jnp.float32)

    @property
    def env_id(self) -> pgx.EnvId:
        """Environment id = safety_grid"""
        return ENV_ID  # type: ignore


class SafetyGrid(pgx.Env):
    def __init__(
        self,
        grid_size=MAP_SIZE,
        reward_map=REWARD_MAP,
        cost_map=COST_MAP,
        cost_threshold=COST_THRESHOLD,
        reward_locs=REWARD_LOCS,
        max_steps=40,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reward_map = reward_map
        self.reward_locs = reward_locs
        self.cost_map = cost_map
        self.cost_threshold = cost_threshold

    @property
    def id(self) -> pgx.EnvId:
        """Environment id."""
        return ENV_ID  # type: ignore

    @property
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        return "0.0.1"

    def step(
        self,
        state: core.State,
        action: Array,
        key: Optional[Array] = None,
    ) -> core.State:
        """Step function."""
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards), costs=jnp.zeros_like(state.costs)),  # type: ignore
            lambda: self._step(state.replace(_step_count=state._step_count + 1), action, key),  # type: ignore
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),  # type: ignore
            lambda: state,
        )

        observation = self.observe(state)
        state = state.replace(observation=observation)  # type: ignore

        return state

    @property
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        return 1

    def _init(self, key: PRNGKey) -> SafetyGridState:
        observation = jnp.zeros([self.grid_size, self.grid_size], dtype=jnp.bool)
        observation = observation.at[..., 0, 0].set(True)  # Initial location is in the top-left.
        return SafetyGridState(observation=observation)

    def move(self, position: Array, direction: Literal["U", "D", "L", "R", "N"]):

        if direction == "U":
            position = position.at[1].set(jnp.maximum(position[1] - 1, 0))
        elif direction == "D":
            position = position.at[1].set(jnp.minimum(position[1] + 1, self.grid_size - 1))
        elif direction == "L":
            position = position.at[0].set(jnp.maximum(position[0] - 1, 0))
        elif direction == "R":
            position = position.at[0].set(jnp.minimum(position[0] + 1, self.grid_size - 1))
        else:
            pass

        return position

    def move_down(self, position: Array):
        position[1] = jnp.maximum(position[1] - 1, 0)

        return position

    # TODO: Make sure that terminated states are not updated
    def _step(self, state: SafetyGridState, action: Array, key: PRNGKey) -> SafetyGridState:
        assert isinstance(state, SafetyGridState)
        assert action.ndim == 0  # check action is a number =>  U=0, D=1, L=2, R=3

        # update position
        new_position = state._position

        new_position = jax.lax.cond(
            action == 0, lambda pos: self.move(pos, "U"), lambda pos: self.move(pos, "N"), operand=new_position
        )

        new_position = jax.lax.cond(
            action == 1, lambda pos: self.move(pos, "D"), lambda pos: self.move(pos, "N"), operand=new_position
        )

        new_position = jax.lax.cond(
            action == 2, lambda pos: self.move(pos, "L"), lambda pos: self.move(pos, "N"), operand=new_position
        )

        new_position = jax.lax.cond(
            action == 3, lambda pos: self.move(pos, "R"), lambda pos: self.move(pos, "N"), operand=new_position
        )

        # update observation
        new_observation = jnp.zeros_like(state.observation)
        new_observation = new_observation.at[..., new_position[0], new_position[1]].set(
            True
        )  # Initial location is in the top-left.

        pos_flat_idx = jnp.argmax(new_observation)
        pos_r, pos_c = jnp.unravel_index(pos_flat_idx, new_observation.shape)

        # update reward
        new_rewards = jnp.expand_dims(self.reward_map[pos_r, pos_c], axis=0)

        # update cost
        new_costs = jnp.expand_dims(self.cost_map[pos_r, pos_c], axis=0)

        # update cum cost
        new_cum_costs = state._cum_costs + new_costs

        # update termination
        new_terminated = (
            (state._step_count >= self.max_steps)
            | (jnp.any(jnp.all(self.reward_locs == jnp.array([pos_r, pos_c]), axis=1)))
            | (new_cum_costs[0] > self.cost_threshold)
        )
        new_terminated = jnp.bool_(new_terminated)

        return state.replace(  # type: ignore
            observation=new_observation,
            _position=new_position,
            rewards=new_rewards,
            terminated=new_terminated,
            costs=new_costs,
            _cum_costs=new_cum_costs,
        )

    def _observe(self, state: pgx.State, player_id: Array) -> Array:
        assert isinstance(state, SafetyGridState)
        return state.observation
