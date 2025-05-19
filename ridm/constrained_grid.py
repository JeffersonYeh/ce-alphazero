from dataclasses import dataclass
import jax
import jax.numpy as jnp
from pgx import Env, State
from typing import Optional, Tuple
from flax import struct

@struct.dataclass
class ConstrainedGridState(State):
    current_player: jnp.ndarray
    observation: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    legal_action_mask: jnp.ndarray
    _step_count: jnp.ndarray
    position: jnp.ndarray
    right_moves: jnp.ndarray

    @property
    def env_id(self) -> str:
        return "constrained_grid"

class ConstrainedGridEnv(Env):
    def __init__(self):
        self.grid_size = 5
        self.max_right_moves = 3

    def _init(self, key: jax.random.PRNGKey) -> ConstrainedGridState:
        position = jnp.array(0)
        right_moves = jnp.array(0)
        observation = jnp.array([position, right_moves])
        legal_action_mask = jnp.array([True, True])  # [left, right]
        return ConstrainedGridState(
            current_player=jnp.array(0),
            observation=observation,
            rewards=jnp.zeros(1),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            legal_action_mask=legal_action_mask,
            _step_count=jnp.array(0),
            position=position,
            right_moves=right_moves
        )

    def _step(self, state: ConstrainedGridState, action: int, key: Optional[jax.random.PRNGKey]) -> ConstrainedGridState:
        position = state.position
        right_moves = state.right_moves
        reward = jnp.array(0.0)
        terminated = state.terminated

        def move_left():
            new_position = jnp.maximum(position - 1, 0)
            return new_position, right_moves, reward

        def move_right():
            can_move = right_moves < self.max_right_moves
            new_right_moves = right_moves + 1
            new_position = jnp.minimum(position + 1, self.grid_size - 1)
            new_reward = jnp.where(can_move, 1.0, 0.0)
            return new_position, new_right_moves, new_reward

        new_position, new_right_moves, reward = jax.lax.cond(
            action == 0,
            move_left,
            move_right
        )

        terminated = new_position == (self.grid_size - 1)
        legal_action_mask = jnp.array([
            new_position > 0,
            new_right_moves < self.max_right_moves
        ])

        observation = jnp.array([new_position, new_right_moves])

        return ConstrainedGridState(
            current_player=state.current_player,
            observation=observation,
            rewards=jnp.array([reward]),
            terminated=jnp.array(terminated),
            truncated=state.truncated,
            legal_action_mask=legal_action_mask,
            _step_count=state._step_count + 1,
            position=new_position,
            right_moves=new_right_moves
        )

    def _observe(self, state: ConstrainedGridState, player_id: jnp.ndarray) -> jnp.ndarray:
        return state.observation

    @property
    def id(self) -> str:
        return "constrained_grid"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def num_players(self) -> int:
        return 1
