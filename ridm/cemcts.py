from typing import Any, Tuple, Callable, Optional
from flax.struct import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import pgx

@dataclass
class Node:
    visit_count: jnp.ndarray
    q_value: jnp.ndarray
    value_variance: jnp.ndarray
    children: Any  # Dict[int, Node] or PyTree structure


class Cemcts:
    def __init__(
        self,
        env: pgx.Env,
        value_fn: Callable[[Any], float],           # V̂
        reward_fn: Callable[[Any, int], float],     # R̂
        uncertainty_fn: Callable[[Any], float],     # V[V̂], V[R̂]
        gamma: float = 1.0,
        c_uct: float = 1.0,
        beta: float = 0.0,
        max_depth: int = 50,
        budget: int = 100,
    ):
        self.env = env
        self.value_fn = value_fn
        self.reward_fn = reward_fn
        self.uncertainty_fn = uncertainty_fn
        self.gamma = gamma
        self.c_uct = c_uct
        self.beta = beta
        self.max_depth = max_depth
        self.budget = budget

    def search(self, state: pgx.State, key: jax.random.PRNGKey) -> int:
        """Top-level EMCTS search call."""
        tree = {}  # Dict[state_hash] = Node
        for _ in range(self.budget):
            key, subkey = jax.random.split(key)
            self._select(state, depth=0, key=subkey, tree=tree)

        # Extract action distribution from root
        legal_actions = jnp.where(state.legal_action_mask)[0]
        root_node = tree[self._hash_state(state)]
        visits = root_node.visit_count
        probs = visits / jnp.sum(visits)
        return int(jax.random.choice(key, legal_actions, p=probs))

    def _select(self, state, depth, key, tree):
        state_hash = self._hash_state(state)

        if state_hash not in tree:
            return self._expand(state, tree)

        node = tree[state_hash]
        legal_actions = jnp.where(state.legal_action_mask)[0]

        def uct(a):
            q = node.q_value[a]
            n = node.visit_count[a]
            total_n = jnp.sum(node.visit_count)
            ucb = self.c_uct * jnp.sqrt(jnp.log(total_n + 1) / (n + 1))
            return q + ucb

        action = legal_actions[jnp.argmax(jnp.array([uct(a) for a in legal_actions]))]
        next_state = self.env.step(state, action)
        ret, unc = self._select(next_state, depth + 1, key, tree)

        # Backup step
        ret_val = self.reward_fn(state, action) + self.gamma * ret
        ret_unc = self.uncertainty_fn(state) + self.gamma**2 * unc

        node.visit_count = node.visit_count.at[action].add(1)
        node.q_value = node.q_value.at[action].add((ret_val - node.q_value[action]) / node.visit_count[action])
        node.value_variance = node.value_variance.at[action].set(ret_unc)

        return ret_val, ret_unc

    def _expand(self, state, tree):
        legal_actions = jnp.where(state.legal_action_mask)[0]
        n_actions = self.env.num_actions

        visit_count = jnp.zeros(n_actions)
        q_value = jnp.zeros(n_actions)
        value_variance = jnp.zeros(n_actions)

        node = Node(
            visit_count=visit_count,
            q_value=q_value,
            value_variance=value_variance,
            children=None
        )

        tree[self._hash_state(state)] = node

        value = self.value_fn(state)
        uncertainty = self.uncertainty_fn(state)

        return value, uncertainty

    def _hash_state(self, state: pgx.State) -> int:
        # Naive hash for immutable JAX state — replace with better fingerprinting if needed
        return hash(state.observation.tobytes())
