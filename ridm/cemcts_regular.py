from typing import Any, Tuple, Callable, Optional
from flax.struct import dataclass, field
import jax
import jax.numpy as jnp
import pgx
from typing import List, Optional
import jax.random as random
from jax import Array


@dataclass
class Node:
    id: int
    state: pgx.State
    visit_count: jnp.ndarray = field(init=False)
    parent: Optional[int] = None
    children: List[Optional['Node']] = field(init=False)

    def __post_init__(self):
        n_actions = len(self.state.legal_action_mask)
        self.visit_count = jnp.zeros(n_actions, dtype=jnp.int32)
        self.children = [None] * n_actions

epsilon = jnp.array(1e-8, dtype=jnp.float32)

def hash_state(state: pgx.State) -> int:
    # Naive hash for immutable JAX state — replace with better fingerprinting if needed
    return hash(state.observation.tobytes())

def q_value(node: Node, action: int, beta: float) -> float:
    return 1.5

def bonus(node: Node, action: int, c_uct) -> float:
    s = jnp.sum(node.visit_count)
    return c_uct * jnp.sqrt((2 * jnp.log(s)) / (node.visit_count.at[action] + epsilon))

def safety_critic_dummy(observation: jnp.ndarray, legal_action_mask: jnp.ndarray) -> jnp.ndarray:
    return legal_action_mask


def sample_action(visit_count: jnp.ndarray, key: jax.random.PRNGKey) -> Array:
    total = jnp.sum(visit_count)
    probs = jax.lax.cond(
        total > 0,
        lambda _: visit_count / total,
        lambda _: jnp.ones_like(visit_count) / len(visit_count),
        operand=None,
    )
    return random.choice(key, a=len(visit_count), p=probs)


class Cemcts:
    def __init__(
        self,
        env: pgx.Env,
        value_fn: Callable[[Any], float],           # V̂
        reward_fn: Callable[[Any, int], float],     # R̂
        uncertainty_fn: Callable[[Any], float],     # V[V̂], V[R̂]
        safety_critic: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], # (observation, legal_mask) -> stricter_legal_mask
        gamma: float = 1.0,
        c_uct: float = 1.0,
        beta: float = 0.0,
        budget: int = 100,
    ):
        self.env = env
        self.value_fn = value_fn
        self.reward_fn = reward_fn
        self.uncertainty_fn = uncertainty_fn
        self.safety_critic = safety_critic
        self.gamma = gamma
        self.c_uct = c_uct
        self.beta = beta
        self.budget = budget
        self.tree = {}

    def f(self, node: Node, action: Array) -> Node:
        if node.children[action] is None:
            new_state = self.env.step(node.state, action)
            new_node = Node(
                id=hash_state(new_state),
                state=new_state,
                parent=node.id
            )
            node.children[action] = new_node
            # self.tree[new_node.id] = new_node
            return new_node
        else:
            return node.children[action]

    def emcts(self, state: pgx.State, key: jax.random.PRNGKey) -> Array:
        # tree = {}  # Dict[state_hash] = Node
        id = hash_state(state)
        root = Node(id=id, state=state)
        for _ in range(self.budget):
            self.select(root, self.beta)

        return sample_action(root.visit_count, key)

    def select(self, node, beta):
        stricter_legal_mask = self.safety_critic(node.state.observation, node.state.legal_action_mask) # Safety critic
        num_actions = stricter_legal_mask.shape[0]
        scores = jnp.zeros(num_actions, dtype=jnp.float32)

        for action in range(num_actions):
            if node.state.legal_action_mask[action]:
                q = q_value(node, action, beta)
                b = bonus(node, action, self.c_uct)
                scores = scores.at[action].set(q + b)

        best_action = jnp.argmax(scores)

        if node.children[best_action] is None:
            self.expand(node, best_action)
        else:
            self.select(self.f(node, best_action), beta)


    def expand(self, node: Node, best_action: Array) -> None:
        pass