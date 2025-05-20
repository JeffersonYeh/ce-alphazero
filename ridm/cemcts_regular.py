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
    parent: Optional['Node'] = None
    children: List[Optional['Node']] = field(init=False)
    recent_action: Optional[jnp.ndarray] = None
    q_values: jnp.ndarray = field(init=False)
    variance_r_hat: Optional[jnp.ndarray] = field(default=None)
    variance_v_hat: Optional[jnp.ndarray] = field(default=None)


    def __post_init__(self):
        n_actions = len(self.state.legal_action_mask)
        self.visit_count = jnp.zeros(n_actions, dtype=jnp.int32)
        self.children = [None] * n_actions
        self.q_values = jnp.zeros(n_actions, dtype=jnp.float32)

epsilon = jnp.array(1e-8, dtype=jnp.float32)

def value_function_dummy(node: Node) -> (jnp.ndarray, jnp.ndarray): # V hat
    mu = jnp.array(1.0, dtype=jnp.float32)
    variance = jnp.array(1.0, dtype=jnp.float32)
    return (mu, variance)

def reward_function_dummy(node: Node) -> (jnp.ndarray, jnp.ndarray): # R hat
    mu = jnp.array(2.0, dtype=jnp.float32)
    variance = jnp.array(2.0, dtype=jnp.float32)
    return (mu, variance)


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
        value_function: Callable[[Node], Tuple[jnp.ndarray, jnp.ndarray]], # V hat
        reward_function: Callable[[Node], Tuple[jnp.ndarray, jnp.ndarray]], # R hat
        gamma: jnp.ndarray = jnp.array([0.999]),
        c_uct: float = 1.0,
        beta: float = 0.0,
        budget: int = 100,
    ):
        self.env = env
        self.value_fn = value_fn
        self.reward_fn = reward_fn
        self.uncertainty_fn = uncertainty_fn
        self.safety_critic = safety_critic
        self.value_function = value_function
        self.reward_function = reward_function
        self.gamma = gamma
        self.c_uct = c_uct
        self.beta = beta
        self.budget = budget
        self.tree = {}

    def f(self, node: Node, action: Array) -> Node:
        if node.children[action] is None:
            new_state = self.env.step(node.state, action)
            node.recent_action = action
            new_node = Node(
                id=hash_state(new_state),
                state=new_state,
                parent=node
            )
            node.children[action] = new_node
            return new_node
        else:
            return node.children[action]

    def nu(self, node: Node, action: Array, key) -> (Array, Array):
        current = node
        result = jnp.array([0])
        j = jnp.array([0])
        while True:
            result += jnp.power(self.gamma, j) * current.state.rewards
            j += 1
            child = current.children[action]
            if child is None:  # Rollout from here
                result += self.rollout(current, key)



    def rollout(self, node: Node, key) -> Array: # Return (mu, variance)
        rewards = []
        current_state = node.state
        while True:
            stricter_legal_mask = self.safety_critic(current_state.observation, current_state.legal_action_mask)
            if jnp.all(~stricter_legal_mask):
                break
            true_indices = jnp.where(stricter_legal_mask)[0]  # Shape: (num_true,)
            rand_idx = jax.random.randint(key, (), 0, true_indices.shape[0])
            uniform_random_action = true_indices[rand_idx]
            new_state = self.env.step(current_state, uniform_random_action)
            rewards.append(new_state.rewards)
            terminated_all = bool(jnp.all(new_state.terminated))
            truncated_all = bool(jnp.all(new_state.truncated))  # if it's a JAX array

            if terminated_all or truncated_all:
                break
            else:
                current_state = new_state
        reward_array = jnp.stack(rewards)
        mean = jnp.mean(reward_array, axis=0)
        # variance = jnp.var(reward_array, axis=0)
        return mean



    def emcts(self, state: pgx.State, key: jax.random.PRNGKey) -> Array:
        # tree = {}  # Dict[state_hash] = Node
        id = hash_state(state)
        root = Node(id=id, state=state)
        for _ in range(self.budget):
            self.select(root, self.beta)

        return sample_action(root.visit_count, key)

    def select(self, node, beta, key: jax.random.PRNGKey):
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
            self.expand(node, best_action, key)
        else:
            self.select(self.f(node, best_action), beta, key)


    def expand(self, node: Node, best_action: Array, key: jax.random.PRNGKey) -> None:
        new_node = self.f(node=node, action=best_action)
        value_mu, value_var = self.value_function(new_node)
        reward_mu, reward_var = self.reward_function(new_node)
        new_node.variance_r_hat = reward_var
        new_node.variance_v_hat = value_var


        node.recent_action = best_action


    def backup(self, node: Node, v_hat: Array, v_hat_var: Array, key: jax.random.PRNGKey) -> None:
        previous_node = node.parent
        previous_action = previous_node.recent_action
        nu_value = self.nu(previous_node, previous_action, key)
        pass