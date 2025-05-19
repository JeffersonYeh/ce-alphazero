import jax
from pgx import make
from constrained_grid import ConstrainedGridEnv
import jax.numpy as jnp

# Initialize environment
env = ConstrainedGridEnv()
key = jax.random.PRNGKey(0)
state = env.init(key)

# Simulate steps
for _ in range(10):
    action = jax.random.choice(key, jnp.array([0, 1]))  # Randomly choose left or right
    state = env.step(state, action)
    print(f"Position: {state.position}, Right Moves: {state.right_moves}, Reward: {state.rewards[0]}")
    if state.terminated:
        print("Reached the goal!")
        break
