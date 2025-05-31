from typing import Optional
import chex
import haiku as hk
import jax

ForwardFn = hk.TransformedWithState
Model = tuple[hk.MutableParams, hk.MutableState]

Array = jax.Array
PRNGKey = chex.PRNGKey

Observation = Array

# Value = Array
# ValueVariance = Array
# ExploitationPolicy = Array
# ExplorationPolicy = Array
# RewardVariance = Array
# NetworkOutput = tuple[Value, ValueVariance, ExploitationPolicy, ExplorationPolicy, RewardVariance]


@chex.dataclass(frozen=True)
class NetworkOutput:
    exploitation_logits: Array
    exploration_logits: Array
    value: Array
    value_epistemic_variance: Array
    reward_epistemic_variance: Array

    cost_value: Optional[Array]
    cost_value_epistemic_variance: Optional[Array]
    cost_epistemic_variance: Optional[Array]
