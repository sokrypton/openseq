import jax
import random

def get_random_key(seed: int = None):
  """
  Get a JAX PRNGKey.

  Args:
    seed (int, optional): Seed for the random number generator.
                          If None, a random seed is generated.

  Returns:
    jax.random.PRNGKey: A JAX PRNG key.
  """
  if seed is None:
    seed = random.randint(0, 2**31 - 1) # Ensure seed is within JAX's typical int32 range
  return jax.random.PRNGKey(seed)

# The stateful KEY class from laxy.py is intentionally omitted
# in favor of explicit key management in JAX.
