import pytest
import jax
import jax.numpy as jnp
from openseq.models import MRF
# from openseq.utils.random import get_random_key # Not needed for fixed params

class TestMRFMinimal:
    def test_minimal_mrf_get_w(self):
        L_test = 3
        A_test = 2

        # Simplified config for MRF (k=1)
        model = MRF(L=L_test, A=A_test, k=1, lam=0.01, learning_rate=0.01, seed=0)

        # Create small, fixed JAX arrays for w and b
        key = jax.random.PRNGKey(42) # Use a fixed key for reproducibility if random needed
        # For maximum simplicity, let's use jnp.ones or arange if shape allows
        # w_param = jax.random.normal(key, (L_test, A_test, L_test, A_test)) * 0.1
        # b_param = jax.random.normal(key, (L_test, A_test)) * 0.1

        w_val = 0.1 * jnp.arange(L_test*A_test*L_test*A_test).reshape(L_test, A_test, L_test, A_test)
        b_val = 0.05 * jnp.arange(L_test*A_test).reshape(L_test, A_test)

        model.load_parameters({'w': w_val, 'b': b_val})

        # Assertions directly from the problematic test
        assert model.params is not None, "model.params should be a dict after load_parameters"
        assert 'w' in model.params, "'w' key should be in model.params"
        assert model.params['w'] is not None, "model.params['w'] should be a JAX array, not None"
        assert model.params['w'].shape == (L_test, A_test, L_test, A_test)

        print(f"DEBUG: In test, before get_w, model.params['w'].sum() = {model.params['w'].sum()}")

        w_eff = model.get_w()

        print(f"DEBUG: In test, after get_w, w_eff is None: {w_eff is None}")
        if w_eff is not None:
            print(f"DEBUG: In test, after get_w, w_eff.sum() = {w_eff.sum()}")


        assert w_eff is not None, "model.get_w() should not return None in minimal test"
        assert w_eff.shape == (L_test, A_test, L_test, A_test)

        # Basic check if processing happened (values might change due to norm/symm)
        # This is just to see if it's a JAX array that underwent some ops
        assert not jnp.allclose(w_eff, w_val) or L_test==1 # If L=1, symm might not change it much from zeros

        print("Minimal MRF get_w test completed its assertions.")
