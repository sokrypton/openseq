import pytest
import jax
import jax.numpy as jnp
from openseq.models.simple_mrf import SimpleMRF

class TestSimpleMRF:
    def test_simple_mrf_get_w_direct_return(self):
        L_test = 3
        A_test = 2

        model = SimpleMRF(L=L_test, A=A_test)

        # Create small, fixed JAX arrays for w and b
        w_val = 0.1 * jnp.arange(L_test*A_test*L_test*A_test, dtype=jnp.float32).reshape(L_test, A_test, L_test, A_test)
        b_val = 0.05 * jnp.arange(L_test*A_test, dtype=jnp.float32).reshape(L_test, A_test)

        print(f"\nDEBUG (SimpleMRF Test): Initial model.params['w'].sum() from __init__: {model.params['w'].sum()}")

        model.load_parameters({'w': w_val, 'b': b_val})

        assert model.params is not None, "model.params should be a dict after load_parameters"
        assert 'w' in model.params, "'w' key should be in model.params"
        assert model.params['w'] is not None, "model.params['w'] should be a JAX array, not None"
        assert model.params['w'].shape == (L_test, A_test, L_test, A_test)
        print(f"DEBUG (SimpleMRF Test): In test, before get_w, model.params['w'].sum() = {model.params['w'].sum()}")

        w_eff = model.get_w()

        print(f"DEBUG (SimpleMRF Test): In test, after get_w, w_eff is None: {w_eff is None}")
        if w_eff is not None:
            print(f"DEBUG (SimpleMRF Test): In test, after get_w, w_eff.sum() = {w_eff.sum()}")
            print(f"DEBUG (SimpleMRF Test): In test, after get_w, w_eff type: {type(w_eff)}")


        assert w_eff is not None, "SimpleMRF.get_w() should not return None"
        assert w_eff.shape == (L_test, A_test, L_test, A_test)
        assert jnp.allclose(w_eff, w_val), "SimpleMRF.get_w() should return the loaded 'w' parameter directly"

        print("SimpleMRF get_w test completed its assertions.")
