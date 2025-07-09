import pytest
import jax
import jax.numpy as jnp
import numpy as np
from openseq.models import MRF_BM
from openseq.utils.data_processing import ALPHABET
from openseq.utils.random import get_random_key
from openseq.utils.stats import get_stats # For checking positive/negative phase stats if needed

# Helper to create dummy MSA for testing
def generate_dummy_msa_bm(num_seqs: int, length: int, alphabet_size: int, key):
    seq_indices = jax.random.randint(key, (num_seqs, length), 0, alphabet_size)
    msa_one_hot = jax.nn.one_hot(seq_indices, alphabet_size)
    return msa_one_hot

class TestMRFBM:
    @pytest.fixture
    def mrf_bm_config_k1(self):
        return {"L": 3, "A": 2, "k": 1, "lam": 0.01, "learning_rate": 0.01, "seed": 0,
                "num_cd_samples": 10, "cd_burn_in": 2, "cd_temperature": 1.0}

    @pytest.fixture
    def mrf_bm_config_k2(self):
        return {"L": 3, "A": 2, "k": 2, "lam": 0.01, "learning_rate": 0.01, "seed": 1,
                "num_cd_samples": 10, "cd_burn_in": 2, "cd_temperature": 1.0}

    @pytest.fixture
    def dummy_msa_bm_k1(self, mrf_bm_config_k1):
        key = get_random_key(mrf_bm_config_k1["seed"] + 100)
        N = 5
        msa = generate_dummy_msa_bm(N, mrf_bm_config_k1["L"], mrf_bm_config_k1["A"], key)
        weights = jnp.ones(N) / N
        return msa, weights

    @pytest.fixture
    def dummy_msa_bm_k2(self, mrf_bm_config_k2):
        key = get_random_key(mrf_bm_config_k2["seed"] + 101)
        N = 6 # Divisible by k=2 for some tests if needed
        msa = generate_dummy_msa_bm(N, mrf_bm_config_k2["L"], mrf_bm_config_k2["A"], key)
        weights = jnp.ones(N) / N
        return msa, weights

    def test_mrf_bm_initialization_k1(self, mrf_bm_config_k1):
        model = MRF_BM(**mrf_bm_config_k1)
        assert model.L == mrf_bm_config_k1["L"]
        assert model.A == mrf_bm_config_k1["A"]
        assert model.k == 1
        assert 'w' in model.params and model.params['w'].shape == (model.L, model.A, model.L, model.A)
        assert 'b' in model.params and model.params['b'].shape == (model.L, model.A)
        assert model.optimizer_state is not None

    def test_mrf_bm_initialization_k2(self, mrf_bm_config_k2):
        model = MRF_BM(**mrf_bm_config_k2)
        k = mrf_bm_config_k2["k"]
        assert model.k == k
        assert 'w' in model.params and model.params['w'].shape == (k, model.L, model.A, model.L, model.A)
        assert 'b' in model.params and model.params['b'].shape == (k, model.L, model.A)
        assert 'c' in model.params and model.params['c'].shape == (k,)

    def test_mrf_bm_regularization_loss(self, mrf_bm_config_k1):
        model = MRF_BM(**mrf_bm_config_k1)
        # Test with non-zero params
        key = get_random_key(0)
        params_k1 = {
            'w': jax.random.normal(key, (model.L, model.A, model.L, model.A)),
            'b': jax.random.normal(key, (model.L, model.A))
        }
        reg_loss = model._regularization_loss(params_k1)
        assert reg_loss > 0
        expected_reg = 0.5 * model.lam * jnp.sum(jnp.square(params_k1['w'])) + \
                       model.lam * jnp.sum(jnp.square(params_k1['b']))
        assert jnp.isclose(reg_loss, expected_reg)

    def test_mrf_bm_compute_gradients_k1_structure(self, mrf_bm_config_k1, dummy_msa_bm_k1):
        model = MRF_BM(**mrf_bm_config_k1)
        msa_batch, weights_batch = dummy_msa_bm_k1
        key = get_random_key(mrf_bm_config_k1["seed"] + 200)

        # For k=1, positive_phase_labels_one_hot is None
        grads = model._compute_bm_gradients(model.params, msa_batch, weights_batch, key, positive_phase_labels_one_hot=None)

        assert 'w' in grads and grads['w'].shape == model.params['w'].shape
        assert 'b' in grads and grads['b'].shape == model.params['b'].shape
        assert not jnp.all(grads['w'] == 0) # Gradients should ideally be non-zero
        assert not jnp.all(grads['b'] == 0)


    def test_mrf_bm_fit_k1_params_change(self, mrf_bm_config_k1, dummy_msa_bm_k1):
        model = MRF_BM(**mrf_bm_config_k1)
        msa, weights = dummy_msa_bm_k1
        key = get_random_key(mrf_bm_config_k1["seed"] + 300)

        initial_params_w = jnp.copy(model.params['w'])
        initial_params_b = jnp.copy(model.params['b'])

        model.fit(msa, X_weight=weights, steps=2, batch_size=2, verbose=False, key=key)

        assert not jnp.allclose(model.params['w'], initial_params_w)
        assert not jnp.allclose(model.params['b'], initial_params_b)

    def test_mrf_bm_get_w_and_predict_k1(self, mrf_bm_config_k1):
        model = MRF_BM(**mrf_bm_config_k1)
        L, A = model.L, model.A
        key = get_random_key(mrf_bm_config_k1["seed"] + 400)
        # Load some non-zero params for meaningful test
        w_param = jax.random.normal(key, (L,A,L,A)) * 0.1
        b_param = jax.random.normal(key, (L,A)) * 0.1
        model.load_parameters({'w': w_param, 'b': b_param})

        w_eff = model.get_w()
        assert w_eff is not None, "get_w() should not return None"
        assert w_eff.shape == (L, A, L, A)

        raw_map, apc_map = model.predict()
        assert raw_map.shape == (L, L)
        assert apc_map.shape == (L, L)

    def test_mrf_bm_sample_k1(self, mrf_bm_config_k1):
        model = MRF_BM(**mrf_bm_config_k1)
        # Use zero params for predictable (though uniform) sampling
        model.load_parameters({
            'w': jnp.zeros_like(model.params['w']),
            'b': jnp.zeros_like(model.params['b'])
        })
        key = get_random_key(mrf_bm_config_k1["seed"] + 500)
        num_samples = 3

        samples_oh = model.sample(num_samples, key=key, burn_in=1, return_one_hot=True)
        assert samples_oh.shape == (num_samples, model.L, model.A)
        assert jnp.all((samples_oh == 0) | (samples_oh == 1))
        assert jnp.all(jnp.sum(samples_oh, axis=-1) == 1)

    def test_mrf_bm_get_load_parameters_k1(self, mrf_bm_config_k1, dummy_msa_bm_k1):
        model1 = MRF_BM(**mrf_bm_config_k1)
        msa, weights = dummy_msa_bm_k1
        key = get_random_key(mrf_bm_config_k1["seed"] + 600)
        model1.fit(msa, X_weight=weights, steps=1, verbose=False, key=key) # Fit briefly

        params1 = model1.get_parameters()
        assert 'w' in params1 and 'b' in params1

        config2 = mrf_bm_config_k1.copy()
        config2["seed"] += 1
        model2 = MRF_BM(**config2)
        model2.load_parameters(params1)
        params2 = model2.get_parameters()

        assert jnp.allclose(params1['w'], params2['w'])
        assert jnp.allclose(params1['b'], params2['b'])

        # Check fit runs after load
        key, fit_key = jax.random.split(key)
        try:
            model2.fit(msa, X_weight=weights, steps=1, verbose=False, key=fit_key)
        except Exception as e:
            pytest.fail(f"Fit after load_parameters failed for MRF_BM: {e}")

    # TODO: Add tests for mixture MRF_BM (k > 1)
    # This will require more careful setup for positive phase labels and stats.
    # def test_mrf_bm_fit_k2_params_change(self, mrf_bm_config_k2, dummy_msa_bm_k2): ...
    # def test_mrf_bm_compute_gradients_k2_structure(self, mrf_bm_config_k2, dummy_msa_bm_k2): ...
