import pytest
import jax
import jax.numpy as jnp
import numpy as np
from openseq.models import SMURF
from openseq.utils.data_processing import ALPHABET, mk_msa
from openseq.utils.random import get_random_key

# Helper for SMURF tests
def smurf_preprocess_unaligned(seq_list: list[str], max_len: int, alphabet_list: list[str]):
    """Simplified preprocessor for test data."""
    A_size = len(alphabet_list)
    a2n = {a: i for i, a in enumerate(alphabet_list)}

    X_padded_list = []
    lengths = []
    for seq_str in seq_list:
        lengths.append(len(seq_str))
        indices = [a2n.get(c, A_size - 1) for c in seq_str]
        if len(indices) < max_len:
            indices.extend([A_size - 1] * (max_len - len(indices)))
        elif len(indices) > max_len:
            indices = indices[:max_len]
        X_padded_list.append(indices)

    if not X_padded_list:
        return jnp.empty((0, max_len, A_size)), jnp.empty((0,), dtype=jnp.int32)

    X_indices = jnp.array(X_padded_list, dtype=jnp.int32)
    X_one_hot = jax.nn.one_hot(X_indices, A_size, dtype=jnp.float32)
    return X_one_hot, jnp.array(lengths, dtype=jnp.int32)

class TestSMURF:
    @pytest.fixture
    def smurf_config_minimal(self):
        # Minimal config for basic tests
        return {
            "A": len(ALPHABET),
            "L_ref": 5,
            "filters": 8,
            "win": 3,
            "mrf_lam": 0.01,
            "sw_temp": 1.0,
            "sw_open": 1.0, # Penalty
            "sw_gap": 0.5,  # Penalty
            "learning_rate": 0.01,
            "seed": 0
        }

    @pytest.fixture
    def dummy_unaligned_data(self, smurf_config_minimal):
        seq_list = ["ARND", "RNDC", "ANDC"]
        X_ref_str = "ARNDC" # Length L_ref = 5
        max_len = max(len(s) for s in seq_list) # max_len = 4

        X_unaligned_one_hot, lengths = smurf_preprocess_unaligned(seq_list, max_len, ALPHABET)
        X_ref_one_hot = jnp.asarray(mk_msa([X_ref_str]))

        return X_unaligned_one_hot, lengths, X_ref_one_hot, X_ref_str, seq_list


    def test_smurf_initialization(self, smurf_config_minimal):
        model = SMURF(**smurf_config_minimal)
        assert model.A == smurf_config_minimal["A"]
        assert model.L_ref == smurf_config_minimal["L_ref"]
        assert model.filters == smurf_config_minimal["filters"]

        assert "emb" in model.params
        assert "w" in model.params["emb"] and model.params["emb"]["w"].shape == (model.filters, model.A, model.win)

        assert "mrf" in model.params
        assert "w" in model.params["mrf"] and \
               model.params["mrf"]["w"].shape == (model.L_ref, model.A, model.L_ref, model.A)
        assert "b" in model.params["mrf"] and \
               model.params["mrf"]["b"].shape == (model.L_ref, model.A)

        assert "sw_temp_param" in model.params # Even if not learned, stored as fixed
        assert "sw_gap_params" in model.params
        assert model.optimizer_state is not None

    def test_smurf_embed_sequences(self, smurf_config_minimal, dummy_unaligned_data):
        model = SMURF(**smurf_config_minimal)
        X_unaligned, _, X_ref, _, _ = dummy_unaligned_data
        key = get_random_key(0)

        emb_seqs = model._embed_sequences(model.params["emb"], X_unaligned, key_noise=key)
        assert emb_seqs.shape == (X_unaligned.shape[0], X_unaligned.shape[1], model.filters)

        emb_ref = model._embed_sequences(model.params["emb"], X_ref, key_noise=None)
        assert emb_ref.shape == (X_ref.shape[0], model.L_ref, model.filters)

    def test_smurf_get_sw_similarity_matrix(self, smurf_config_minimal, dummy_unaligned_data):
        model = SMURF(**smurf_config_minimal)
        X_unaligned, lengths, X_ref, _, _ = dummy_unaligned_data
        key = get_random_key(1)

        emb_x_batch = model._embed_sequences(model.params['emb'], X_unaligned, key_noise=key)
        emb_x_ref = model._embed_sequences(model.params['emb'], X_ref, key_noise=None)

        sim_matrix, sw_mask = model._get_sw_similarity_matrix(emb_x_batch, emb_x_ref, lengths)

        N, L_seq_max, _ = X_unaligned.shape
        L_ref = model.L_ref
        assert sim_matrix.shape == (N, L_seq_max, L_ref)
        assert sw_mask.shape == (N, L_seq_max, L_ref)
        assert jnp.all((sw_mask == 0) | (sw_mask == 1))

    def test_smurf_align_sequences_sw(self, smurf_config_minimal, dummy_unaligned_data):
        model = SMURF(**{**smurf_config_minimal, "use_pseudo_alignment": False}) # Ensure SW
        X_unaligned, lengths, X_ref, _, _ = dummy_unaligned_data
        key_emb, key_align = jax.random.split(get_random_key(2))

        emb_x_batch = model._embed_sequences(model.params['emb'], X_unaligned, key_noise=key_emb)
        emb_x_ref = model._embed_sequences(model.params['emb'], X_ref, key_noise=None)
        sim_matrix, sw_mask = model._get_sw_similarity_matrix(emb_x_batch, emb_x_ref, lengths)

        aln = model._align_sequences(sim_matrix, sw_mask, lengths, key_align=key_align)
        N, L_seq_max, _ = X_unaligned.shape
        L_ref = model.L_ref
        assert aln.shape == (N, L_seq_max, L_ref)

    def test_smurf_align_sequences_pseudo(self, smurf_config_minimal, dummy_unaligned_data):
        model = SMURF(**{**smurf_config_minimal, "use_pseudo_alignment": True})
        X_unaligned, lengths, X_ref, _, _ = dummy_unaligned_data
        key_emb, key_align = jax.random.split(get_random_key(3))

        emb_x_batch = model._embed_sequences(model.params['emb'], X_unaligned, key_noise=key_emb)
        emb_x_ref = model._embed_sequences(model.params['emb'], X_ref, key_noise=None)
        sim_matrix, sw_mask = model._get_sw_similarity_matrix(emb_x_batch, emb_x_ref, lengths)

        aln = model._align_sequences(sim_matrix, sw_mask, lengths, key_align=key_align)
        N, L_seq_max, _ = X_unaligned.shape
        L_ref = model.L_ref
        assert aln.shape == (N, L_seq_max, L_ref)
        assert jnp.all(aln >= 0) and jnp.all(aln <= 1.00001) # Softmax probabilities (sqrt)

    def test_smurf_model_loss_fn_runs(self, smurf_config_minimal, dummy_unaligned_data):
        model = SMURF(**smurf_config_minimal)
        X_unaligned, lengths, X_ref, _, _ = dummy_unaligned_data
        key = get_random_key(4)

        loss = model._model_loss_fn(model.params, X_unaligned, lengths, X_ref, key)
        assert isinstance(loss.item(), float)
        assert not jnp.isnan(loss) and not jnp.isinf(loss)

    def test_smurf_fit_runs_params_change(self, smurf_config_minimal, dummy_unaligned_data):
        model = SMURF(**smurf_config_minimal)
        _, _, _, X_ref_str, X_unaligned_list = dummy_unaligned_data
        key = get_random_key(5)

        initial_mrf_w = jnp.copy(model.params['mrf']['w'])

        # Use very small number of steps for quick test
        model.fit(X_unaligned_list, X_ref_str, steps=2, batch_size=2, verbose=False, key=key, max_len_pad=6)

        assert not jnp.allclose(model.params['mrf']['w'], initial_mrf_w)

    def test_smurf_get_w_and_predict(self, smurf_config_minimal):
        model = SMURF(**smurf_config_minimal)
        L_ref, A = model.L_ref, model.A
        # Load some non-zero params for MRF part
        key = get_random_key(6)
        w_param = jax.random.normal(key, (L_ref, A, L_ref, A)) * 0.1
        b_param = jax.random.normal(key, (L_ref, A)) * 0.1

        # Create a full params dict to load
        temp_params = model.get_parameters() # Get default structure
        temp_params['mrf']['w'] = w_param
        temp_params['mrf']['b'] = b_param
        model.load_parameters(temp_params)

        w_eff = model.get_w()
        assert w_eff is not None
        assert w_eff.shape == (L_ref, A, L_ref, A)

        raw_map, apc_map = model.predict() # X_unaligned, lengths not needed for this
        assert raw_map.shape == (L_ref, L_ref)
        assert apc_map.shape == (L_ref, L_ref)

    def test_smurf_sample_runs(self, smurf_config_minimal):
        model = SMURF(**smurf_config_minimal)
        key = get_random_key(7)
        # Sample method in SMURF currently samples from its MRF component (L_ref length)
        samples = model.sample(num_samples=2, key=key, return_one_hot=True)
        assert samples.shape == (2, model.L_ref, model.A)

    def test_smurf_get_load_parameters(self, smurf_config_minimal, dummy_unaligned_data):
        model1 = SMURF(**smurf_config_minimal)
        _, _, _, X_ref_str, X_unaligned_list = dummy_unaligned_data
        key = get_random_key(8)
        model1.fit(X_unaligned_list, X_ref_str, steps=1, verbose=False, key=key, max_len_pad=6)

        params1 = model1.get_parameters()
        assert "emb" in params1 and "mrf" in params1

        config2 = smurf_config_minimal.copy()
        config2["seed"] +=1
        model2 = SMURF(**config2)
        model2.load_parameters(params1)
        params2 = model2.get_parameters()

        assert jnp.allclose(params1['emb']['w'], params2['emb']['w'])
        assert jnp.allclose(params1['mrf']['w'], params2['mrf']['w'])
        assert jnp.allclose(params1['mrf']['b'], params2['mrf']['b'])
