import pytest
import jax # Import jax itself
import jax.numpy as jnp
import numpy as np # For creating test arrays easily
from openseq.utils.data_processing import parse_fasta, mk_msa, ALPHABET, get_eff, ar_mask
import os

# Fixture for creating a temporary FASTA file
@pytest.fixture
def temp_fasta_file(tmp_path):
    fasta_content = """>seq1\nARND\n>seq2\nCQE\n"""
    file_path = tmp_path / "test.fasta"
    file_path.write_text(fasta_content)
    return str(file_path)

@pytest.fixture
def temp_a3m_file(tmp_path):
    a3m_content = """>seq1\nARND\n>seq2\nCqeG\n""" # Mixed case for a3m
    file_path = tmp_path / "test.a3m"
    file_path.write_text(a3m_content)
    return str(file_path)

class TestParseFasta:
    def test_parse_simple_fasta(self, temp_fasta_file):
        headers, seqs = parse_fasta(temp_fasta_file)
        assert headers == ["seq1", "seq2"]
        assert seqs == ["ARND", "CQE"]

    def test_parse_a3m(self, temp_a3m_file):
        headers, seqs = parse_fasta(temp_a3m_file, a3m=True)
        assert headers == ["seq1", "seq2"]
        assert seqs == ["ARND", "CG"] # Lowercase 'q' and 'e' removed

    def test_parse_stop_limit(self, temp_fasta_file):
        headers, seqs = parse_fasta(temp_fasta_file, stop=1)
        assert len(headers) == 1
        assert headers == ["seq1"]
        assert len(seqs) == 1
        assert seqs == ["ARND"]

    def test_parse_empty_file(self, tmp_path):
        file_path = tmp_path / "empty.fasta"
        file_path.write_text("")
        headers, seqs = parse_fasta(str(file_path))
        assert headers == []
        assert seqs == []

class TestMkMsa:
    def test_mk_msa_simple(self):
        seqs = ["AR", "N-"]
        msa_one_hot = mk_msa(seqs)

        assert msa_one_hot.shape == (2, 2, len(ALPHABET))

        # Expected: A=0, R=1, N=2, D=3, ... -=20
        # Seq1: AR
        assert msa_one_hot[0, 0, ALPHABET.index('A')] == 1
        assert msa_one_hot[0, 1, ALPHABET.index('R')] == 1
        assert np.sum(msa_one_hot[0,0,:]) == 1
        assert np.sum(msa_one_hot[0,1,:]) == 1

        # Seq2: N-
        assert msa_one_hot[1, 0, ALPHABET.index('N')] == 1
        assert msa_one_hot[1, 1, ALPHABET.index('-')] == 1
        assert np.sum(msa_one_hot[1,0,:]) == 1
        assert np.sum(msa_one_hot[1,1,:]) == 1

    def test_mk_msa_padding(self):
        seqs = ["ARND", "CQE"] # Max len 4
        msa_one_hot = mk_msa(seqs)
        assert msa_one_hot.shape == (2, 4, len(ALPHABET))
        # Check padding for CQE (length 3)
        assert msa_one_hot[1, 2, ALPHABET.index('E')] == 1 # Last char of CQE
        assert msa_one_hot[1, 3, ALPHABET.index('-')] == 1 # Padded with gap
        assert np.sum(msa_one_hot[1,3,:]) == 1


    def test_mk_msa_empty_input(self):
        msa_one_hot = mk_msa([])
        assert msa_one_hot.shape == (0, 0, len(ALPHABET))

    def test_mk_msa_all_empty_seqs(self):
        msa_one_hot = mk_msa(["", ""])
        assert msa_one_hot.shape == (2, 0, len(ALPHABET))


class TestGetEff:
    def test_get_eff_simple(self):
        # A R N D C Q E G H I L K M F P S T W Y V -
        # Seq1: A A A
        # Seq2: A A A (Identical)
        # Seq3: R R R (Different)
        seq1 = [ALPHABET.index('A')] * 3
        seq2 = [ALPHABET.index('A')] * 3
        seq3 = [ALPHABET.index('R')] * 3

        msa_int = jnp.array([seq1, seq2, seq3])
        msa_one_hot = jax.nn.one_hot(msa_int, len(ALPHABET))

        weights = get_eff(msa_one_hot, eff_cutoff=0.8)
        # Expected: seq1 and seq2 are identical (ID=1.0). seq3 is different from both (ID=0.0).
        # For seq1: ID with seq1=1, seq2=1, seq3=0. Sum >= 0.8 is 2. Weight = 1/2.
        # For seq2: ID with seq1=1, seq2=1, seq3=0. Sum >= 0.8 is 2. Weight = 1/2.
        # For seq3: ID with seq1=0, seq2=0, seq3=1. Sum >= 0.8 is 1. Weight = 1/1.
        expected_weights = jnp.array([0.5, 0.5, 1.0])
        assert weights.shape == (3,)
        assert jnp.allclose(weights, expected_weights, atol=1e-5)

    def test_get_eff_single_sequence(self):
        msa_one_hot = mk_msa(["ARND"])
        weights = get_eff(jnp.asarray(msa_one_hot), eff_cutoff=0.8)
        assert weights.shape == (1,)
        assert jnp.allclose(weights, jnp.array([1.0]))

    def test_get_eff_empty_msa(self):
        msa_one_hot = mk_msa([])
        weights = get_eff(jnp.asarray(msa_one_hot))
        assert weights.shape == (0,)


class TestArMask:
    def test_ar_mask_diag_true(self):
        order = jnp.array([0, 1, 2]) # Standard order
        L = len(order)
        mask = ar_mask(order, diag=True)
        expected_mask = jnp.array([
            [1, 0, 0],  # Pos 0 depends on 0
            [1, 1, 0],  # Pos 1 depends on 0, 1
            [1, 1, 1]   # Pos 2 depends on 0, 1, 2
        ])
        # The ar_mask means M[i,j]=1 if j contributes to i.
        # If order=[0,1,2], then for i=0, only j=0 contributes. For i=1, j=0,1 contribute.
        # So, M_ij=1 if inv_order[j] <= inv_order[i]
        # inv_order = [0,1,2] for order=[0,1,2]
        # M_00: inv_order[0] <= inv_order[0] (0<=0) -> 1
        # M_10: inv_order[0] <= inv_order[1] (0<=1) -> 1
        # M_11: inv_order[1] <= inv_order[1] (1<=1) -> 1
        # M_01: inv_order[1] <= inv_order[0] (1<=0) -> 0
        assert mask.shape == (L, L)
        assert jnp.allclose(mask, expected_mask)

    def test_ar_mask_diag_false(self):
        order = jnp.array([0, 1, 2])
        L = len(order)
        mask = ar_mask(order, diag=False)
        expected_mask = jnp.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0]
        ])
        assert mask.shape == (L, L)
        assert jnp.allclose(mask, expected_mask)

    def test_ar_mask_permuted_order_diag_true(self):
        order = jnp.array([0, 2, 1]) # Process 0, then 2, then 1
        # inv_order: pos 0 is 0th, pos 1 is 2nd, pos 2 is 1st
        # inv_order = [0, 2, 1]
        L = len(order)
        mask = ar_mask(order, diag=True)
        # Expected: M_ij = 1 if inv_order[j] <= inv_order[i]
        # M[0,0]: inv[0]<=inv[0] (0<=0) T
        # M[0,1]: inv[1]<=inv[0] (2<=0) F
        # M[0,2]: inv[2]<=inv[0] (1<=0) F

        # M[1,0]: inv[0]<=inv[1] (0<=2) T
        # M[1,1]: inv[1]<=inv[1] (2<=2) T
        # M[1,2]: inv[2]<=inv[1] (1<=2) T

        # M[2,0]: inv[0]<=inv[2] (0<=1) T
        # M[2,1]: inv[1]<=inv[2] (2<=1) F
        # M[2,2]: inv[2]<=inv[2] (1<=1) T
        expected_mask = jnp.array([
            [1, 0, 0],  # Pos 0 (order[0]) depends on 0
            [1, 1, 1],  # Pos 1 (order[2]) depends on 0, 1, 2 (because 0,2 came before 1 in `order`)
            [1, 0, 1]   # Pos 2 (order[1]) depends on 0, 2
        ])
        assert mask.shape == (L, L)
        assert jnp.allclose(mask, expected_mask)
