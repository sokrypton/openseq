import string
import numpy as np
import jax
import jax.numpy as jnp

ALPHABET = list("ARNDCQEGHILKMFPSTWYV-") # 20 amino acids + gap

def parse_fasta(filename: str, a3m: bool = False, stop: int = 100000):
    """
    Parses a FASTA file.

    Args:
        filename (str): Path to the FASTA file.
        a3m (bool): If True, treat as A3M format (remove lowercase letters).
        stop (int): Maximum number of sequences to read.

    Returns:
        tuple: (headers, sequences) where headers is a list of sequence headers
               and sequences is a list of sequence strings.
    """

    rm_lc = None
    if a3m:
        # for a3m files the lowercase letters are removed
        # as these do not align to the query sequence
        rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    headers, sequences = [], []
    with open(filename, "r") as lines:
        for line in lines:
            line = line.rstrip()
            if len(line) > 0:
                if line[0] == ">":
                    if len(headers) == stop:
                        break
                    else:
                        headers.append(line[1:])
                        sequences.append([])
                else:
                    if rm_lc:
                        line = line.translate(rm_lc)
                    else:
                        line = line.upper()
                    sequences[-1].append(line)

    sequences = [''.join(seq) for seq in sequences]
    return headers, sequences

def mk_msa(seqs: list[str]):
    """
    One-hot encode an MSA (Multiple Sequence Alignment).

    Args:
        seqs (list[str]): A list of protein sequences.

    Returns:
        numpy.ndarray: A 3D numpy array representing the one-hot encoded MSA (N, L, A),
                       where N is number of sequences, L is length, A is alphabet size.
    """
    states = len(ALPHABET)
    a2n = {a: n for n, a in enumerate(ALPHABET)}

    msa_ori = []
    if not seqs:
        return np.array([]).reshape(0,0,states) # Handle empty input

    max_len = len(max(seqs, key=len)) if seqs else 0

    for seq in seqs:
        # Pad sequences to the same length (max_len) using the gap character index
        padded_seq_indices = [a2n.get(aa, states - 1) for aa in seq]
        padding_needed = max_len - len(padded_seq_indices)
        if padding_needed > 0:
            padded_seq_indices.extend([states - 1] * padding_needed)
        msa_ori.append(padded_seq_indices)

    msa_np = np.array(msa_ori)

    if msa_np.size == 0: # if all sequences were empty or seqs was empty
        return np.array([]).reshape(len(seqs), max_len, states)

    return np.eye(states)[msa_np] # N, L, A

def get_eff(msa: jnp.ndarray, eff_cutoff: float = 0.8):
    """
    Compute effective sequence weights for an MSA.
    Weights are calculated based on sequence identity to reduce redundancy.

    Args:
        msa (jax.numpy.ndarray): One-hot encoded MSA of shape (N, L, A).
        eff_cutoff (float): Effective sequence identity cutoff. Sequences with identity
                            greater or equal to this cutoff contribute to the weight.

    Returns:
        jax.numpy.ndarray: An array of weights, one for each sequence in the MSA.
    """
    if msa.shape[0] == 0:
        return jnp.array([])
    if msa.shape[0] == 1:
        return jnp.array([1.0])

    # If memory is a concern for large MSAs, the original loop-based approach could be an option.
    # For now, using the all-to-all comparison as it's more common and was in the JAX version.
    # Ensure msa is (N, L, A) where A is alphabet size (e.g., 21)
    # Identity = sum of matching one-hot vectors / length
    # msa_ident = jnp.tensordot(msa, msa, [[1,2],[1,2]]) / msa.shape[1]

    # Convert one-hot to integer sequences for hamming distance like calculation
    msa_int = msa.argmax(-1) # (N, L)

    # Pairwise identity calculation (1 - Hamming distance normalized by length)
    # This requires broadcasting and careful summation.
    # (N, 1, L) vs (1, N, L) -> (N, N, L) for equality check
    matches = (msa_int[:, None, :] == msa_int[None, :, :])
    msa_ident = jnp.sum(matches, axis=-1) / msa.shape[1] # (N, N)

    return 1.0 / (msa_ident >= eff_cutoff).sum(-1)

def ar_mask(order: jnp.ndarray, diag: bool = True):
    """
    Compute autoregressive mask, given order of positions.
    The mask is used to ensure that predictions for a position only depend on preceding positions
    in the given order.

    Args:
        order (jax.numpy.ndarray): A 1D array specifying the order of positions (e.g., [0, 1, 2, ... L-1]
                                   or a permutation).
        diag (bool): If True, the diagonal is included in the mask (position depends on itself).
                     If False, diagonal is excluded.

    Returns:
        jax.numpy.ndarray: A 2D mask array (L, L) where mask[i, j] = 1 if position j can influence
                           position i under the autoregressive ordering.
    """
    L = order.shape[0]
    # Create an index array [0, 1, ..., L-1]
    # The rank r[i] gives the position of original index i in the new order.
    # Example: order = [0, 2, 1] (L=3)
    # r = argsort([0, 2, 1]) = [0, 2, 1] (original index 0 is 0th, 1 is 2nd, 2 is 1st in order)
    r = jnp.argsort(order) # Correctly, this should be argsort of argsort or inverse permutation.
                           # Let's use the simpler logic that was there:
                           # r = order[::-1].argsort() -> this was for a specific kind of ordering.
                           # A more general way:
                           # Create a grid of positions in the new order
                           # pos_in_order_i = r[None, :]
                           # pos_in_order_j = r[:, None]
                           # mask = pos_in_order_j <= pos_in_order_i (if k=0 for triu)
                           # The original was:

    # r = order.argsort() # r[k] is the original index of the k-th element in the sorted order
    # A simpler interpretation: order[i] is the i-th position in the sequence to be processed.
    # We want a mask M such that M[order[i], order[j]] = 1 if j comes before or at i in the order.

    # The original ar_mask from utils.py:
    # r = order[::-1].argsort()
    # tri = jnp.triu(jnp.ones((L,L)),k=not diag)
    # return tri[r[None,:],r[:,None]]
    # Let's test this logic:
    # order = [0, 1, 2] (standard order) -> r = [2, 1, 0]
    # tri = [[1,1,1],[0,1,1],[0,0,1]] (diag=True)
    # M[i,j] = tri[r[i], r[j]]
    # M[0,0] = tri[r[0],r[0]] = tri[2,2] = 1
    # M[0,1] = tri[r[0],r[1]] = tri[2,1] = 0 -> Incorrect. For standard order, pos 0 depends on nothing before it (or itself if diag).
                                          # Pos 1 depends on 0 (and 1 if diag).
                                          # Pos 2 depends on 0, 1 (and 2 if diag).
    # A standard causal mask M_ij = 1 if j <= i (for standard order)
    # For a given `order`, if we process in `order[0], order[1], ...`, then
    # `order[k]` can depend on `order[0]...order[k]`.
    # So, if `i = order[k_i]` and `j = order[k_j]`, we need `k_j <= k_i`.

    # Let inv_order be the inverse permutation: inv_order[k] = index of k in `order`.
    # e.g., order = [0,2,1], inv_order = [0,2,1] (pos 0 is 0th, pos 1 is 2nd, pos 2 is 1st)
    inv_order = jnp.zeros_like(order)
    inv_order = inv_order.at[order].set(jnp.arange(L))

    # M_ij = 1 if inv_order[j] <= inv_order[i]
    idx_grid_j, idx_grid_i = jnp.meshgrid(inv_order, inv_order)
    mask = (idx_grid_j <= idx_grid_i).astype(jnp.float32)

    if not diag:
        mask = mask * (1 - jnp.eye(L))

    return mask

# Add any other general utilities from the original utils.py if they were missed
# or deemed general enough.
# For example, if ALPHABET needs to be configurable, it could be passed to functions.
# For now, keeping it global as in the original.
