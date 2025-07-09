import jax
import jax.numpy as jnp
import numpy as np # Retained for potential type hints or minor utilities if needed

def get_stats(X: jnp.ndarray,
              X_weight: jnp.ndarray = None,
              labels: jnp.ndarray = None,
              add_f_ij: bool = True,
              add_mf_ij: bool = False,
              add_c: bool = False):
    """
    Compute first and second order statistics (f_i, f_ij) for an MSA.
    Optionally computes mixture statistics if labels are provided.

    Args:
        X (jnp.ndarray): One-hot encoded MSA of shape (N, L, A).
        X_weight (jnp.ndarray, optional): Sequence weights of shape (N,). Defaults to ones.
        labels (jnp.ndarray, optional): Cluster/mixture labels for each sequence.
                                        Can be integer labels (N,) or one-hot (N, K).
        add_f_ij (bool): If True, compute pairwise frequencies f_ij.
        add_mf_ij (bool): If True and labels provided, compute mixture pairwise frequencies mf_ij.
        add_c (bool): If True, compute covariance matrices c_ij (and mc_ij if applicable).

    Returns:
        dict: A dictionary containing computed statistics.
              Keys can include 'f_i', 'f_ij', 'c_ij', 'mf_i', 'mf_ij', 'mc_ij'.
    """
    N, L, A = X.shape
    n_none = None # For using None in slicing for newaxis

    if X_weight is None:
        Xn = X  # For sums (counts)
        Xs = X  # For products (used in f_ij) - effectively same as Xn if no sqrt weights
    else:
        # Ensure X_weight is (N, 1, 1) for broadcasting with X (N, L, A)
        X_weight_reshaped = X_weight[:, n_none, n_none]
        Xn = X * X_weight_reshaped
        # For f_ij, original code used X * sqrt(X_weight).
        # This is common if aiming for a weighted covariance interpretation.
        Xs = X * jnp.sqrt(X_weight_reshaped)

    f_i = Xn.sum(0) # (L, A)
    # Normalize f_i per position: sum_A f_i(pos, A) = 1
    # The original code normalized by f_i.sum(1,keepdims=True) which is sum over A.
    # However, f_i itself after Xn.sum(0) is sum of weights for each (L,A).
    # Normalization should be f_i / sum(f_i over A for each L)
    # No, the original was: o = {"f_i": f_i / f_i.sum(1,keepdims=True)}
    # f_i.sum(1,keepdims=True) would be (L,1). This is correct.
    # Wait, f_i = Xn.sum(0) is (L,A). f_i.sum(1, keepdims=True) sums over A for each L. This is total weight per pos.
    # This is fine.
    # The original code stats.py: `o = {"f_i": f_i / f_i.sum(1,keepdims=True)}`
    # This normalization makes each f_i[l,:] sum to 1.
    # This is unusual. Typically f_i is sum of weights for each (L,A) / TotalEffectiveSequences.
    # Let's assume Neff is implicitly handled by X_weight.
    # If X_weight sums to Neff, then f_i from Xn.sum(0) is Neff * P_i.
    # The normalization f_i / f_i.sum(1,keepdims=True) makes it P(A | L).
    # This seems to be the convention in this codebase.

    o = {"f_i": f_i / (f_i.sum(axis=1, keepdims=True) + 1e-8)} # (L,A)

    if add_f_ij:
        # f_ij = jnp.tensordot(Xs, Xs, [[0], [0]]) # (L, A, L, A)
        # This is equivalent to einsum('nia,njb->iajb', Xs, Xs) if Xs is (N,L,A)
        f_ij = jnp.einsum('nla,nqb->laqb', Xs, Xs) # (L,A,L,A)

        # Normalize f_ij: sum_AB f_ij(pos1, A, pos2, B) = 1
        # Original: f_ij / f_ij.sum((1,3),keepdims=True)
        # sum over A for pos1, and B for pos2. Correct.
        o["f_ij"] = f_ij / (f_ij.sum(axis=(1, 3), keepdims=True) + 1e-8) # (L,A,L,A)
        if add_c:
            # c_ij = f_ij - f_i (outer product) f_i (L,A)
            # f_i[:,:,n_none,n_none] -> (L,A,1,1)
            # f_i[n_none,n_none,:,:] -> (1,1,L,A)
            o["c_ij"] = o["f_ij"] - o["f_i"][:, :, n_none, n_none] * o["f_i"][n_none, n_none, :, :]

    if labels is not None:
        if jnp.issubdtype(labels.dtype, jnp.integer):
            if labels.max() >= 0 : # Ensure there are actual labels
                 num_clusters = labels.max() + 1
                 labels_one_hot = jax.nn.one_hot(labels, num_clusters) # (N, K)
            else: # Handle case with no valid labels (e.g. all -1)
                labels_one_hot = jnp.zeros((N,0)) # No clusters
        else: # Already one-hot
            labels_one_hot = labels

        K = labels_one_hot.shape[1]
        if K == 0: # No valid clusters
            return o # Return basic stats

        # mf_i: mixture first-order stats (K, L, A)
        # labels_one_hot (N,K), Xn (N,L,A) -> einsum('nk,nla->kla', labels_one_hot, Xn)
        mf_i = jnp.einsum("nc,nia->cia", labels_one_hot, Xn) # (K, L, A)
        # Normalize mf_i: sum_A mf_i(k,l,A) = 1 for each k,l
        # Original: mf_i/mf_i.sum((0,2),keepdims=True) -> sum over K and A. This is unusual.
        # Should be sum over A for each K,L: mf_i.sum(axis=2, keepdims=True)
        # Let's re-check original: mf_i.sum((0,2),keepdims=True) is (1,L,1).
        # This normalizes each position L across all mixtures K and all alphabets A.
        # This makes P(A,K | L).
        # If we want P(A | L,K), it should be sum(axis=2, keepdims=True).
        # Given the pattern for f_i, P(A|L), it's likely P(A | L, K) is intended.
        o["mf_i"] = mf_i / (mf_i.sum(axis=2, keepdims=True) + 1e-8)

        if add_mf_ij:
            # mf_ij: mixture pairwise stats (K, L, A, L, A)
            # labels_one_hot (N,K), Xs (N,L,A), Xs (N,L,A) -> einsum('nk,nla,nqb->klaqb')
            mf_ij = jnp.einsum("nc,nia,njb->ciajb", labels_one_hot, Xs, Xs) # (K,L,A,L,A)
            # Normalize mf_ij: sum_AB mf_ij(k,l1,A,l2,B) = 1 for each k,l1,l2
            # Original: mf_ij/mf_ij.sum((0,2,4),keepdims=True) -> sum over K, A1, A2. (1,L,1,L,1)
            # This is P(A1,A2,K | L1,L2).
            # If P(A1,A2 | L1,L2,K) is intended, sum should be axis=(2,4).
            o["mf_ij"] = mf_ij / (mf_ij.sum(axis=(2, 4), keepdims=True) + 1e-8)
            if add_c:
                # mc_ij = mf_ij - mf_i (outer product) mf_i
                # mf_i (K,L,A)
                # mf_i[:,:,:,n_none,n_none] -> (K,L,A,1,1)
                # mf_i[:,n_none,n_none,:,:] -> (K,1,1,L,A)
                o["mc_ij"] = o["mf_ij"] - o["mf_i"][:, :, :, n_none, n_none] * o["mf_i"][:, n_none, n_none, :, :]
    return o

def get_r(a: jnp.ndarray, b: jnp.ndarray):
    """Compute Pearson correlation coefficient between two flattened arrays."""
    a_flat = jnp.array(a).flatten()
    b_flat = jnp.array(b).flatten()
    return jnp.corrcoef(a_flat, b_flat)[0, 1]

def inv_cov(X: jnp.ndarray, X_weight: jnp.ndarray = None, pseudo_count: float = None):
    """
    Compute inverse covariance matrix from MSA.

    Args:
        X (jnp.ndarray): One-hot encoded MSA (N, L, A).
        X_weight (jnp.ndarray, optional): Sequence weights (N,).
        pseudo_count (float, optional): Pseudocount for covariance regularization.
                                        Original used 4.5 / sqrt(N_eff). If None, no pseudocount added here.

    Returns:
        jnp.ndarray: Inverse covariance matrix of shape (L, A, L, A).
    """
    X = jnp.asarray(X)
    N, L, A = X.shape

    if X_weight is None:
        num_points = N
    else:
        X_weight = jnp.asarray(X_weight)
        num_points = X_weight.sum()

    stats = get_stats(X, X_weight, add_f_ij=True, add_c=True)
    c_ij = stats["c_ij"] # (L,A,L,A)

    c_ij_reshaped = c_ij.reshape(L * A, L * A)

    # Regularization (shrinkage)
    # The original code used: shrink = 4.5 / jnp.sqrt(num_points) * jnp.eye(c.shape[0])
    # This is a specific form of shrinkage. If pseudo_count is passed, use it.
    # Otherwise, one might need to pass Neff to replicate original.
    # For now, this function will just compute inv(c_ij). Regularization can be external.
    if pseudo_count is None and num_points > 0: # Replicate original if no pseudo_count given
        shrink_coeff = 4.5 / jnp.sqrt(num_points)
    elif pseudo_count is not None:
        shrink_coeff = pseudo_count
    else: # num_points is 0 or pseudo_count is explicitly 0
        shrink_coeff = 0.0

    identity_matrix = jnp.eye(c_ij_reshaped.shape[0])
    regularized_c_ij = c_ij_reshaped + shrink_coeff * identity_matrix

    inv_c_ij_reshaped = jnp.linalg.inv(regularized_c_ij)
    return inv_c_ij_reshaped.reshape(L, A, L, A)

def get_mtx(W: jnp.ndarray, alphabet_size_no_gap: int = 20):
    """
    Compute raw and APC (Average Product Correction) matrices from couplings W.
    Assumes W is (L, A, L, A).

    Args:
        W (jnp.ndarray): Coupling matrix (L, A_with_gap, L, A_with_gap).
        alphabet_size_no_gap (int): Size of the alphabet excluding gaps (e.g., 20 for amino acids).
                                    Couplings involving gaps are excluded.

    Returns:
        tuple: (raw_matrix, apc_matrix) both of shape (L, L).
    """
    W = jnp.asarray(W)
    L = W.shape[0]

    # L2 norm of A_no_gap x A_no_gap submatrices (ignoring gaps)
    # W[:, :A_no_gap, :, :A_no_gap]
    raw = jnp.sqrt(jnp.sum(jnp.square(W[:, :alphabet_size_no_gap, :, :alphabet_size_no_gap]), axis=(1, 3)))
    # raw = raw.at[jnp.diag_indices_from(raw)].set(0) # Original sets diagonal to 0 after norm
    # A better way to ensure diagonal is zero:
    raw = raw * (1 - jnp.eye(L))


    # APC (Average Product Correction)
    # ap = raw.sum(0,keepdims=True) * raw.sum(1,keepdims=True) / raw.sum()
    # Need to handle sum over all elements carefully if raw can be all zeros
    sum_raw_total = raw.sum()
    if sum_raw_total == 0: # Avoid division by zero if raw matrix is all zeros
        ap = jnp.zeros_like(raw)
    else:
        ap = raw.sum(axis=0, keepdims=True) * raw.sum(axis=1, keepdims=True) / sum_raw_total

    apc = raw - ap
    # apc = apc.at[jnp.diag_indices_from(apc)].set(0) # Original sets diagonal to 0 after APC
    apc = apc * (1 - jnp.eye(L))

    return raw, apc

def con_auc(true_contacts: jnp.ndarray, pred_contacts: jnp.ndarray,
            mask: jnp.ndarray = None, min_separation: int = 6):
    """
    Compute contact AUC (Area Under Curve) for contact prediction.
    Compares predicted contact map to a true contact map.

    Args:
        true_contacts (jnp.ndarray): True contact map (L, L), binary or distance values.
                                     If distances, assumes smaller is better (contact).
                                     The original con_auc in network_functions.py used `true > thresh`.
                                     Here, let's assume true_contacts is binary 0/1.
        pred_contacts (jnp.ndarray): Predicted contact scores (L, L). Higher score means stronger prediction.
        mask (jnp.ndarray, optional): Mask for valid positions (L, L) or (L,). If (L,), it's expanded.
        min_separation (int): Minimum sequence separation for contacts to be considered (e.g., i, j s.t. |i-j| >= min_separation).

    Returns:
        jnp.ndarray: Array of precision values at different recall points (L/10, L/5, ..., L).
                     The mean of this array is often taken as the AUC score.
    """
    true_contacts = jnp.asarray(true_contacts)
    pred_contacts = jnp.asarray(pred_contacts)
    L = true_contacts.shape[0]

    if mask is not None:
        mask = jnp.asarray(mask)
        if mask.ndim == 1: # If 1D mask (L,), expand to (L,L)
            idx = mask > 0
            true_contacts = true_contacts[idx, :][:, idx]
            pred_contacts = pred_contacts[idx, :][:, idx]
            L = true_contacts.shape[0] # Update L
        elif mask.ndim == 2: # If 2D mask (L,L)
            # This case was not explicitly handled in original stats.py con_auc
            # Usually, mask is for positions. If it's for pairs, apply directly.
            # Assuming 1D mask logic for now.
            idx = jnp.sum(mask, axis=-1) > 0 # Example if mask was pair-based, sum to find active pos
            true_contacts = true_contacts[idx, :][:, idx]
            pred_contacts = pred_contacts[idx, :][:, idx]
            L = true_contacts.shape[0]


    # Consider only upper triangle with minimum separation
    row_idx, col_idx = jnp.triu_indices(L, k=min_separation)

    pred_flat = pred_contacts[row_idx, col_idx]
    true_flat = true_contacts[row_idx, col_idx]

    # Sort predictions in descending order
    sort_indices = jnp.argsort(pred_flat)[::-1]
    pred_sorted = pred_flat[sort_indices]
    true_sorted = true_flat[sort_indices]

    # Calculate precision at different L/x points
    # Original used L_factors = (jnp.linspace(0.1,1.0,10)*len(true)).astype(jnp.int32)
    # which means L_factors are absolute number of contacts.
    # If L is small, this might be problematic.
    # Using fixed L/x points: L/10, L/5, L/2, L
    # Or simply use all sorted predictions to compute ROC AUC like score.
    # The original con_auc averaged precision at L/10, L/5, ... L.

    num_eval_points = 10
    # Evaluate at top K predictions, where K = L, L/2, L/5 etc.
    # For robustness with small L, let's use fixed proportions of total possible pairs.
    # Or, more simply, use fixed number of top predictions if L is large enough.

    # Replicating original L_factors logic:
    if L == 0 or len(true_flat) == 0:
      return jnp.array([0.0] * num_eval_points) # Or handle as error / nan

    # Consider number of actual pairs evaluated by triu_indices
    num_triu_pairs = len(true_flat)

    # Define evaluation points as fractions of num_triu_pairs
    # fractions = jnp.linspace(0.1, 1.0, num_eval_points) # Fractions of considered pairs
    # top_k_values = (fractions * num_triu_pairs).astype(jnp.int32)
    # top_k_values = jnp.unique(jnp.maximum(1, top_k_values)) # Ensure K >= 1 and unique

    # The original con_auc in network_functions.py used L_factors = (np.linspace(0.1,1.0,10)*L_protein).astype("int")
    # This implies K is relative to protein length L, not number of pairs.
    # Let's use that for consistency.
    L_factors = (jnp.linspace(0.1, 1.0, num_eval_points) * L).astype(jnp.int32)
    L_factors = jnp.unique(jnp.maximum(1, L_factors))


    precisions = []
    for k_top in L_factors:
        if k_top > num_triu_pairs: # Cannot take more than available pairs
            k_top = num_triu_pairs
        if k_top == 0:
            precisions.append(0.0) # Or handle as appropriate, e.g. if L_protein is very small
            continue

        precision_at_k = jnp.sum(true_sorted[:k_top]) / k_top
        precisions.append(precision_at_k)

    if not precisions: # Should not happen if L_factors is handled well
        return jnp.array([0.0])

    return jnp.asarray(precisions)

# Note: `inv_cov` from stats.py used `get_stats` which is now refactored.
# `get_mtx` from stats.py is also refactored here.
# The `con_auc` from network_functions.py was slightly different (used a threshold for true contacts).
# This version assumes true_contacts is binary or that higher value in true_contacts means contact.
# The one in stats.py (this file) is kept.
```
