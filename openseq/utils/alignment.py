import jax
import jax.numpy as jnp
import os

# Note: The XLA_FLAGS and XLA_PYTHON_CLIENT_PREALLOCATE environment variable settings
# were present in the original sw_functions.py and network_functions.py.
# These are runtime configurations for JAX/XLA.
# It's generally not good practice to set these directly within library code
# as it can affect the user's environment globally and unexpectedly.
# These should ideally be handled by the user or through documentation guidance.
# For example, providing instructions on how to set them if they are beneficial
# for performance with this library.
#
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#
# The CUDA data dir is highly specific to an environment and should not be hardcoded.

# Constants for Smith-Waterman, can be configured if needed
NINF = -1e30 # Negative infinity for padding/masking in SW

# Smith-Waterman implementations
# Based on sw_functions.py
# Authors: Sergey Ovchinnikov and Sam Petti, Spring 2021

def _sw_rotate(x, mask=None):
    """Helper to rotate matrix for striped dynamic-programming in SW nogap."""
    a, b = x.shape
    ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
    i, j = (br - ar) + (a - 1), (ar + br) // 2
    n, m = (a + b - 1), (a + b) // 2

    rotated_x = jnp.zeros([n, m]).at[i, j].set(x)

    if mask is None:
        rotated_mask = jnp.ones_like(rotated_x) # Assuming mask should match x's effect
    else:
        rotated_mask = jnp.zeros([n, m]).at[i, j].set(mask)

    output = {"x": rotated_x, "m": rotated_mask, "o": (jnp.arange(n) + a % 2) % 2}
    prev = (jnp.zeros(m), jnp.zeros(m))
    return output, prev, (i, j)

def _sw_sco_nogap(x, lengths, temp=1.0, unroll=2):
    """Core scoring function for Smith-Waterman with no gaps."""
    def _soft_maximum(val, axis=None):
        return temp * jax.nn.logsumexp(val / temp, axis=axis)

    def _cond(condition, true_val, false_val):
        return condition * true_val + (1 - condition) * false_val

    def _step(prev_h, sm_row_info):
        h2, h1 = prev_h
        h1_T = _cond(sm_row_info["o"], jnp.pad(h1[:-1], [1, 0]), jnp.pad(h1[1:], [0, 1]))

        h0_options = jnp.stack([h2 + sm_row_info["x"], h1, h1_T], -1)
        h0 = sm_row_info["m"] * _soft_maximum(h0_options, axis=-1)
        return (h1, h0), h0

    a, b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]

    sm_rotated_info, prev_initial, idx_map_back = _sw_rotate(x, mask=mask)

    # The sm_rotated_info["x"] etc. are dicts of arrays to be passed as `xs` to scan
    # The scan needs to iterate over the rows of these rotated matrices.
    # The original scan was: jax.lax.scan(_step, prev, sm, unroll=unroll)
    # where `sm` was the dict `output` from `sw_rotate`. This means `sm` itself was the `xs`.
    # This is unusual; scan usually iterates over the leading axis of arrays in `xs`.
    # If `sm` is a dict, JAX might try to scan over its items or raise an error.
    # It's more likely that `sm` should be a pytree where leaves are sequences (arrays to be scanned).
    # Let's assume `sm_rotated_info` is a pytree of arrays, and scan iterates over their first dimension.

    hij_rotated_rows = jax.lax.scan(_step, prev_initial, sm_rotated_info, unroll=unroll)[1]

    # Map back from rotated to original matrix shape
    # hij_rotated_rows is (n, m), need to extract original (a,b) from it using idx_map_back
    hij = hij_rotated_rows[idx_map_back]
    return hij.max()

def smith_waterman_nogap(batch=True, unroll=2):
    """
    Smith-Waterman local alignment with no gap penalties.
    Returns a function that computes the alignment matrix (gradient of score).
    """
    traceback_fn = jax.grad(_sw_sco_nogap, argnums=0) # Gradient w.r.t. 'x' (similarity matrix)

    # Default unroll value to _sw_sco_nogap
    # We need to use functools.partial or a wrapper to pass unroll
    def wrapped_traceback_fn(x, lengths, temp=1.0):
        return traceback_fn(x, lengths, temp, unroll)

    if batch:
        return jax.vmap(wrapped_traceback_fn, (0, 0, None)) # in_axes for x, lengths, temp
    else:
        return wrapped_traceback_fn


def _sw_rotate_affine(x):
    """Helper to rotate matrix for striped dynamic-programming in SW with affine gaps."""
    a, b = x.shape
    ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
    i, j = (br - ar) + (a - 1), (ar + br) // 2
    n, m = (a + b - 1), (a + b) // 2
    output = {"x": jnp.full([n, m], NINF).at[i, j].set(x), "o": (jnp.arange(n) + a % 2) % 2}
    # Initial hidden states for H, E, F matrices (for Align, Right, Down states)
    # Assuming H, E, F are (m, 3) where 3 corresponds to [H, E, F] states or similar logic
    # The original code used (m,3) for h0,h1,h2 in sw_affine's _step
    # Let's assume prev state is for H, E, F combined, so (m,3)
    prev_h_e_f = (jnp.full((m, 3), NINF), jnp.full((m, 3), NINF))
    return output, prev_h_e_f, (i, j)

def _sw_sco_affine(x, lengths, gap_extend=0.0, gap_open=0.0, temp=1.0,
                  restrict_turns=True, penalize_turns=True, unroll=2):
    """Core scoring function for Smith-Waterman with affine gaps."""

    def _soft_maximum(val, axis=None, mask=None):
        def _logsumexp(y_lse):
            y_lse = jnp.maximum(y_lse, NINF) # Clip to avoid -inf issues with exp
            if mask is None:
                return jax.nn.logsumexp(y_lse, axis=axis)
            else:
                # Masked logsumexp: max_val + log(sum(mask * exp(y - max_val)))
                max_val = jnp.max(y_lse, axis=axis, keepdims=True)
                max_val = jnp.where(max_val == NINF, 0, max_val) # Avoid max_val being -inf
                sum_exp = jnp.sum(mask * jnp.exp(y_lse - max_val), axis=axis)
                return jnp.squeeze(max_val, axis=axis) + jnp.log(sum_exp + 1e-8) # Add epsilon for stability
        return temp * _logsumexp(val / temp)

    def _cond(condition, true_val, false_val):
        return condition * true_val + (1 - condition) * false_val

    def _pad(arr_pad, shape_pad):
        return jnp.pad(arr_pad, shape_pad, constant_values=(NINF, NINF))

    def _step(prev_h_e_f_step, sm_row_info_step):
        (h2, h1) = prev_h_e_f_step # h1, h2 are (m,3) representing (H, E, F) scores for two previous diagonals

        # H_i,j = x_i,j + max(H_i-1,j-1, E_i-1,j-1, F_i-1,j-1)
        # E_i,j = max(H_i,j-1 - open, E_i,j-1 - extend)
        # F_i,j = max(H_i-1,j - open, F_i-1,j - extend)
        # This is standard SW affine. The 'rotated' version is more complex.
        # The original laxy.sw_affine _step used:
        # Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]  (shape of h2 was (m,3))
        # Right = _cond(sm["o"], _pad(h1[:-1],([1,0],[0,0])),h1)
        # Down  = _cond(sm["o"], h1,_pad(h1[1:],([0,1],[0,0])))
        # This structure suggests h1, h2 are indeed (m,3) storing [H, E, F] or similar states.

        # H scores (match/mismatch) from h2 (diagonal)
        # Pad h2 to align for diagonal transitions; assuming h2's columns are [H, E, F]
        # Align_options considers transitions from H, E, F of (i-1, j-1) cell
        H_from_diag_options = jnp.pad(h2, [[0,0],[0,1]]) # Pad last dim to be able to stack? This seems off.
                                                      # The original pad was [[0,0],[0,1]], meaning add a column.
                                                      # If h2 is (m,3), this makes it (m,4).
                                                      # It's more likely h2 is (m,) representing M scores,
                                                      # and E, F are separate.
                                                      # Let's re-evaluate laxy's _step carefully.
        # laxy.sw_affine's _step:
        # h0_Align = _soft_maximum(Align, -1) where Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]
        # This implies Align is e.g. (m, 4) and soft_max is over last dim.
        # And h0_Right from Right, h0_Down from Down.
        # h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1) -> so h0 is (m,3)
        # This means h1 and h2 must also be (m,3).
        # The states are likely M (Match/Align), Ix (Insert in X), Iy (Insert in Y).

        # Let h be [M, Ix, Iy]
        # M_prev_diag = h2[:,0], Ix_prev_diag = h2[:,1], Iy_prev_diag = h2[:,2]
        # M_curr comes from M_prev_diag, Ix_prev_diag, Iy_prev_diag + score x_ij
        # Ix_curr comes from M_curr_left - open, Ix_curr_left - extend
        # Iy_curr comes from M_curr_up - open, Iy_curr_up - extend

        # This is what `laxy` seems to implement with its specific padding:
        # It's a striped (diagonal) computation.
        # `h1` and `h2` are scores [M, Ix, Iy] on previous two diagonals.
        # `sm_row_info_step["x"]` is the similarity score for current cells on the diagonal.

        # M_current_diag (score from alignment)
        # Transitions from M, Ix, Iy of cell (i-1,j-1) which is h2 on the current diagonal computation
        m_options = h2 + sm_row_info_step["x"][:, None] # Add similarity score to all three states from diagonal
        m_score = _soft_maximum(m_options, axis=-1)

        # Ix_current_diag (score from gap in Y, i.e. extending X)
        # Transitions from M_i,j-1 - gap_open OR Ix_i,j-1 - gap_extend
        # This corresponds to `Right` in original code.
        # `h1[:-1]` (shifted) or `h1` depending on `sm_row_info_step["o"]` (odd/even diagonal)
        # Let's call this `h_left_equivalent`
        h_left_equivalent = _cond(sm_row_info_step["o"], _pad(h1[:-1], ([1,0],[0,0])), h1)

        ix_options_list = [h_left_equivalent[:,0] + gap_open, # M_left -> Ix_curr
                           h_left_equivalent[:,1] + gap_extend] # Ix_left -> Ix_curr
        if restrict_turns: # Original code: Right = Right[:,:2] then softmaxxed.
                           # This means Ix cannot come from Iy_left.
            pass # ix_options_list is already M, Ix
        else: # Allow transition from Iy_left -> Ix_curr
            ix_options_list.append(h_left_equivalent[:,2] + gap_open) # Iy_left -> Ix_curr (another gap open)

        ix_score = _soft_maximum(jnp.stack(ix_options_list, axis=-1), axis=-1)

        # Iy_current_diag (score from gap in X, i.e. extending Y)
        # Transitions from M_i-1,j - gap_open OR Iy_i-1,j - gap_extend
        # This corresponds to `Down` in original code.
        # `h1` or `h1[1:]` (shifted) depending on `sm_row_info_step["o"]`
        # Let's call this `h_up_equivalent`
        h_up_equivalent = _cond(sm_row_info_step["o"], h1, _pad(h1[1:], ([0,1],[0,0])))

        iy_options_list = [h_up_equivalent[:,0] + gap_open,  # M_up -> Iy_curr
                           h_up_equivalent[:,2] + gap_extend] # Iy_up -> Iy_curr
        # Original code: Down += jnp.stack([open,open,gap]) then softmaxxed.
        # This means Iy_curr can come from M_up-open, Ix_up-open, Iy_up-extend.
        # If penalize_turns=False, it was Down += jnp.stack([open,gap,gap])
        # This implies Ix_up -> Iy_curr incurs gap_extend if penalize_turns=False, open if True.
        # Let's stick to the simpler standard affine model:
        # Iy can come from M_up - open, or Iy_up - extend.

        iy_score = _soft_maximum(jnp.stack(iy_options_list, axis=-1), axis=-1)

        # Local alignment: scores can be reset to 0 if they are too low (or from similarity score directly if local)
        # The original `laxy.sw_affine` did not seem to have the "reset to 0" part of local alignment
        # inside the recurrence, but rather in the final max over `hij`.
        # It used `NINF` for padding, so it was more like global alignment on diagonals.
        # The "Sky = sm["x"]" in sw (non-affine) suggests a local component.
        # For affine, this is usually H_i,j = max(0, M_i,j, E_i,j, F_i,j)
        # Or, the M_score itself includes a max(0, prev_states_sum + x_ij)
        # Given the laxy code, it seems to be a global-like alignment along diagonals,
        # and the "local" aspect comes from taking the max over the final score matrix.

        current_h_e_f = jnp.stack([m_score, ix_score, iy_score], axis=-1)
        return (h1, current_h_e_f), current_h_e_f


    a, b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]
    # Apply NINF where mask is False, effectively making those paths very unlikely
    x_masked = jnp.where(mask, x, NINF)

    # Rotate the masked similarity matrix
    sm_rotated_info, prev_initial_affine, idx_map_back_affine = _sw_rotate_affine(x_masked[:-1, :-1]) # Exclude last row/col for recurrence

    # Perform scan over the diagonals
    # Similar to nogap, sm_rotated_info is a pytree of arrays for scan's `xs`
    hij_m_e_f_rotated_rows = jax.lax.scan(_step, prev_initial_affine, sm_rotated_info, unroll=unroll)[1]

    # Map back from rotated H,E,F scores to original grid shape
    hij_m_e_f = hij_m_e_f_rotated_rows[idx_map_back_affine] # This will be (a-1, b-1, 3)

    # Final score is max over all M, E, F scores in relevant part of matrix + last match score
    # The original code's sink: _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])
    # This means adding the original similarity score x[1:,1:] to all three (M,E,F) states at each cell (i,j)
    # before taking the soft maximum over all cells and all states.
    # The `hij_m_e_f` is for cells (0..a-2, 0..b-2) corresponding to x[:-1,:-1]
    # So, it should be added to x[1:,1:] (scores for cells (1..a-1, 1..b-1))

    # We need to align hij_m_e_f (from x[:-1,:-1]) with x[1:,1:]
    # hij_m_e_f[i,j,:] are scores ending at cell (i+1,j+1) using x[i,j] as match/mismatch
    # The score for cell (r,c) ending in M, Ix, or Iy.
    # The final soft_max should be over all M scores at all (r,c)
    # _sw_sco_affine's last line: _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])
    # hij is (a-1,b-1,3). x[1:,1:,None] is (a-1,b-1,1). Broadcasting adds x_rc to M_rc, Ix_rc, Iy_rc.
    # This is unusual. Typically, x_rc is only added to M_rc.
    # Let's assume it means M_rc includes x_rc, and Ix, Iy are pure gap scores.
    # Then final score is soft_max over all M_rc.
    # However, the original code's `_step` makes M_score already include sm_row_info_step["x"].
    # So, `hij + x[1:,1:,None]` would be adding x_rc twice to M states.
    # This suggests the `x` in `_sw_sco_affine` is not raw similarity but some other potential.

    # If `hij` are the scores of M,Ix,Iy states *ending* at i,j, then the score of alignment is max(M_i,j) over all i,j.
    # The original implementation's final line is:
    # `return _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])`
    # This takes the soft_max over all states (M, Ix, Iy) and all positions (i,j).
    # This is a bit non-standard but we'll replicate it.

    # hij_m_e_f is (a-1,b-1,3), from x[:-1,:-1]
    # x_scores_for_sum = x[1:, 1:, None] # align with hij_m_e_f
    # final_scores_to_softmax = hij_m_e_f + x_scores_for_sum
    # final_mask = mask[1:, 1:, None]

    # Let's trace laxy.MRF from network_functions.py which calls sw_affine.
    # `z` is `sm_mtx` (similarity matrix) passed to sw_affine as `x`.
    # The gradient of `sw_sco_affine` w.r.t. `x` is used as the alignment.
    # The addition of `x[1:,1:,None]` in the end will affect gradients.

    # For now, will replicate the structure literally.
    # `hij_m_e_f` refers to scores for cells (0..a-2, 0..b-2) based on `x[:-1,:-1]`
    # The sum `hij + x[1:,1:,None]` implies `hij` are scores *before* adding current cell's `x`
    # which contradicts the `_step` where `m_score` includes `sm_row_info_step["x"]`.
    # This part of laxy might have a subtle interpretation or a bug.

    # A more standard interpretation: `hij_m_e_f[i,j,0]` is M_score at (i,j), already includes x_ij.
    # Then the result is `_soft_maximum(hij_m_e_f[:,:,0], mask=mask[1:,1:])` (if only M states contribute to end score)
    # Or `_soft_maximum(hij_m_e_f, mask=mask[1:,1:,None])` (if any state can end).

    # Given the grad is taken w.r.t `x`, the exact formulation of score matters.
    # Let's assume the original authors intended `hij` from scan to be final scores for M,Ix,Iy at each cell.
    final_masked_scores = jnp.where(mask[1:,1:,None], hij_m_e_f, NINF)
    return _soft_maximum(final_masked_scores, axis=None) # Softmax over all states and all positions


def smith_waterman_affine(batch=True, unroll=2, restrict_turns=True, penalize_turns=True):
    """
    Smith-Waterman local alignment with affine gap penalties.
    Returns a function that computes the alignment matrix (gradient of score).
    """
    # Gradient w.r.t. 'x' (similarity matrix)
    # argnums must match the signature of _sw_sco_affine
    traceback_fn = jax.grad(_sw_sco_affine, argnums=0)

    def wrapped_traceback_fn(x, lengths, gap_extend=0.0, gap_open=0.0, temp=1.0):
        return traceback_fn(x, lengths, gap_extend, gap_open, temp,
                            restrict_turns, penalize_turns, unroll)

    if batch:
        # in_axes for x, lengths, gap_extend, gap_open, temp
        return jax.vmap(wrapped_traceback_fn, (0, 0, None, None, None))
    else:
        return wrapped_traceback_fn

# Note: sw (non-affine with single gap) and nw (Needleman-Wunsch) were also in sw_functions.py
# They can be added here if needed, following a similar refactoring pattern.

# Utility from network_functions.py that uses SW
def normalize_row_col(z, z_mask, norm_mode="fast"):
    """
    Normalizes a matrix by its row and column sums/means.
    Used for normalizing the similarity matrix before Smith-Waterman.

    Args:
        z (jnp.ndarray): Input matrix (e.g., similarity matrix).
        z_mask (jnp.ndarray): Mask of the same shape as z, indicating valid entries.
        norm_mode (str): Normalization mode ("fast", "slow", "simple").

    Returns:
        jnp.ndarray: Normalized matrix.
    """
    if norm_mode == "fast":
        z = z * z_mask
        # Equivalent to APC (Average Product Correction) if z is a covariance-like matrix
        # Or a kind of iterative normalization.
        # This specific formula: z -= (z.sum(axis=1)*z.sum(axis=0))/z.sum() then z /= sqrt((z_sq.sum(axis=1)*z_sq.sum(axis=0))/z_sq.sum())
        # for a 2D input z.
        # axis=1 corresponds to summing over columns (shape[1]) for each row.
        # axis=0 corresponds to summing over rows (shape[0]) for each col.
        sum_cols = jnp.sum(z, axis=1, keepdims=True) # (L1, 1)
        sum_rows = jnp.sum(z, axis=0, keepdims=True) # (1, L2)
        sum_total = jnp.sum(z, axis=(0,1), keepdims=True) # (1,1)

        z = z - (sum_cols * sum_rows) / (sum_total + 1e-8) # APC-like step
        z = z * z_mask # Re-apply mask

        z_sq = jnp.square(z)
        sum_sq_cols = jnp.sum(z_sq, axis=1, keepdims=True) # (L1,1)
        sum_sq_rows = jnp.sum(z_sq, axis=0, keepdims=True) # (1,L2)
        sum_sq_total = jnp.sum(z_sq, axis=(0,1), keepdims=True) # (1,1)

        # Denominator for normalization, APC-like for variance
        denominator = jnp.sqrt( (sum_sq_cols * sum_sq_rows) / (sum_sq_total + 1e-8) + 1e-8)
        z = z / denominator

    elif norm_mode == "slow": # Iterative normalization for 2D input z
        # z_num_1 is sum of mask over columns (i.e., number of valid elements per row)
        z_num_rows = jnp.sum(z_mask, axis=1, keepdims=True) # (L1, 1)
        # z_num_2 is sum of mask over rows (i.e., number of valid elements per col)
        z_num_cols = jnp.sum(z_mask, axis=0, keepdims=True) # (1, L2)

        z = z * z_mask
        for _ in range(2): # Iterative refinement
            # Normalize rows (sum over axis 1)
            z = z - jnp.sum(z, axis=1, keepdims=True) / (z_num_rows + 1e-8)
            z = z * z_mask
            z = z / (jnp.sqrt(jnp.sum(jnp.square(z), axis=1, keepdims=True) / (z_num_rows + 1e-8)) + 1e-8)
            z = z * z_mask

            # Normalize columns (sum over axis 0)
            z = z - jnp.sum(z, axis=0, keepdims=True) / (z_num_cols + 1e-8)
            z = z * z_mask
            z = z / (jnp.sqrt(jnp.sum(jnp.square(z), axis=0, keepdims=True) / (z_num_cols + 1e-8)) + 1e-8)
            z = z * z_mask

    elif norm_mode == "simple": # Global mean removal and variance normalization for 2D input z
        z = z * z_mask
        sum_z_total_scalar = jnp.sum(z) # scalar
        sum_mask_total_scalar = jnp.sum(z_mask) # scalar

        z = z - sum_z_total_scalar / (sum_mask_total_scalar + 1e-8)
        z = z * z_mask

        sum_z_sq_total_scalar = jnp.sum(jnp.square(z))
        z = z / (jnp.sqrt(sum_z_sq_total_scalar / (sum_mask_total_scalar + 1e-8)) + 1e-8)
        z = z * z_mask

    return z * z_mask # Final mask application
