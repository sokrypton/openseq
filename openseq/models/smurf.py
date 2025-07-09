from .base import Model
# from ..layers.convolution import Conv1D # Example if we create internal layers
# from ..utils.alignment import smith_waterman_nogap, smith_waterman_affine # etc.
import jax
import jax
import jax.numpy as jnp
import optax
from .base import Model
from ..utils.random import get_random_key
from ..utils.data_processing import ALPHABET # Import ALPHABET
from ..layers.convolution import conv1d_init_params

class SMURF(Model):
    def __init__(self,
                 A: int, # Alphabet size
                 L_ref: int, # Reference length for the MRF component
                 filters: int = 512,
                 win: int = 18,
                 mrf_lam: float = 0.01,
                 sw_temp: float = 1.0,
                 sw_open: float = 0.0, # Typically negative if a penalty
                 sw_gap: float = -1.0, # Typically negative if a penalty
                 sw_unroll: int = 4,
                 sw_learn_temp: bool = False,
                 sw_learn_gap: bool = False,
                 norm_mode: str = "fast",
                 ss_hide: float = 0.15, # Self-supervision hide ratio
                 msa_memory_factor: float = 0.0, # Renamed from msa_memory to avoid bool confusion
                 align_to_msa_frac: float = 0.0,
                 pid_thresh: float = 1.0,
                 use_pseudo_alignment: bool = False,
                 learning_rate: float = 0.1,
                 optimizer_type: str = 'adam',
                 seed: int = 0,
                 initial_params: dict = None
                 ):
        """
        Initialize the SMURF model.

        Args:
            A (int): Alphabet size.
            L_ref (int): Length of the reference sequence for the MRF part.
            filters (int): Number of filters for the Conv1D embedding layer.
            win (int): Window size for the Conv1D embedding layer.
            mrf_lam (float): L2 regularization strength for the MRF parameters.
            sw_temp (float): Temperature for Smith-Waterman alignment.
            sw_open (float): Gap open penalty for Smith-Waterman (if learnable or fixed).
            sw_gap (float): Gap extend penalty for Smith-Waterman (if learnable or fixed).
            sw_unroll (int): Unroll factor for Smith-Waterman scan.
            sw_learn_temp (bool): Whether to learn SW temperature.
            sw_learn_gap (bool): Whether to learn SW gap penalties.
            norm_mode (str): Normalization mode for SW similarity matrix.
            ss_hide (float): Fraction of residues to hide for self-supervised learning.
            msa_memory_factor (float): Factor for updating MSA memory (0 for no memory, 1 for replace).
                                      Original `msa_memory` was bool/float. Let's use a factor.
            align_to_msa_frac (float): Fraction for aligning to MSA memory vs. initial reference.
            pid_thresh (float): PID threshold for updating MSA memory.
            use_pseudo_alignment (bool): If True, use softmax-based pseudo-alignment instead of SW.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_type (str): Type of optimizer ('adam', 'sgd').
            seed (int): Seed for JAX PRNG key generation.
            initial_params (dict, optional): Pre-trained parameters.
        """
        super().__init__()

        self.A = A
        self.L_ref = L_ref
        self.filters = filters
        self.win = win
        self.mrf_lam = mrf_lam # Note: MRF class uses `lam`. Be consistent or map.

        self.sw_temp_val = sw_temp
        self.sw_open_val = sw_open
        self.sw_gap_val = sw_gap
        self.sw_unroll = sw_unroll
        self.sw_learn_temp = sw_learn_temp
        self.sw_learn_gap = sw_learn_gap
        self.norm_mode = norm_mode

        self.ss_hide = ss_hide
        self.msa_memory_factor = msa_memory_factor
        self.align_to_msa_frac = align_to_msa_frac
        self.pid_thresh = pid_thresh
        self.use_pseudo_alignment = use_pseudo_alignment

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.key_master = get_random_key(seed)

        if initial_params:
            self.params = initial_params
        else:
            self.params = self._initialize_parameters()

        if self.optimizer_type == 'adam':
            self.optimizer = optax.adam(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optax.sgd(learning_rate=self.learning_rate)
        else:
            print(f"Warning: Unknown SMURF optimizer type '{optimizer_type}'. Defaulting to Adam.")
            self.optimizer = optax.adam(learning_rate=self.learning_rate)

        self.optimizer_state = self.optimizer.init(self.params)

    def _initialize_parameters(self):
        """Helper to initialize SMURF model parameters."""
        k_emb, k_mrf_w, k_mrf_b, k_sw_temp, k_sw_gap = jax.random.split(self.key_master, 5)
        self.key_master = k_emb # Update master key

        params = {}

        # Embedding layer parameters
        # Input to Conv1D is A (alphabet size), output is self.filters
        params['emb'] = conv1d_init_params(
            in_dims=self.A,
            out_dims=self.filters,
            window_size=self.win,
            use_bias=True, # Original Conv1D_custom had bias
            key=k_emb
        )

        # MRF layer parameters (based on L_ref)
        params['mrf'] = {
            'w': jnp.zeros((self.L_ref, self.A, self.L_ref, self.A)),
            'b': jnp.zeros((self.L_ref, self.A))
            # Could add small random noise like in MRF class if beneficial
        }

        # Learnable Smith-Waterman parameters
        if self.sw_learn_temp:
            # Temperature is typically positive. Initialize carefully, e.g. log_temp then exp.
            # For now, direct init.
            params['sw_temp_param'] = jnp.array(self.sw_temp_val)
        else: # Store as fixed value if not learned, for use in alignment function
            params['sw_temp_param'] = jnp.array(self.sw_temp_val) # Will be frozen by not being in trainable params
                                                                 # Or handle as attribute if strictly fixed.

        if self.sw_learn_gap:
            params['sw_gap_params'] = {
                'open': jnp.array(self.sw_open_val), # gap open penalty
                'gap': jnp.array(self.sw_gap_val)    # gap extend penalty
            }
        else:
             params['sw_gap_params'] = { # Store fixed values
                'open': jnp.array(self.sw_open_val),
                'gap': jnp.array(self.sw_gap_val)
            }


        # MSA memory buffer (if used)
        if self.msa_memory_factor > 0:
            # Initialize with zeros, will be filled by reference sequence in fit()
            params['msa_memory_buffer'] = jnp.zeros((self.L_ref, self.A))

        return params

    def _embed_sequences(self, params_emb: dict, x_batch: jnp.ndarray, key_noise: jax.random.PRNGKey = None) -> jnp.ndarray:
        """Applies the Conv1D embedding layer."""
        from ..layers.convolution import conv1d_apply

        # w_scale was a hyperparameter in the original network_functions.MRF for Conv1D noise
        noise_scale = getattr(self, 'w_scale_conv_noise', 0.1) # Use a default if not set as instance attr
                                                              # Or make it a required part of config.

        return conv1d_apply(params_emb, x_batch, add_noise_key=key_noise, noise_scale=noise_scale)

    def _get_sw_similarity_matrix(self, emb_seqs: jnp.ndarray, emb_ref: jnp.ndarray,
                                  lengths_seqs: jnp.ndarray, # Actual lengths of sequences in emb_seqs
                                  # emb_msa_memory is optional, passed if msa_memory_factor > 0
                                  emb_msa_memory: jnp.ndarray = None
                                 ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates the Smith-Waterman similarity matrix and its mask.
        Args:
            emb_seqs: (N, L_seq_max, Filters) - Batch of query sequence embeddings.
            emb_ref: (1, L_ref, Filters) - Reference sequence embedding.
            lengths_seqs: (N,) - Actual lengths of query sequences.
            emb_msa_memory: (L_ref, Filters) - Optional embedding of MSA memory.

        Returns:
            Tuple of (normalized_sim_matrix, sw_mask), shapes
            (N, L_seq_max, L_ref) and (N, L_seq_max, L_ref).
        """
        from ..utils.alignment import normalize_row_col

        # Ensure emb_ref is (L_ref, Filters) for processing
        if emb_ref.ndim == 3 and emb_ref.shape[0] == 1:
            emb_ref_processed = emb_ref[0]
        elif emb_ref.ndim == 2 and emb_ref.shape[0] == self.L_ref: # Should be (L_ref, Filters)
            emb_ref_processed = emb_ref
        else:
            raise ValueError(f"emb_ref has unexpected shape: {emb_ref.shape}, expected (1, {self.L_ref}, Filters) or ({self.L_ref}, Filters)")

        # Combine reference with MSA memory if provided and align_to_msa_frac > 0
        if emb_msa_memory is not None and self.align_to_msa_frac > 0 and self.msa_memory_factor > 0:
            target_emb_for_dot = ( (1.0 - self.align_to_msa_frac) * emb_ref_processed + \
                                   self.align_to_msa_frac * emb_msa_memory ).T # (Filters, L_ref)
        else:
            target_emb_for_dot = emb_ref_processed.T # (Filters, L_ref)

        # sim_matrix: (N, L_seq_max, L_ref)
        sim_matrix = jnp.einsum('nlf,fr->nlr', emb_seqs, target_emb_for_dot)

        # Create mask for similarity matrix based on actual sequence lengths
        L_seq_max_dim = emb_seqs.shape[1] # Max length in the current batch

        seq_indices = jnp.arange(L_seq_max_dim)
        # lengths_seqs is (N,), seq_indices is (L_seq_max_dim,)
        # seq_mask should be (N, L_seq_max_dim)
        seq_mask = (seq_indices[None, :] < lengths_seqs[:, None]).astype(jnp.float32)

        ref_indices = jnp.arange(self.L_ref)
        ref_mask = (ref_indices < self.L_ref).astype(jnp.float32) # length_ref is self.L_ref

        sw_mask = seq_mask[:, :, None] * ref_mask[None, None, :] # (N, L_seq_max_dim, L_ref)

        # Normalize similarity matrix (vmap over batch dimension N)
        normalized_sim_matrix = jax.vmap(normalize_row_col, in_axes=(0, 0, None))(sim_matrix, sw_mask, self.norm_mode)

        return normalized_sim_matrix, sw_mask


    def _align_sequences(self, sim_matrix_batch: jnp.ndarray, sw_mask_batch: jnp.ndarray,
                         lengths_seqs_batch: jnp.ndarray, # Actual lengths of sequences
                         key_align: jax.random.PRNGKey = None
                        ) -> jnp.ndarray:
        """
        Performs Smith-Waterman alignment or pseudo-alignment.
        Args:
            sim_matrix_batch: (N, L_seq_max, L_ref) - Normalized similarity scores.
            sw_mask_batch: (N, L_seq_max, L_ref) - Mask for valid regions in sim_matrix.
            lengths_seqs_batch: (N,) - Actual lengths of query sequences in the batch.
            key_align: PRNG key (currently unused by SW, but good practice).
        Returns:
            Alignment tensor `aln` (N, L_seq_max, L_ref).
        """
        from ..utils.alignment import smith_waterman_affine

        sw_temp_to_use = self.params.get('sw_temp_param', jnp.array(self.sw_temp_val))
        sw_gap_params_dict = self.params.get('sw_gap_params',
                                            {'open': jnp.array(self.sw_open_val),
                                             'gap': jnp.array(self.sw_gap_val)})
        sw_open_to_use = sw_gap_params_dict['open']
        sw_gap_to_use = sw_gap_params_dict['gap']

        # Ensure temperature is positive
        sw_temp_to_use = jnp.maximum(sw_temp_to_use, 1e-6) # Avoid zero or negative temp

        if self.use_pseudo_alignment:
            # Apply mask before softmax to handle variable lengths correctly
            # Masked values in sim_matrix_batch should be large negative for softmax
            masked_sim_matrix = jnp.where(sw_mask_batch.astype(bool), sim_matrix_batch, -1e30) # Apply mask with large neg value

            prob_given_seq_pos = jax.nn.softmax(masked_sim_matrix * sw_temp_to_use, axis=2)
            prob_given_ref_pos = jax.nn.softmax(masked_sim_matrix * sw_temp_to_use, axis=1)
            aln = jnp.sqrt(prob_given_seq_pos * prob_given_ref_pos)
        else:
            batch_size = sim_matrix_batch.shape[0]
            len_ref_tiled = jnp.full_like(lengths_seqs_batch, self.L_ref, dtype=jnp.int32)
            sw_lengths_arg = jnp.stack([lengths_seqs_batch.astype(jnp.int32), len_ref_tiled], axis=-1)

            sw_fn = smith_waterman_affine(batch=True, unroll=self.sw_unroll)

            # Smith-Waterman alignment function expects penalties (positive values).
            # If sw_open_to_use and sw_gap_to_use are initialized as negative (scores), convert them.
            # Assuming they are already positive penalties as per typical SW cost formulation.
            aln = sw_fn(
                sim_matrix_batch,
                sw_lengths_arg,
                sw_gap_to_use,    # gap_extend cost
                sw_open_to_use,   # gap_open cost
                sw_temp_to_use
            )
        return aln

    def _apply_mrf_to_aligned(self, params_mrf: dict, x_aligned: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the MRF layer (w,b) to aligned sequences.
        x_aligned: (N, L_ref, A)
        params_mrf should contain 'w': (L_ref, A, L_ref, A) and 'b': (L_ref, A)
        """
        w = params_mrf['w']
        b = params_mrf['b']

        # Symmetrize and mask diagonal of w (standard MRF practice)
        w_symm = 0.5 * (w + w.transpose((2, 3, 0, 1))) # L_ref,A,L_ref,A
        # Create a mask that is 0 for i=j and 1 otherwise for L_ref indices
        diag_mask_L = (1.0 - jnp.eye(self.L_ref))[:, jnp.newaxis, :, jnp.newaxis] # (L_ref,1,L_ref,1)
        w_final = w_symm * diag_mask_L

        # Compute logits:
        # x_aligned (N, L_ref, A), w_final (L_ref, A, L_ref, A), b (L_ref, A)
        # Logits output shape: (N, L_ref, A)
        # einsum: n=batch, r=L_ref positions in x_aligned, a=alphabet for x_aligned
        #         q=L_ref positions in w_final (summed over), b=alphabet for w_final (summed over)
        # Resulting position should be r (or q), resulting alphabet should be a (or b).
        # Let's use 'nra,raqb->nqb' where q is L_ref_pos_idx, b is Alphabet_idx for that position
        # No, this is what MRF uses: 'nqa,qalp->nlp' where n=N, q=L_ref, a=A, l=L_ref, p=A
        # So logits[n,l,p] = sum_{q,a} x_aligned[n,q,a] * w[q,a,l,p] + b[l,p]
        logits = jnp.einsum('nqa,qalp->nlp', x_aligned, w_final) + b[None, :, :]
        return logits

    def _model_loss_fn(self, params: dict,
                       x_unaligned_batch_one_hot: jnp.ndarray, # (N, L_seq_max, A)
                       lengths_batch: jnp.ndarray,    # (N,)
                       x_ref_one_hot: jnp.ndarray,    # (1, L_ref, A) - initial reference
                       key: jax.random.PRNGKey):
        """
        Computes the forward pass and loss for SMURF.
        """
        N_batch, L_seq_max, _ = x_unaligned_batch_one_hot.shape
        key_ss_mask, key_emb_noise, key_align_noise = jax.random.split(key, 3)

        # 1. Self-Supervision (Masking)
        x_input_for_emb = x_unaligned_batch_one_hot
        x_target_for_loss = x_unaligned_batch_one_hot # Target is original if no masking

        # Create a mask for valid sequence positions (non-padding) for loss calculation later
        seq_indices_loss_mask = jnp.arange(L_seq_max)
        valid_pos_loss_mask = (seq_indices_loss_mask[None, :] < lengths_batch[:, None]) # (N, L_seq_max)

        loss_calc_mask = valid_pos_loss_mask # Default: consider all valid tokens for loss
        if self.ss_hide > 0 and self.ss_hide < 1.0:
            # Only hide valid tokens
            hide_probs = jnp.full_like(valid_pos_loss_mask, self.ss_hide, dtype=jnp.float32)
            effective_hide_probs = hide_probs * valid_pos_loss_mask # Apply ss_hide only to valid tokens

            token_hide_mask = jax.random.bernoulli(key_ss_mask, effective_hide_probs) # (N, L_seq_max), 1 where hidden

            # Masked input: set hidden tokens to zero vector (or a special MASK token if available)
            x_input_for_emb = x_unaligned_batch_one_hot * (1.0 - token_hide_mask[..., None])
            loss_calc_mask = token_hide_mask # For SS, loss is only on hidden tokens

        # 2. Embedding
        emb_x_batch = self._embed_sequences(params['emb'], x_input_for_emb, key_noise=key_emb_noise)
        emb_x_ref = self._embed_sequences(params['emb'], x_ref_one_hot, key_noise=None) # Noise usually not for ref

        # 3. MSA Memory (if active)
        emb_msa_mem_target = None
        if self.msa_memory_factor > 0 and 'msa_memory_buffer' in params:
            emb_msa_mem_target = self._embed_sequences(params['emb'], params['msa_memory_buffer'][None,...], key_noise=None)
            emb_msa_mem_target = emb_msa_mem_target[0] # Remove batch dim -> (L_ref, Filters)

        # 4. Alignment Core
        sim_matrix, sw_mask = self._get_sw_similarity_matrix(
            emb_x_batch, emb_x_ref, lengths_batch, emb_msa_memory=emb_msa_mem_target
        )
        aln = self._align_sequences(sim_matrix, sw_mask, lengths_batch, key_align=key_align_noise)

        # 5. MSA Memory Update
        # This step needs to be handled carefully in the `fit` loop if `params['msa_memory_buffer']`
        # is state that changes across batches/epochs.
        # If `msa_memory_buffer` is part of `params` PyTree, JAX expects it to be returned by this loss function
        # if it's modified. For simplicity of loss_fn, we'll assume msa_memory_buffer update happens
        # in the fit loop based on `aln` and `x_input_for_emb` from this step.
        # Or, it's treated as a non-differentiable update.
        # For now, this function does not return the updated msa_memory_buffer.

        # 6. Align batch sequences to reference space (using x_input_for_emb, which might be masked)
        x_aligned = jnp.einsum('nla,nlr->nra', x_input_for_emb, aln)

        # 7. Apply MRF to aligned sequences
        x_aligned_pred_logits = self._apply_mrf_to_aligned(params['mrf'], x_aligned)

        # 8. Project predictions back to unaligned sequence space
        aln_transpose = aln.transpose((0, 2, 1)) # (N, L_ref, L_seq_max)
        x_unaligned_pred_logits = jnp.einsum('nra,nrl->nla', x_aligned_pred_logits, aln_transpose)

        # 9. Calculate Loss (Categorical Cross-Entropy)
        log_probs = jax.nn.log_softmax(x_unaligned_pred_logits, axis=-1)

        # Element-wise CCE contribution, using x_target_for_loss (original unmasked sequences)
        cce_per_token = -jnp.sum(x_target_for_loss * log_probs, axis=-1) # (N, L_seq_max)
        masked_cce_per_token = cce_per_token * loss_calc_mask # Apply loss_calc_mask (either valid_pos or hidden_tokens)

        num_loss_tokens = jnp.sum(loss_calc_mask) + 1e-9 # Avoid division by zero
        cce_loss = jnp.sum(masked_cce_per_token) / num_loss_tokens

        # 10. Regularization
        reg_loss = 0.0
        if 'w' in params['mrf']: # MRF weights
            reg_loss += 0.5 * self.mrf_lam * jnp.sum(jnp.square(params['mrf']['w']))
        if 'b' in params['mrf']: # MRF biases
            reg_loss += self.mrf_lam * jnp.sum(jnp.square(params['mrf']['b']))

        # Optional: Regularize embedding weights
        # emb_reg_lam = getattr(self, 'emb_lam', 0.0) # if we add this hyperparam
        # if emb_reg_lam > 0 and 'w' in params['emb']:
        #     reg_loss += 0.5 * emb_reg_lam * jnp.sum(jnp.square(params['emb']['w']))

        # Optional: Regularize learnable SW params (e.g., gap penalties)
        # This is less common or might need different regularization form.

        total_loss = cce_loss + reg_loss

        # Return loss and potentially other intermediates for logging or MSA memory update
        # For now, just loss. If msa_memory_buffer is a learnable param, it's handled.
        # If it's updated via side-effect or as state, fit loop needs to manage it.
        return total_loss


    def fit(self, X_unaligned_list: list[str], X_ref_str: str,
            steps: int = 500, batch_size: int = 128,
            learning_rate: float = None, key: jax.random.PRNGKey = None,
            max_len_pad: int = None, # Max length to pad unaligned sequences to
            verbose: bool = True, print_every: int = 10):
        """
        Fit/train the SMURF model.

        Args:
            X_unaligned_list (list[str]): List of unaligned protein/RNA sequences.
            X_ref_str (str): The reference sequence string.
            steps (int): Number of training steps.
            batch_size (int): Batch size for training.
            learning_rate (float, optional): Learning rate. Overrides instance default if provided.
            key (jax.random.PRNGKey, optional): JAX PRNG key. Uses model's master key if None.
            max_len_pad (int, optional): Maximum length to pad sequences in X_unaligned_list.
                                         If None, determined by the longest sequence in X_unaligned_list.
            verbose (bool): If True, print training progress.
            print_every (int): Print loss every `print_every` steps if verbose.

        Returns:
            SMURF: The fitted model instance (self).
        """
        from ..utils.data_processing import mk_msa # For X_ref, potentially for X_unaligned if pre-padded

        N_total = len(X_unaligned_list)
        if N_total == 0:
            if verbose: print("No unaligned sequences provided to fit. Skipping.")
            return self

        # Convert reference sequence to one-hot
        # mk_msa expects a list of sequences.
        x_ref_one_hot = jnp.asarray(mk_msa([X_ref_str])) # Shape (1, L_ref, A)
        assert x_ref_one_hot.shape[1] == self.L_ref, \
            f"Provided X_ref_str length {x_ref_one_hot.shape[1]} != model L_ref {self.L_ref}"

        # Preprocess all unaligned sequences: pad and one-hot encode
        # This can be memory intensive if L_seq_max is very large and N_total is large.
        # Consider batch-wise preprocessing if memory becomes an issue.
        if max_len_pad is None:
            max_len_pad = len(max(X_unaligned_list, key=len)) if X_unaligned_list else 0

        # This utility would convert list of strings to (N, L_max, A) and (N,) lengths
        # For now, let's assume a utility similar to smurf_example.preprocess_unaligned_for_smurf
        # but it should be part of openseq.utils if general enough.
        # For simplicity, let's assume X_unaligned and lengths are pre-processed if not list[str]
        # This part of data handling needs refinement.
        # For now, let's assume a placeholder preprocessing.

        # Placeholder for robust preprocessing:
        # X_unaligned_padded_one_hot, lengths_all = preprocess_unaligned(X_unaligned_list, max_len_pad, self.A)
        # This is a simplified version for now:
        temp_X_padded_list = []
        lengths_list = []
        a2n = {a: i for i, a in enumerate(self.alphabet if hasattr(self, 'alphabet') else ALPHABET)} # Use ALPHABET

        for seq_str in X_unaligned_list:
            lengths_list.append(len(seq_str))
            indices = [a2n.get(c, self.A -1) for c in seq_str]
            if len(indices) < max_len_pad:
                indices.extend([self.A-1]*(max_len_pad - len(indices)))
            elif len(indices) > max_len_pad:
                indices = indices[:max_len_pad] # Truncate
            temp_X_padded_list.append(indices)

        if not temp_X_padded_list: # Should be caught by N_total == 0
             if verbose: print("Processed unaligned sequences list is empty. Skipping fit.")
             return self

        X_unaligned_indices = jnp.array(temp_X_padded_list, dtype=jnp.int32)
        X_unaligned_padded_one_hot = jax.nn.one_hot(X_unaligned_indices, self.A)
        lengths_all = jnp.array(lengths_list, dtype=jnp.int32)
        # End placeholder preprocessing

        if key is None:
            self.key_master, key = jax.random.split(self.key_master)

        # Handle learning rate update for this specific fit call
        current_lr = self.learning_rate
        current_optimizer = self.optimizer
        current_optimizer_state = self.optimizer_state
        if learning_rate is not None and learning_rate != self.learning_rate:
            current_lr = learning_rate
            if self.optimizer_type == 'adam': current_optimizer = optax.adam(learning_rate=current_lr)
            elif self.optimizer_type == 'sgd': current_optimizer = optax.sgd(learning_rate=current_lr)
            else: current_optimizer = optax.adam(learning_rate=current_lr)
            current_optimizer_state = current_optimizer.init(self.params)

        # Initialize MSA memory buffer if active
        if self.msa_memory_factor > 0 and 'msa_memory_buffer' in self.params:
            # Initialize with the reference sequence
            self.params['msa_memory_buffer'] = x_ref_one_hot[0] # (L_ref, A)

        # JIT the training step
        @jax.jit
        def train_step_smurf(params, opt_state, x_unaligned_b, lengths_b, x_ref_b, key_b):
            # Note: x_ref_b will be the same x_ref_one_hot for all batches
            loss_val, grads = jax.value_and_grad(self._model_loss_fn)(
                params, x_unaligned_b, lengths_b, x_ref_b, key_b
            )
            updates, new_opt_state = current_optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # MSA Memory Update (simplified, non-differentiable update within step)
            # A more JAX-idiomatic way might involve passing msa_memory as explicit state.
            if self.msa_memory_factor > 0 and 'msa_memory_buffer' in new_params:
                # This requires _model_loss_fn to also return info needed for update, e.g., x_aligned.
                # For now, this update is conceptual. A proper update would be:
                # x_aligned_batch = ... (recompute or get from _model_loss_fn)
                # msa_bias = x_aligned_batch.mean(0) # (L_ref, A)
                # if pid_thresh logic: filter msa_bias
                # new_msa_mem = (1-self.msa_memory_factor)*new_params['msa_memory_buffer'] + self.msa_memory_factor*msa_bias
                # new_params['msa_memory_buffer'] = new_msa_mem
                # This kind of update inside a JITted function for a parameter being optimized is complex.
                # Often, such stateful updates are handled outside the JITted gradient computation,
                # or the msa_memory_buffer is treated as a separate stateful object.
                # For now, we assume the optimizer handles its update if it's part of `params`.
                # The original code did `params["msa"] = self.p["msa_memory"] * params["msa"] + (1-self.p["msa_memory"])* x_msa_bias_restricted[:p["x_ref_len"],...]`
                # This is a direct update. If `msa_memory_buffer` is in `params`, optax updates it based on its grad.
                # If we want this specific moving average, it needs custom handling.
                # Let's assume for now that if 'msa_memory_buffer' gradients are computed, optax handles it.
                # If not, this is a non-differentiable update to be done outside/carefully.
                pass

            return new_params, new_opt_state, loss_val

        indices = jnp.arange(N_total)
        actual_batch_size = batch_size if batch_size is not None and batch_size < N_total else N_total

        loss_history = []
        if verbose: print(f"Starting SMURF training for {steps} steps with batch size {actual_batch_size}...")

        for step_num in range(steps):
            key, perm_key, step_key = jax.random.split(key, 3)

            if actual_batch_size < N_total:
                permuted_indices = jax.random.permutation(perm_key, indices)
                batch_indices = permuted_indices[:actual_batch_size]
            else:
                batch_indices = indices

            x_unaligned_b = X_unaligned_padded_one_hot[batch_indices, ...]
            lengths_b = lengths_all[batch_indices]

            # x_ref_one_hot is (1, L_ref, A), already prepared

            self.params, current_optimizer_state, loss = train_step_smurf(
                self.params, current_optimizer_state,
                x_unaligned_b, lengths_b, x_ref_one_hot,
                step_key
            )
            loss_history.append(loss.item())

            if verbose and (step_num + 1) % print_every == 0:
                avg_loss = np.mean(loss_history[-(print_every):])
                print(f"Step {step_num + 1}/{steps}, Avg Loss (last {print_every}): {avg_loss:.4f}, Current Batch Loss: {loss:.4f}")

        self.optimizer_state = current_optimizer_state # Persist final optimizer state
        if verbose:
            final_loss = loss_history[-1] if loss_history else float('nan')
            print(f"SMURF fit completed. Final Loss: {final_loss:.4f}")

        return self

    def get_w(self) -> jnp.ndarray:
        """
        Computes the effective MRF coupling matrix (W) from learned SMURF parameters.
        This applies to the MRF component operating in the reference space of length L_ref.
        Handles symmetrization and normalization.
        """
        if self.params is None or 'mrf' not in self.params or self.params['mrf'] is None:
            raise ValueError("Model not yet fit or MRF parameters not initialized.")
        if 'w' not in self.params['mrf'] or self.params['mrf']['w'] is None:
            raise ValueError("MRF coupling parameters 'w' not found or not initialized.")

        current_w = self.params['mrf']['w'] # Shape (L_ref, A, L_ref, A)

        # Ensure current_w is appropriate shape
        expected_shape = (self.L_ref, self.A, self.L_ref, self.A)
        if current_w.shape != expected_shape:
            raise ValueError(f"MRF 'w' has unexpected shape: {current_w.shape}. Expected {expected_shape}.")

        # Symmetrize: W_iajb = W_jbia
        w_symmetrized = 0.5 * (current_w + current_w.transpose((2, 3, 0, 1)))

        # Normalization (mean subtraction per position-pair submatrix)
        mean_per_pos_pair = w_symmetrized.mean(axis=(1,3), keepdims=True)
        w_normalized = w_symmetrized - mean_per_pos_pair

        return w_normalized

    def predict(self, X_unaligned=None, lengths=None, **kwargs):
        """
        Predicts contacts from the learned MRF component of the SMURF model.

        Args:
            X_unaligned, lengths: Currently ignored for contact prediction from learned MRF params,
                                  kept for API consistency.
            alphabet_size_no_gap (int, optional): Size of alphabet excluding gap.
                                                  Defaults to self.A - 1.

        Returns:
            tuple: (raw_contact_map, apc_contact_map) for the reference space (L_ref, L_ref).
        """
        if self.params is None or 'mrf' not in self.params or self.params['mrf'] is None:
            raise ValueError("Model not yet fit or MRF parameters not available.")

        w_couplings = self.get_w() # Get (L_ref, A, L_ref, A) effective couplings

        alphabet_size_no_gap = kwargs.get('alphabet_size_no_gap', self.A - 1 if self.A > 0 else 0)
        if alphabet_size_no_gap <= 0:
            alphabet_size_no_gap = self.A # Default to full alphabet if A-1 is not sensible

        from ..utils.stats import get_mtx
        raw_map, apc_map = get_mtx(w_couplings, alphabet_size_no_gap=alphabet_size_no_gap)

        return raw_map, apc_map

    # get_contacts was a specific method in network_functions.MRF, predict serves this role now.
    # If a distinct get_contacts (e.g. without APC) is needed, it can be added.

    def sample(self, num_samples: int, key: jax.random.PRNGKey,
               burn_in: int = 10, temperature: float = 1.0,
               order: jnp.ndarray = None, # For autoregressive sampling in L_ref space
               return_one_hot: bool = True) -> jnp.ndarray:
        """
        Generate new sequences from the MRF component of the SMURF model.
        These sequences will be of length L_ref (the reference length).
        This method does not perform "un-alignment" to variable length sequences.

        Args:
            num_samples (int): The number of sequences to generate.
            key (jax.random.PRNGKey): JAX PRNG key for sampling.
            burn_in (int): Burn-in steps for Gibbs/sequential sampling.
            temperature (float): Temperature for sampling.
            order (jnp.ndarray, optional): Order for sequential/AR sampling in L_ref space.
            return_one_hot (bool): If True, returns one-hot encoded sequences.

        Returns:
            jax.numpy.ndarray: Generated sequences of shape (num_samples, L_ref, A) or (num_samples, L_ref).
        """
        if self.params is None or 'mrf' not in self.params or self.params['mrf'] is None:
            raise ValueError("Model not yet fit or MRF parameters not initialized for sampling.")

        L_ref, A = self.L_ref, self.A
        mrf_params = self.params['mrf']

        has_couplings = ('w' in mrf_params and mrf_params['w'] is not None)
        use_sequential_logic = has_couplings and (burn_in > 1 or (burn_in == 1 and order is not None))

        msa_shape = (num_samples, L_ref, A)
        msa = jnp.zeros(msa_shape) # Initial state for sampling

        # Internal sampling step (similar to MRF/MRF_BM, but simpler as k=1 for MRF part of SMURF)
        def _smurf_mrf_sample_step(msa_state, current_key_input, iter_pos_idx=None):
            b_eff = mrf_params['b'] # (L_ref, A)
            w_eff = mrf_params.get('w') # (L_ref, A, L_ref, A)

            if iter_pos_idx is not None: # Sequential sampling for one position
                site_bias = b_eff[iter_pos_idx, :] # (A,)
                current_logits = site_bias # (num_samples, A) after broadcasting

                if w_eff is not None and has_couplings:
                    # Field at pos iter_pos_idx from other positions
                    coupling_field = jnp.einsum('njb,ajb->na', msa_state, w_eff[iter_pos_idx, :, :, :])
                    current_logits += coupling_field

                sampled_chars_one_hot = jax.nn.one_hot(
                    jax.random.categorical(current_key_input, current_logits / temperature), A)
                msa_state = msa_state.at[:, iter_pos_idx, :].set(sampled_chars_one_hot)
            else: # Independent sampling from biases
                current_logits_full = jnp.repeat(b_eff[None, ...], num_samples, axis=0) # (num_samples, L_ref, A)
                sampled_chars_one_hot_all_pos = jax.nn.one_hot(
                    jax.random.categorical(current_key_input, current_logits_full / temperature), A)
                msa_state = sampled_chars_one_hot_all_pos
            return msa_state, None

        # Main sampling loop
        if use_sequential_logic:
            for _ in range(burn_in):
                key, iter_key = jax.random.split(key)
                current_order_for_sample = order
                if order is None:
                    current_order_for_sample = jax.random.permutation(iter_key, jnp.arange(L_ref))

                pos_keys = jax.random.split(iter_key, L_ref)

                def body_seq_sample_smurf(carry_msa, pos_and_key_slice):
                    pos_idx, pos_key = pos_and_key_slice
                    new_msa, _ = _smurf_mrf_sample_step(carry_msa, pos_key, iter_pos_idx=pos_idx)
                    return new_msa, None
                msa, _ = jax.lax.scan(body_seq_sample_smurf, msa, (current_order_for_sample, pos_keys))
        else:
            key, final_sample_key = jax.random.split(key)
            msa, _ = _smurf_mrf_sample_step(msa, final_sample_key, iter_pos_idx=None)

        if return_one_hot:
            return msa
        else:
            return msa.argmax(-1)

    def get_parameters(self):
        return self.params

    def load_parameters(self, params):
        self.params = params
        pass

    def get_contacts(self): # Specific method from network_functions.MRF
        # Placeholder, similar to predict() but might return raw 'w' or APC matrix directly
        if self.params["mrf"] is None:
            raise ValueError("Model not yet fit.")
        # Placeholder
        return None
