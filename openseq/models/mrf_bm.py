import jax
import jax.numpy as jnp
import optax
from .base import Model
from ..utils.random import get_random_key
# from .kmeans import KMeans # Potentially for initializing mixture biases if k > 1

class MRF_BM(Model):
    def __init__(self, L: int, A: int, k: int = 1,
                 lam: float = 0.01, learning_rate: float = 0.01,
                 optimizer_type: str = 'adam', seed: int = 0,
                 initial_params: dict = None,
                 # BM-specific sampling parameters for Contrastive Divergence (CD)
                 num_cd_samples: int = 1000,
                 cd_burn_in: int = 10,
                 cd_temperature: float = 1.0):
        """
        Initialize the Boltzmann Machine Markov Random Field (MRF_BM) model.

        Args:
            L (int): Length of the sequences.
            A (int): Alphabet size.
            k (int): Number of mixture components. Defaults to 1 (no mixture).
            lam (float): L2 regularization strength. Defaults to 0.01.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.01.
            optimizer_type (str): Type of optimizer to use ('adam', 'sgd', etc.). Defaults to 'adam'.
            seed (int): Seed for JAX PRNG key generation. Defaults to 0.
            initial_params (dict, optional): Pre-trained parameters to load. Defaults to None.
            num_cd_samples (int): Number of samples for the negative phase in CD.
            cd_burn_in (int): Burn-in steps for Gibbs sampling in CD.
            cd_temperature (float): Temperature for sampling in CD.
        """
        super().__init__()

        self.L = L
        self.A = A
        self.k = k
        self.lam = lam
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.key_master = get_random_key(seed)

        # BM-specific settings
        self.num_cd_samples = num_cd_samples
        self.cd_burn_in = cd_burn_in
        self.cd_temperature = cd_temperature

        # Initialize parameters
        if initial_params:
            self.params = initial_params
        else:
            self.params = self._initialize_parameters() # Basic init, data-dependent can be in fit

        # Initialize optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = optax.adam(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optax.sgd(learning_rate=self.learning_rate)
        else:
            print(f"Warning: Unknown optimizer type '{optimizer_type}'. Defaulting to Adam.")
            self.optimizer = optax.adam(learning_rate=self.learning_rate)

        self.optimizer_state = self.optimizer.init(self.params)

    def _initialize_parameters(self):
        """
        Helper function to initialize model parameters.
        For MRF_BM, parameters are often initialized to zeros or small random values.
        Data-dependent initialization (like using f_i for biases 'b', or k-means for 'mb')
        is deferred to the `fit` method or a separate pre-training step, as it requires data.
        """
        self.key_master, key_w, key_b, key_c = jax.random.split(self.key_master, 4)

        params = {}

        if self.k == 1:
            # w: couplings (L, A, L, A)
            # b: biases (L, A)
            params['w'] = jnp.zeros((self.L, self.A, self.L, self.A))
            params['b'] = jnp.zeros((self.L, self.A))
            # Note: Original mrf.py:MRF_BM initialized 'b' using log of data frequencies (f_i).
            # This requires data, so we'll handle it in fit() if adopted.
        else:
            # For mixtures (k > 1)
            # params['w'] for shared couplings, or params['mw'] for mixture-specific.
            # params['b'] for shared biases, or params['mb'] for mixture-specific.
            # params['c'] for mixture weights/logits.

            # Following the structure of original mrf.py:MRF_BM, which had 'mode' (tied/full)
            # For a simple mixture model where each component has its own 'w' and 'b':
            # This would mean params['w'] is (k,L,A,L,A) and params['b'] is (k,L,A)
            # And params['c'] is (k,)
            # However, the original MRF_BM used:
            # inc = ["b","mb"] if k > 1 else ["b"]
            # if mode == "tied": inc += ["w"]
            # if mode == "full": inc += ["w","mw"] if k > 1 else ["w"]
            # This implies 'w' is shared, 'b' is shared. 'mb' is mixture-specific bias offset. 'mw' is mixture-specific W offset.

            # Let's start with a simpler BM mixture: shared w, shared b, mixture-specific bias offsets (mb), and mixture weights (c).
            # This means:
            # 'w': (L,A,L,A) - shared couplings
            # 'b': (L,A) - shared biases
            # 'mb': (k,L,A) - mixture-specific bias components (offsets from 'b')
            # 'c': (k,) - mixture logits

            # For this iteration, to match MRF's current mixture param structure for consistency:
            # params['w'] = (k,L,A,L,A)
            # params['b'] = (k,L,A)
            # params['c'] = (k,)
            # This means each mixture component has its own full set of w_k and b_k.

            params['w'] = jnp.zeros((self.k, self.L, self.A, self.L, self.A))
            params['b'] = jnp.zeros((self.k, self.L, self.A))
            params['c'] = jnp.zeros((self.k,)) # Logits for mixture components

            # Original MRF_BM initialized params['b'] from data f_i,
            # and params['mb'] from k-means derived mixture means (mf_i),
            # and params['c'] from k-means category proportions.
            # This data-dependent initialization will be handled in fit() or a pretrain step.

        return params

    def _regularization_loss(self, params: dict) -> float:
        """Computes L2 regularization loss for MRF_BM parameters."""
        reg_loss_val = 0.0

        if self.k == 1:
            if 'w' in params:
                reg_loss_val += 0.5 * self.lam * jnp.sum(jnp.square(params['w']))
            if 'b' in params:
                reg_loss_val += self.lam * jnp.sum(jnp.square(params['b']))
        else: # k > 1
            if 'w' in params:
                reg_loss_val += 0.5 * self.lam * jnp.sum(jnp.square(params['w']))
            if 'b' in params:
                reg_loss_val += self.lam * jnp.sum(jnp.square(params['b']))
            if 'c' in params:
                reg_loss_val += self.lam * jnp.sum(jnp.square(params['c']))
        return reg_loss_val

    def _get_data_labels_for_positive_phase(self, params: dict, msa_batch: jnp.ndarray,
                                            weights_batch: jnp.ndarray, key: jax.random.PRNGKey):
        """
        Helper to get or infer one-hot labels for the data batch for positive phase stats when k > 1.
        This version infers P(z|x) and assigns to the most likely component (hard assignment).
        A more advanced version might use expected statistics or fixed labels from initial clustering.
        """
        if self.k <= 1:
            return None

        # Infer P(z_n=k | x_n, params)
        w_k_pos = params['w'] # Shape (k,L,A,L,A)
        b_k_pos = params['b'] # Shape (k,L,A)
        c_log_probs_pos = jax.nn.log_softmax(params['c']) # Shape (k,)

        # Logits for each sequence, for each component, for each site, for each AA
        # msa_batch (N,L,A), w_k_pos (k,L,A,L,A) -> einsum 'nqa,kpalq->nkpa'
        # b_k_pos (k,L,A)
        logits_per_k_pos = jnp.einsum('nqa,kpalq->nkpa', msa_batch, w_k_pos) + b_k_pos[None, :, :, :] # (N_batch, k, L, A)

        # Log pseudo-likelihood of each sequence under each component
        # (msa_batch (N,1,L,A) * log_softmax(logits_per_k_pos (N,k,L,A))) -> sum over A, then L
        site_loglik_per_k_pos = (msa_batch[:,None,:,:] * jax.nn.log_softmax(logits_per_k_pos, axis=-1)).sum(axis=-1) # (N_batch, k, L)
        seq_loglik_per_k_pos = site_loglik_per_k_pos.sum(axis=-1) # (N_batch, k)

        # Posterior log p(z_n=k | x_n) ~ log p(z_n=k) + log p(x_n | z_n=k)
        log_posterior_z_pos = c_log_probs_pos[None,:] + seq_loglik_per_k_pos # (N_batch, k)

        # Hard assignment for stats:
        data_labels_int = log_posterior_z_pos.argmax(axis=1) # (N_batch,)
        data_labels_one_hot = jax.nn.one_hot(data_labels_int, self.k) # (N_batch, k)
        return data_labels_one_hot

    def _compute_bm_gradients(self, params: dict,
                              msa_batch: jnp.ndarray, weights_batch: jnp.ndarray,
                              key: jax.random.PRNGKey, # Key moved before optional arg
                              positive_phase_labels_one_hot: jnp.ndarray = None
                              ):
        """
        Computes gradients for MRF_BM using Contrastive Divergence.
        The 'gradient' for optax is E_model[stats] - E_data[stats] + d(RegLoss)/d(params),
        as optimizers typically minimize.
        """
        N_batch, L, A = msa_batch.shape
        # N_eff_batch = jnp.sum(weights_batch) + 1e-9 # Not explicitly used if stats are frequencies

        from ..utils.stats import get_stats as compute_stats_util
        key_pos_labels, key_neg_init, key_neg_gibbs, key_neg_stats = jax.random.split(key, 4)

        # --- Positive Phase ---
        positive_phase_labels = None
        if self.k > 1:
            # For positive phase, if k>1, we need component assignments for the data.
            # This can be from an initial k-means (passed to fit) or inferred.
            # Here, we infer based on current parameters.
            if hasattr(self, '_current_batch_data_labels_one_hot'): # Set by fit method
                positive_phase_labels = self._current_batch_data_labels_one_hot
            else: # Fallback: infer (less ideal for stable CD training if labels fluctuate wildly)
                 positive_phase_labels = self._get_data_labels_for_positive_phase(params, msa_batch, weights_batch, key_pos_labels)

        stats_positive = compute_stats_util(
            msa_batch, weights_batch,
            labels=positive_phase_labels, # Will be None if k=1
            add_f_ij=('w' in params and (self.k == 1 or 'w_shared' in params)), # Simplified logic
            add_mf_ij=('w' in params and self.k > 1)
        )
        if self.k > 1 and positive_phase_labels is not None:
            stats_positive['mc'] = jnp.sum(positive_phase_labels * weights_batch[:,None], axis=0) / (jnp.sum(weights_batch) + 1e-9)
        elif self.k > 1: # Should have labels if k>1
             stats_positive['mc'] = jnp.ones(self.k) / self.k


        # --- Negative Phase ---
        msa_neg_shape = (self.num_cd_samples, L, A)
        neg_sampled_mixture_indices = None # For k > 1

        if self.k == 1:
            b_for_init_neg = params['b']
            logits_init_neg = jnp.repeat(b_for_init_neg[None,...], self.num_cd_samples, axis=0)
        else:
            neg_mixture_logits = params['c']
            key_neg_mix_idx, key_neg_init = jax.random.split(key_neg_init) # Split from neg_init key
            neg_sampled_mixture_indices = jax.random.categorical(
                key_neg_mix_idx, neg_mixture_logits, shape=(self.num_cd_samples,)
            )
            b_for_init_neg = params['b'][neg_sampled_mixture_indices]
            logits_init_neg = b_for_init_neg

        msa_neg = jax.nn.one_hot(
            jax.random.categorical(key_neg_init, logits_init_neg / self.cd_temperature), A
        )

        def _negative_gibbs_step_scan_body(msa_state_neg, slice_data):
            pos_idx_to_sample_neg, key_input_step = slice_data

            # Determine effective w and b for negative sampling based on mixture or k=1
            if self.k == 1:
                b_eff_neg = params['b']
                w_eff_neg = params.get('w')
            else: # k > 1, neg_sampled_mixture_indices is defined
                b_eff_neg = params['b'][neg_sampled_mixture_indices]
                w_eff_neg = params.get('w')[neg_sampled_mixture_indices]

            site_bias_neg = b_eff_neg[..., pos_idx_to_sample_neg, :]
            if self.k == 1 and b_eff_neg.ndim == 2:
                site_bias_neg = b_eff_neg[pos_idx_to_sample_neg, :]

            current_logits_neg = site_bias_neg

            if w_eff_neg is not None:
                if self.k == 1:
                    coupling_field_neg = jnp.einsum('njb,ajb->na',
                                                    msa_state_neg, w_eff_neg[pos_idx_to_sample_neg, :, :, :])
                else:
                    coupling_field_neg = jnp.einsum('njb,najb->na',
                                                    msa_state_neg, w_eff_neg[:, pos_idx_to_sample_neg, :, :, :])
                current_logits_neg += coupling_field_neg

            sampled_chars_neg = jax.nn.one_hot(
                jax.random.categorical(key_input_step, current_logits_neg / self.cd_temperature), A)
            return msa_state_neg.at[:, pos_idx_to_sample_neg, :].set(sampled_chars_neg), None

        # Gibbs sampling loop
        msa_neg_final = msa_neg
        key_gibbs_loop_master = key_neg_gibbs # Use the key_neg_gibbs from the initial split
        for _ in range(self.cd_burn_in):
            # Split the master Gibbs key for this iteration's needs
            key_gibbs_loop_master, perm_key_neg, iter_keys_key_for_scan = jax.random.split(key_gibbs_loop_master, 3)

            current_perm_order_neg = jax.random.permutation(perm_key_neg, jnp.arange(L))
            pos_sample_keys_neg = jax.random.split(iter_keys_key_for_scan, L)

            msa_neg_final, _ = jax.lax.scan(
                _negative_gibbs_step_scan_body,
                msa_neg_final,
                (current_perm_order_neg, pos_sample_keys_neg)
            )

        weights_negative = jnp.ones(self.num_cd_samples) / self.num_cd_samples
        neg_labels_one_hot_for_stats = jax.nn.one_hot(neg_sampled_mixture_indices, self.k) if self.k > 1 else None

        stats_negative = compute_stats_util(
            msa_neg_final, weights_negative,
            labels=neg_labels_one_hot_for_stats,
            add_f_ij=('w' in params and (self.k == 1 or 'w_shared' in params)),
            add_mf_ij=('w' in params and self.k > 1)
        )
        if self.k > 1 and neg_labels_one_hot_for_stats is not None:
            stats_negative['mc'] = jnp.sum(neg_labels_one_hot_for_stats * weights_negative[:,None], axis=0)
        elif self.k > 1:
             stats_negative['mc'] = jnp.ones(self.k) / self.k


        # --- Gradient Calculation (Model Stats - Data Stats for minimization) ---
        computed_grads = {}
        if self.k == 1:
            computed_grads['b'] = (stats_negative['f_i'] - stats_positive['f_i'])
            if 'w' in params:
                 w_grad_mask = (jnp.arange(L)[:, None] != jnp.arange(L)[None, :])[:, None, :, None].astype(params['w'].dtype)
                 computed_grads['w'] = (stats_negative['f_ij'] - stats_positive['f_ij']) * w_grad_mask
        else: # k > 1
            # Ensure positive stats have mixture versions if needed
            pos_mf_i = stats_positive.get('mf_i', jnp.zeros_like(stats_negative['mf_i']))
            pos_mf_ij = stats_positive.get('mf_ij', jnp.zeros_like(stats_negative['mf_ij']))
            pos_mc = stats_positive.get('mc', jnp.zeros_like(stats_negative['mc']))

            computed_grads['b'] = (stats_negative['mf_i'] - pos_mf_i)
            if 'w' in params:
                w_grad_mask_k = (jnp.arange(L)[None,:,None] != jnp.arange(L)[None,None,:] )[None,:,None,:,None].astype(params['w'].dtype)
                computed_grads['w'] = (stats_negative['mf_ij'] - pos_mf_ij) * w_grad_mask_k
            computed_grads['c'] = (stats_negative['mc'] - pos_mc)

        # Add regularization gradient
        reg_grads = jax.grad(self._regularization_loss)(params)
        for p_name in computed_grads:
            if p_name in reg_grads:
                computed_grads[p_name] += reg_grads[p_name]
            elif p_name in params: # If param exists but no stat-diff grad, ensure it still has reg grad if any
                if p_name not in reg_grads: # Should not happen if _regularization_loss covers all params
                     computed_grads[p_name] += jnp.zeros_like(params[p_name])

        for p_name in reg_grads: # Add reg_grads for params not involved in stat diffs
            if p_name not in computed_grads and p_name in params:
                computed_grads[p_name] = reg_grads[p_name]

        return computed_grads

    def fit(self, X: jnp.ndarray, X_weight: jnp.ndarray = None,
            steps: int = 1000, batch_size: int = None,
            learning_rate: float = None, key: jax.random.PRNGKey = None,
            # ar_options for BM are less common, usually full model.
            # For now, BM fit won't use AR specific logic beyond what params structure implies.
            initial_kmeans_labels_for_mixtures: bool = True, # If k>1, use kmeans for initial pos phase labels
            verbose: bool = True, print_every: int = 10):
        """
        Fit/train the MRF_BM model using Contrastive Divergence.

        Args:
            X (jnp.ndarray): One-hot encoded MSA of shape (N, L, A).
            X_weight (jnp.ndarray, optional): Sequence weights of shape (N,). Defaults to uniform.
            steps (int): Number of training steps. Defaults to 1000.
            batch_size (int, optional): Batch size. If None, uses full batch. Defaults to None.
            learning_rate (float, optional): Learning rate. If None, uses rate from __init__.
            key (jax.random.PRNGKey, optional): JAX PRNG key. If None, uses model's master key.
            initial_kmeans_labels_for_mixtures (bool): If k > 1, whether to run KMeans on X
                to get initial labels for the positive phase statistics and initialize mixture params 'c' and 'b'.
                If False, positive phase labels for mixtures will be inferred using `_get_data_labels_for_positive_phase`
                and 'c'/'b' use their zero/default initialization. Defaults to True.
            verbose (bool): If True, print training progress. Defaults to True.
            print_every (int): Print stats every `print_every` steps. Defaults to 10.

        Returns:
            MRF_BM: The fitted model instance (self).
        """
        N, L_data, A_data = X.shape
        assert L_data == self.L and A_data == self.A, "Data dimensions mismatch model config."

        if X_weight is None:
            X_weight = jnp.ones(N) / N
        else:
            X_weight = jnp.asarray(X_weight)
            X_weight = X_weight / (jnp.sum(X_weight) + 1e-9) # Normalize to sum to 1

        if key is None:
            self.key_master, key = jax.random.split(self.key_master)

        current_lr = self.learning_rate
        if learning_rate is not None and learning_rate != self.learning_rate:
            current_lr = learning_rate
            if self.optimizer_type == 'adam': self.optimizer = optax.adam(learning_rate=current_lr)
            elif self.optimizer_type == 'sgd': self.optimizer = optax.sgd(learning_rate=current_lr)
            else: self.optimizer = optax.adam(learning_rate=current_lr)
            self.optimizer_state = self.optimizer.init(self.params)

        # Data-dependent initialization for k > 1 using K-Means (optional)
        # This also sets self._fixed_positive_phase_labels if used.
        self._current_batch_data_labels_one_hot = None # For _compute_bm_gradients positive phase if k > 1

        # Store all data labels if k-means is used for consistent positive phase stats across batches/epochs
        all_data_positive_labels_one_hot = None

        if self.k > 1 and initial_kmeans_labels_for_mixtures:
            if verbose: print(f"Running initial KMeans for {self.k} mixtures...")
            from .kmeans import KMeans # Local import to avoid circularity at module level
            kmeans_model = KMeans(n_clusters=self.k, seed=jax.random.randint(key, (), 0, 100000).item())
            key, kmeans_key = jax.random.split(key) # kmeans uses its own seed mechanism

            # KMeans expects (N, Features). Reshape MSA X if it's (N,L,A)
            X_for_kmeans = X.reshape(N, -1) if X.ndim > 2 else X
            kmeans_model.fit(X_for_kmeans, X_weight=X_weight) # Pass original X_weight before batch normalization

            all_data_positive_labels_int = kmeans_model.labels_ # (N,) integer labels
            all_data_positive_labels_one_hot = jax.nn.one_hot(all_data_positive_labels_int, self.k) # (N, k)

            # Initialize/update params['c'] and params['b'] (mixture biases) based on k-means
            # This follows logic from original mrf.py:MRF_BM initialization
            if 'c' in self.params:
                # kmeans_model.cat_ are proportions (k,)
                # Add small epsilon before log for stability
                self.params['c'] = jnp.log(kmeans_model.cat_ + 1e-8)
                # self.params['c'] = self.params['c'] - self.params['c'].mean() # Optional: center logits

            if 'b' in self.params and self.params['b'].shape[0] == self.k : # params['b'] is (k,L,A)
                # Initialize component biases based on means of sequences in each cluster
                # This requires computing mean one-hot sequence for each cluster.
                # kmeans_model.means_ are (k, L*A) if X_for_kmeans was (N, L*A)
                # We need to reshape them back to (k, L, A)
                cluster_means_reshaped = kmeans_model.means_.reshape(self.k, self.L, self.A)

                # Original MRF_BM: mb = jnp.log(kms["means"] * Neff + small_val) - log(Neff)
                # Here, cluster_means_reshaped are already like probabilities (if from one-hot data)
                # Let's use log directly, similar to how 'b' might be init from f_i.
                # Add small epsilon for log stability.
                self.params['b'] = jnp.log(cluster_means_reshaped + 1e-8)
                # Optional: if there's a shared 'b', then this would be mb_delta = log_means - b_shared

            # Re-initialize optimizer state as params changed
            self.optimizer_state = self.optimizer.init(self.params)
            if verbose: print("KMeans initialization for mixtures complete. Updated params['c'] and params['b'].")


        @jax.jit
        def train_step_bm(params, opt_state, msa_b, weights_b, positive_labels_b, key_b):
            # _compute_bm_gradients expects gradients that *maximize* data likelihood / *minimize* model likelihood
            # Optax optimizers *minimize* a loss. So, if grads are (E_model - E_data), optimizer uses them directly.
            grads = self._compute_bm_gradients(params, msa_b, weights_b, key_b, positive_labels_b) # Corrected arg order
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            # For BM, loss is not typically tracked directly in the same way as PLL.
            # One might track pseudo-log-likelihood on a held-out set, or reconstruction error.
            # For now, we just return new_params and opt_state.
            return new_params, new_opt_state

        indices = jnp.arange(N)
        current_batch_size = batch_size if batch_size is not None and batch_size < N else N

        for step_num in range(steps):
            step_key, perm_key, grad_key = jax.random.split(key, 3)
            key = step_key

            if current_batch_size < N:
                permuted_indices = jax.random.permutation(perm_key, indices)
                batch_indices = permuted_indices[:current_batch_size]
            else:
                batch_indices = indices

            msa_batch_data = X[batch_indices, ...]
            weights_batch_data = X_weight[batch_indices] # Already normalized if done above

            current_pos_labels_batch = None
            if self.k > 1:
                if all_data_positive_labels_one_hot is not None: # From initial K-Means
                    current_pos_labels_batch = all_data_positive_labels_one_hot[batch_indices, ...]
                else: # Infer dynamically (less stable for CD usually)
                    self.key_master, pos_label_key = jax.random.split(self.key_master)
                    current_pos_labels_batch = self._get_data_labels_for_positive_phase(
                        self.params, msa_batch_data, weights_batch_data, pos_label_key
                    )
            # For _compute_bm_gradients, ensure this attribute is set if k > 1
            # This is a bit hacky, better to pass labels directly to _compute_bm_gradients.
            # I've updated _compute_bm_gradients to take positive_phase_labels_one_hot.

            self.params, self.optimizer_state = train_step_bm(
                self.params, self.optimizer_state,
                msa_batch_data, weights_batch_data,
                current_pos_labels_batch, # Pass labels for positive phase if k > 1
                grad_key
            )

            if verbose and (step_num + 1) % print_every == 0:
                # For BM, direct loss is not usually tracked per step.
                # Could compute pseudo-log-likelihood on batch as a proxy if needed.
                print(f"Step {step_num + 1}/{steps} completed.")

        if verbose:
            print(f"MRF_BM fit completed.")
        return self

    def sample(self, num_samples: int, key: jax.random.PRNGKey,
               burn_in: int = 10, temperature: float = 1.0,
               order: jnp.ndarray = None,
               return_one_hot: bool = True):
        """
        Generate new sequences from the trained MRF_BM model.
        This logic is identical to MRF.sample, using the learned parameters.

        Args:
            num_samples (int): The number of sequences to generate.
            key (jax.random.PRNGKey): JAX PRNG key for sampling.
            burn_in (int): Number of burn-in steps for Gibbs/sequential sampling.
            temperature (float): Temperature for sampling. Defaults to 1.0.
            order (jnp.ndarray, optional): Order of positions for sequential/autoregressive sampling.
            return_one_hot (bool): If True, returns one-hot encoded sequences. Defaults to True.

        Returns:
            jax.numpy.ndarray: Generated sequences.
        """
        if self.params is None:
            raise ValueError("Model not yet fit. Parameters are not available.")

        L, A = self.L, self.A
        has_couplings = ('w' in self.params and self.params['w'] is not None)
        use_sequential_logic = has_couplings and (burn_in > 1 or (burn_in == 1 and order is not None))

        current_params_for_sampling = self.params.copy()
        # No need to add L, A, k to current_params_for_sampling, they are instance attributes

        msa_shape = (num_samples, L, A)
        msa = jnp.zeros(msa_shape)

        if self.k > 1:
            if 'c' not in current_params_for_sampling:
                raise ValueError("Mixture weights 'c' not found for sampling with k > 1.")
            key, labels_key = jax.random.split(key)
            mixture_logits = current_params_for_sampling['c']
            sampled_mixture_indices = jax.random.categorical(labels_key, mixture_logits, shape=(num_samples,))
            current_params_for_sampling['sampled_mixture_indices'] = sampled_mixture_indices

        def _sample_step_pll_bm(msa_state, current_key_input, iter_pos_idx=None):
            current_logits_full_shape = (num_samples, L, A)
            current_logits_single_pos_shape = (num_samples, A)

            if self.k == 1:
                b_eff = current_params_for_sampling['b']
                w_eff = current_params_for_sampling.get('w')
            else:
                b_eff = current_params_for_sampling['b'][current_params_for_sampling['sampled_mixture_indices']]
                w_eff = current_params_for_sampling.get('w')[current_params_for_sampling['sampled_mixture_indices']]

            if iter_pos_idx is not None:
                site_bias = b_eff[..., iter_pos_idx, :]
                if self.k == 1 and b_eff.ndim == 2: site_bias = b_eff[iter_pos_idx, :]

                current_logits = site_bias
                if w_eff is not None and has_couplings:
                    if self.k == 1:
                        coupling_field = jnp.einsum('njb,ajb->na', msa_state, w_eff[iter_pos_idx, :, :, :])
                    else:
                        coupling_field = jnp.einsum('njb,najb->na', msa_state, w_eff[:, iter_pos_idx, :, :, :])
                    current_logits += coupling_field

                sampled_chars_one_hot = jax.nn.one_hot(
                    jax.random.categorical(current_key_input, current_logits / temperature), A)
                msa_state = msa_state.at[:, iter_pos_idx, :].set(sampled_chars_one_hot)
            else:
                if self.k == 1:
                    current_logits_full = jnp.repeat(b_eff[None, ...], num_samples, axis=0)
                else:
                    current_logits_full = b_eff

                sampled_chars_one_hot_all_pos = jax.nn.one_hot(
                    jax.random.categorical(current_key_input, current_logits_full / temperature), A)
                msa_state = sampled_chars_one_hot_all_pos
            return msa_state, None

        if use_sequential_logic:
            for _ in range(burn_in):
                key, iter_key = jax.random.split(key)
                current_order_for_sample = order
                if order is None:
                    current_order_for_sample = jax.random.permutation(iter_key, jnp.arange(L))

                pos_keys = jax.random.split(iter_key, L)

                def body_seq_sample_bm(carry_msa, pos_and_key_slice):
                    pos_idx, pos_key = pos_and_key_slice
                    new_msa, _ = _sample_step_pll_bm(carry_msa, pos_key, iter_pos_idx=pos_idx)
                    return new_msa, None
                msa, _ = jax.lax.scan(body_seq_sample_bm, msa, (current_order_for_sample, pos_keys))
        else:
            key, final_sample_key = jax.random.split(key)
            msa, _ = _sample_step_pll_bm(msa, final_sample_key, iter_pos_idx=None)

        if return_one_hot:
            return msa
        else:
            return msa.argmax(-1)

    def get_w(self) -> jnp.ndarray:
        """
        Computes the effective coupling matrix (W) from learned parameters.
        Handles symmetrization, normalization, and combines mixture parameters if k > 1.
        This logic is identical to MRF.get_w().
        """
        if self.params is None:
            raise ValueError("Model not yet fit. Parameters are not available.")

        current_w = None
        if self.k == 1:
            if 'w' not in self.params:
                raise ValueError("Parameters 'w' not found in model for k=1.")
            current_w = self.params['w']
            if current_w is None:
                raise ValueError("DEBUG: self.params['w'] was None in get_w for k=1 after assignment.")
        else: # k > 1 (mixtures)
            if 'w' not in self.params or 'c' not in self.params:
                raise ValueError("Parameters 'w' (for mixtures) and 'c' not found for k>1.")

            mixture_weights = jax.nn.softmax(self.params['c'])
            current_w = jnp.einsum('k,kijab->ijab', mixture_weights, self.params['w'])

        if current_w is None:
             raise ValueError("Effective 'w' is None before processing (e.g. after mixture einsum).")

        if current_w.ndim != 4:
            raise ValueError(f"Effective 'w' has unexpected ndim: {current_w.ndim}. Expected 4 (L,A,L,A).")

        w_symmetrized = 0.5 * (current_w + current_w.transpose((2, 3, 0, 1)))
        mean_per_pos_pair = w_symmetrized.mean(axis=(1,3), keepdims=True)
        w_normalized = w_symmetrized - mean_per_pos_pair

        return w_normalized

    def predict(self, X=None, **kwargs):
        """
        Predicts contacts from the learned MRF_BM model.
        This involves computing the effective coupling matrix W and then applying APC.
        Logic is identical to MRF.predict().

        Args:
            X: This argument is currently ignored but kept for API consistency.
            alphabet_size_no_gap (int, optional): Size of alphabet excluding gap. Defaults to self.A - 1.

        Returns:
            tuple: (raw_contact_map, apc_contact_map)
        """
        if self.params is None:
            raise ValueError("Model not yet fit.")

        w_couplings = self.get_w()

        alphabet_size_no_gap = kwargs.get('alphabet_size_no_gap', self.A - 1 if self.A > 0 else 0)
        if alphabet_size_no_gap <= 0:
            alphabet_size_no_gap = self.A

        from ..utils.stats import get_mtx
        raw_map, apc_map = get_mtx(w_couplings, alphabet_size_no_gap=alphabet_size_no_gap)

        return raw_map, apc_map

    def get_parameters(self):
        return self.params

    def load_parameters(self, params: dict):
        """
        Load pre-trained parameters into the model.
        This also re-initializes the optimizer state.
        """
        self.params = params
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer_state = self.optimizer.init(self.params)
        else:
            self.optimizer_state = None
