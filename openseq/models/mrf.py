import jax
import jax.numpy as jnp
import optax
from .base import Model
from ..utils.random import get_random_key # For default key generation

class MRF(Model):
    def __init__(self, L: int, A: int, k: int = 1,
                 lam: float = 0.01, learning_rate: float = 0.01,
                 optimizer_type: str = 'adam', seed: int = 0,
                 initial_params: dict = None):
        """
        Initialize the Markov Random Field (MRF) model.

        Args:
            L (int): Length of the sequences.
            A (int): Alphabet size.
            k (int): Number of mixture components. Defaults to 1 (no mixture).
            lam (float): L2 regularization strength. Defaults to 0.01.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.01.
            optimizer_type (str): Type of optimizer to use ('adam', 'sgd', etc.). Defaults to 'adam'.
            seed (int): Seed for JAX PRNG key generation. Defaults to 0.
            initial_params (dict, optional): Pre-trained parameters to load. Defaults to None.
        """
        super().__init__() # Call base class init if it has one

        self.L = L
        self.A = A
        self.k = k
        self.lam = lam
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.key_master = get_random_key(seed) # Master key for this model instance

        # Initialize parameters
        if initial_params:
            self.params = initial_params
        else:
            self.params = self._initialize_parameters()

        # Initialize optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = optax.adam(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optax.sgd(learning_rate=self.learning_rate)
        else:
            # Default to Adam if an unknown optimizer type is specified
            print(f"Warning: Unknown optimizer type '{optimizer_type}'. Defaulting to Adam.")
            self.optimizer = optax.adam(learning_rate=self.learning_rate)

        # Ensure params passed to optimizer.init is not the instance's self.params directly,
        # though optax.init shouldn't modify its input. This is for extreme caution.
        params_for_opt_init = jax.tree_util.tree_map(lambda x: x, self.params)
        self.optimizer_state = self.optimizer.init(params_for_opt_init)

    def _initialize_parameters(self):
        """Helper function to initialize model parameters."""
        self.key_master, key_w, key_b = jax.random.split(self.key_master, 3)

        params = {}
        # For k=1 (no mixtures)
        # w: couplings (L, A, L, A)
        # b: biases (L, A)
        # Initialize with zeros or small random values. Original laxy.MRF layer init with zeros.
        # Bias 'b' was initialized using data mean in original mrf.py;
        # here, we'll do simpler init first, data-dependent init can be added in fit or as an option.

        if self.k == 1:
            params['w'] = jnp.zeros((self.L, self.A, self.L, self.A))
            params['b'] = jnp.zeros((self.L, self.A))
        else: # k > 1, initialize mixture params as well
            params['w'] = jnp.zeros((self.k, self.L, self.A, self.L, self.A)) # Or shared w + mw_delta
            params['b'] = jnp.zeros((self.k, self.L, self.A)) # Or shared b + mb_delta
            params['c'] = jnp.zeros((self.k,)) # Logits for mixture components
            # TODO: Full mixture parameter sharing logic from original mrf.py (tied, full, shared)

        return params

    def _loss_fn(self, params: dict, msa_batch: jnp.ndarray, weights_batch: jnp.ndarray,
                 order: jnp.ndarray = None, key: jax.random.PRNGKey = None):
        """
        Computes the loss for the MRF model (pseudo-likelihood + regularization).

        Args:
            params (dict): Current model parameters ('w', 'b', and potentially 'c', 'mw', 'mb' for mixtures).
            msa_batch (jnp.ndarray): A batch of one-hot encoded sequences (batch_size, L, A).
            weights_batch (jnp.ndarray): Weights for each sequence in the batch (batch_size,).
            order (jnp.ndarray, optional): Autoregressive order. If provided, an AR mask is applied.
                                           Defaults to None.
            key (jax.random.PRNGKey, optional): PRNG key, reserved for future use (e.g., stochastic elements).

        Returns:
            float: The computed loss value for the batch.
        """
        L, A = self.L, self.A

        # 1. Regularization Loss
        reg_loss_val = 0.0

        # Common function for L2 reg to handle different param structures (k=1 vs k>1)
        def _l2_reg(tensor, scale_factor=1.0):
            return self.lam * scale_factor * jnp.sum(jnp.square(tensor))

        if self.k == 1:
            # Regularize w: 0.5 * (L-1)*(A-1) is a common scaling factor for Potts models
            # However, the original mrf.py used lam/2*(L-1)*(A-1) for w, and lam for b.
            # Let's use a simpler scaling first, can be refined.
            # The original scaling was: lam/2*(L-1)*(A-1) * jnp.square(params["w"]).sum())
            # and lam * jnp.square(params["b"]).sum()
            # For now, using simpler sum of squares, scaling can be part of self.lam.
            # The factor (L-1)*(A-1)/2 accounts for number of independent parameters in w if symmetric and zero-sum gauge.
            # Let's use the original scaling factor if possible, or make self.lam incorporate it.
            # For now, a simpler direct L2:
            if 'w' in params:
                 reg_loss_val += 0.5 * self.lam * jnp.sum(jnp.square(params['w'])) # Factor of 0.5 is common
            if 'b' in params:
                 reg_loss_val += self.lam * jnp.sum(jnp.square(params['b']))
        else: # Mixtures k > 1
            # Assuming params['w'] is (k,L,A,L,A), params['b'] is (k,L,A), params['c'] is (k,)
            if 'w' in params: # mixture-specific couplings 'mw' in original
                reg_loss_val += 0.5 * self.lam * jnp.sum(jnp.square(params['w']))
            if 'b' in params: # mixture-specific biases 'mb' in original
                reg_loss_val += self.lam * jnp.sum(jnp.square(params['b']))
            if 'c' in params: # mixture weights/logits
                reg_loss_val += self.lam * jnp.sum(jnp.square(params['c']))
            # TODO: Add regularization for shared 'w_shared', 'b_shared' if that scheme is implemented.


        # 2. Pseudo-likelihood Loss
        w = params['w']
        b = params['b']

        # Apply autoregressive mask if order is provided
        if order is not None:
            from ..utils.data_processing import ar_mask # Avoid circular import at top level if utils imports models
            # ar_m is (L, L). We need (L,1,L,1) for w, or (1,L,1,L,1) for mixture w.
            ar_m = ar_mask(order, diag=False) # Typically diag=False for AR prediction
            if self.k == 1:
                w_mask = ar_m[:, jnp.newaxis, :, jnp.newaxis] # (L,1,L,1)
            else: # k > 1
                w_mask = ar_m[jnp.newaxis, :, jnp.newaxis, :, jnp.newaxis] # (1,L,1,L,1)
            w = w * w_mask

        # Symmetrize couplings (often done for MRFs, original mrf.py did this in pll_loss)
        # w_ symmetrized = 0.5 * (w + w.transpose([axis indices for L,A,L,A or k,L,A,L,A]))
        if self.k == 1:
            # w is (L,A,L,A)
            w_symm = 0.5 * (w + w.transpose((2, 3, 0, 1)))
            # Remove diagonal interactions (i interacting with i)
            # Create a mask that is 0 for i=j and 1 otherwise for L indices
            diag_mask_L = (1.0 - jnp.eye(L))[:, jnp.newaxis, :, jnp.newaxis] # (L,1,L,1)
            w_final = w_symm * diag_mask_L

            # Compute logits: einsum('nla,lAqb->nqb', msa_batch, w_final) + b[None,:,:]
            # msa_batch (N,L,A), w_final (L,A,L,A), b (L,A)
            # Output logits should be (N,L,A)
            # Sum_q x_q W_qa,jb (original notation) -> Sum_l' x_l' W_l'a',la (current notation)
            # Field on site (l,a) from all other sites (l',a')
            # logits_ij = sum_{l'a'} W_{la,l'a'} * x_{l'a'}
            # This is effectively: msa_features @ W_flattened + b
            # Original pll_loss: jnp.einsum("nia,iajb->njb", inputs["x"], w)
            # x: (N,L,A) ; w: (L,A,L,A) -> output: (N,L,A) (mistake in original comment, should be (N,L,A))
            # Correct einsum for (N,L,A) output: 'nqa,qalp->nlp' (N=batch, L=length, A=alphabet)
            # where q is summed over (L), a is summed over (A)
            logits = jnp.einsum('nqa,qalp->nlp', msa_batch, w_final) + b[None, :, :]

        else: # Mixtures k > 1
            # params['w'] is (k,L,A,L,A), params['b'] is (k,L,A), params['c'] is (k,)
            # Symmetrize and mask diagonal for each mixture component's w
            w_k_symm = 0.5 * (w + w.transpose((0, 3, 4, 1, 2))) # (k,L,A,L,A)
            diag_mask_L_k = (1.0 - jnp.eye(L))[jnp.newaxis, :, jnp.newaxis, :, jnp.newaxis] # (1,L,1,L,1)
            w_k_final = w_k_symm * diag_mask_L_k

            # Logits for each mixture component: (N, k, L, A)
            # msa_batch (N,L,A), w_k_final (k,L,A,L,A) -> einsum('nqa,kpalq->nkpl', msa_batch, w_k_final) (careful with indices)
            # Target: (N,k,L,A)
            # einsum('nqa,kqlpa->nkpa', msa_batch, w_k_final) (swapped last two axes of w_k_final for target A)
            # No, it should be: 'nqa,kpalq->nkpl' if w is k,L',A',L,A
            # 'nqa,kpals->nkps' if w is k,pos1,alpha1,pos2,alpha2 (s=alpha2)
            # Let's use original notation: w_k_final (k,L,A,L,A) = (k,i,a,j,b)
            # msa_batch (N,L,A) = (N,j,b)
            # Logits per component: logits_k = einsum('njb,kiajb->nkia', msa_batch, w_k_final) + params['b'][None,:,:,:]
            logits_per_k = jnp.einsum('nqb,kpalq->nkpa', msa_batch, w_k_final) + params['b'][None, :, :, :] # (N, k, L, A)

            # Mixture component probabilities (log_softmax over components for stability if needed)
            mixture_log_probs = jax.nn.log_softmax(params['c']) # (k,)

            # Log likelihood for each sequence, for each component: (N, k)
            # log_probs_per_k = jax.nn.log_softmax(logits_per_k, axis=-1) # (N,k,L,A)
            # pll_per_k_site = (msa_batch[:,None,:,:] * log_probs_per_k).sum(axis=(-1,-2)) # (N,k) sum over L,A

            # Instead, logsumexp over mixture components:
            # log P(x_i | params) = log sum_k P(z_k|c) P(x_i | w_k, b_k)
            # log P(x_i | w_k,b_k) = sum_l log P(x_il | x_i\l, w_k, b_k)
            # log P(x_il | ...) = x_il * log_softmax(logits_for_x_il)

            # Logits for each site (l) and alphabet (a), for each sequence (n) and component (k):
            # logits_per_k is (N, k, L, A)
            site_log_likelihood_per_k = (msa_batch[:,None,:,:] * jax.nn.log_softmax(logits_per_k, axis=-1)).sum(axis=-1) # (N,k,L)
            sequence_log_likelihood_per_k = site_log_likelihood_per_k.sum(axis=-1) # (N,k)

            # Combine with mixture probabilities
            # log P(x_n) = logsumexp_k (log P(z_n=k) + log P(x_n | z_n=k))
            # log P(z_n=k) is mixture_log_probs[k]
            # log P(x_n | z_n=k) is sequence_log_likelihood_per_k[n,k]

            log_likelihood_per_seq = jax.scipy.special.logsumexp(
                mixture_log_probs[None,:] + sequence_log_likelihood_per_k, axis=1
            ) # (N,)

            # PLL loss is negative sum of these log likelihoods, weighted
            pll_val = -jnp.sum(weights_batch * log_likelihood_per_seq)
            logits = None # Not directly used for loss if k > 1 with this formulation

        if self.k == 1 or logits is not None: # Ensure logits is defined for k=1 case
            # Categorical cross-entropy (pseudo-likelihood)
            log_probabilities = jax.nn.log_softmax(logits, axis=-1) # (N, L, A)
            # Element-wise product with true one-hot MSA, sum over A and L
            # msa_batch (N,L,A)
            # weights_batch (N,) -> need to reshape for broadcasting with (N,L) or (N)

            # Loss per sequence position: -(msa_batch * log_probabilities).sum(axis=-1) # (N,L)
            # Loss per sequence: sum over L -> (N,)
            # Weighted sum over N
            # Original code: cce_loss = -(inputs["x"] * jax.nn.log_softmax(sum(logits))).sum([1,2])
            # This sums over L and A for each N, resulting in (N,).
            # Then: (cce_loss*inputs["x_weight"]).sum()

            # Check: sum([1,2]) for (N,L,A) means sum over L and A.
            # So, cce_loss is (N,). Then (cce_loss * weights_batch).sum()

            # Ensure msa_batch is one-hot. If it's probabilities, this is fine.
            # If it's integer labels, it needs to be one-hot encoded first for this formula.
            # Assuming msa_batch is one-hot (N,L,A).

            # Negative log pseudo-likelihood for each sequence
            neg_log_pll_per_sequence = -jnp.sum(msa_batch * log_probabilities, axis=(1, 2)) # Shape (N,)

            # Weighted sum of negative log pseudo-likelihoods
            pll_val = jnp.sum(weights_batch * neg_log_pll_per_sequence)

        total_loss = pll_val + reg_loss_val
        return total_loss

    def fit(self, X: jnp.ndarray, X_weight: jnp.ndarray = None,
            steps: int = 100, batch_size: int = None,
            learning_rate: float = None, key: jax.random.PRNGKey = None,
            ar: bool = False, ar_order_type: str = 'entropy', # 'entropy', 'sequential', or pass custom order
            custom_ar_order: jnp.ndarray = None,
            verbose: bool = True, print_every: int = 10):
        """
        Fit/train the MRF model.

        Args:
            X (jnp.ndarray): One-hot encoded MSA of shape (N, L, A).
            X_weight (jnp.ndarray, optional): Sequence weights of shape (N,).
                                             Defaults to uniform weights if None.
            steps (int): Number of training steps. Defaults to 100.
            batch_size (int, optional): Batch size for training. If None, uses full batch.
                                        Defaults to None.
            learning_rate (float, optional): Learning rate. If None, uses rate from __init__.
                                             Defaults to None.
            key (jax.random.PRNGKey, optional): JAX PRNG key for shuffling. If None, a new key is generated
                                                from the model's master key.
            ar (bool): Whether to use autoregressive mode. Defaults to False.
            ar_order_type (str): Method to determine autoregressive order if ar=True.
                                 Options: 'entropy', 'sequential'. Ignored if custom_ar_order is provided.
                                 Defaults to 'entropy'.
            custom_ar_order (jnp.ndarray, optional): A specific autoregressive order to use. (L,).
                                                   If provided, `ar` is implicitly True and `ar_order_type` is ignored.
            verbose (bool): If True, print training progress. Defaults to True.
            print_every (int): Print loss every `print_every` steps if verbose. Defaults to 10.


        Returns:
            MRF: The fitted model instance (self).
        """
        N, L_data, A_data = X.shape
        assert L_data == self.L, f"Input MSA length {L_data} does not match model L {self.L}"
        assert A_data == self.A, f"Input MSA alphabet size {A_data} does not match model A {self.A}"

        if X_weight is None:
            X_weight = jnp.ones(N) / N # Uniform weighting summing to 1
        else:
            X_weight = jnp.asarray(X_weight)
            # Normalize weights to sum to 1. This implies loss is an average, not sum over Neff.
            # Could also pass raw weights and scale loss by 1/N_eff.
            # For now, this matches a common convention if loss is mean loss.
            X_weight = X_weight / (jnp.sum(X_weight) + 1e-9)


        if key is None:
            self.key_master, key = jax.random.split(self.key_master)

        # Handle learning rate update for this specific fit call
        current_lr = self.learning_rate
        if learning_rate is not None and learning_rate != self.learning_rate:
            current_lr = learning_rate
            # Create a new optimizer instance if LR changes for this fit call
            # This also means optimizer state will be reset.
            if self.optimizer_type == 'adam':
                current_optimizer = optax.adam(learning_rate=current_lr)
            elif self.optimizer_type == 'sgd':
                current_optimizer = optax.sgd(learning_rate=current_lr)
            else:
                current_optimizer = optax.adam(learning_rate=current_lr)
            current_optimizer_state = current_optimizer.init(self.params)
        else:
            # Use the optimizer and state from __init__ or previous fit
            current_optimizer = self.optimizer
            current_optimizer_state = self.optimizer_state


        # Determine autoregressive order
        ar_order_to_use = None
        if custom_ar_order is not None:
            ar_order_to_use = jnp.asarray(custom_ar_order)
        elif ar:
            if ar_order_type == 'sequential':
                ar_order_to_use = jnp.arange(self.L)
            elif ar_order_type == 'entropy':
                from ..utils.stats import get_stats # Local import
                # Calculate p_i = P(alphabet | position), sum_alphabet p_i = 1 for each position
                # X is (N,L,A), X_weight is (N,)
                # weighted_X_sum_L_A = (X * X_weight[:, None, None]).sum(axis=0) # (L,A) sum of weights for each (L,A)
                # N_eff_L = weighted_X_sum_L_A.sum(axis=1, keepdims=True) # (L,1) sum of weights for each L
                # p_i = weighted_X_sum_L_A / (N_eff_L + 1e-9)

                # Simpler: Calculate mean character frequencies per position, weighted
                N_eff = jnp.sum(X_weight) + 1e-9 # Effective number of sequences
                p_i_num = jnp.sum(X * X_weight[:, jnp.newaxis, jnp.newaxis], axis=0) # Numerator for p_i, shape (L, A)
                p_i = p_i_num / N_eff # Frequencies p(a_l)

                site_entropy = -jnp.sum(p_i * jnp.log(p_i + 1e-9), axis=-1) # Shape (L,)
                ar_order_to_use = jnp.argsort(site_entropy) # Low to high entropy
            else:
                raise ValueError(f"Unknown ar_order_type: {ar_order_type}")

        # JIT the update step function
        @jax.jit
        def train_step_fn(params, opt_state, msa_b, weights_b, order_b, key_b):
            loss_val, grads = jax.value_and_grad(self._loss_fn)(
                params, msa_b, weights_b, order_b, key_b
            )
            updates, new_opt_state = current_optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        indices = jnp.arange(N)
        current_batch_size = batch_size if batch_size is not None and batch_size < N else N

        loss_history = []

        for step_num in range(steps):
            step_key, perm_key = jax.random.split(key)
            key = step_key

            if current_batch_size < N:
                permuted_indices = jax.random.permutation(perm_key, indices)
                batch_indices = permuted_indices[:current_batch_size]
            else:
                batch_indices = indices

            msa_batch_data = X[batch_indices, ...]
            weights_batch_data = X_weight[batch_indices]

            # Normalize batch weights to sum to 1 for this batch
            # This makes loss per batch comparable regardless of minor variations in sum(weights_batch_data)
            # if X_weight was not initially normalized to sum to 1.
            current_batch_sum_weights = jnp.sum(weights_batch_data) + 1e-9
            scaled_weights_batch_data = weights_batch_data / current_batch_sum_weights


            current_params, current_optimizer_state, loss = train_step_fn(
                self.params, current_optimizer_state, # Use current_optimizer_state
                msa_batch_data, scaled_weights_batch_data, # Pass scaled weights
                ar_order_to_use, perm_key
            )
            self.params = current_params # Update model parameters
            loss_history.append(loss.item())

            if verbose and (step_num + 1) % print_every == 0:
                avg_loss_since_last_print = np.mean(loss_history[-(print_every):]) # Use numpy for simple mean of list
                print(f"Step {step_num + 1}/{steps}, Avg Loss (last {print_every}): {avg_loss_since_last_print:.4f}, Current Batch Loss: {loss:.4f}")

        # Persist the potentially updated optimizer state back to the instance
        self.optimizer_state = current_optimizer_state

        if verbose:
             final_loss = loss_history[-1] if loss_history else float('nan')
             print(f"Fit completed. Final Loss: {final_loss:.4f}")
        return self

    def get_w(self) -> jnp.ndarray:
        """
        Computes the effective coupling matrix (W) from learned parameters.
        Handles symmetrization, normalization, and combines mixture parameters if k > 1.

        Returns:
            jnp.ndarray: The effective coupling matrix, typically of shape (L, A, L, A)
                         or (L, A_no_gap, L, A_no_gap) after processing.
                         For k > 1, it's an average or combined coupling.
        """
        if self.params is None:
            raise ValueError("Model not yet fit. Parameters are not available.")

        current_w = None
        if self.k == 1:
            if 'w' not in self.params:
                raise ValueError("Parameters 'w' not found in model for k=1.")
            # The test assertion model.params['w'] is not None should catch this if params were loaded.
            # If model is freshly initialized, params['w'] is jnp.zeros, so not None.
            current_w = self.params['w']
        else: # k > 1 (mixtures)
            if 'w' not in self.params or 'c' not in self.params:
                raise ValueError("Parameters 'w' or 'c' for mixtures not found or are None.")
            if self.params['w'] is None or self.params['c'] is None:
                 raise ValueError("Mixture parameters 'w' or 'c' are None.")

            mixture_weights = jax.nn.softmax(self.params['c'])
            # self.params['w'] for k>1 is (k,L,A,L,A)
            current_w = jnp.einsum('k,kijab->ijab', mixture_weights, self.params['w'])

        if current_w is None:
             # This path should ideally not be reached if above checks are comprehensive.
             raise ValueError("Effective 'w' evaluated to None unexpectedly.")

        # Ensure current_w is the expected (L,A,L,A) shape after potential mixture processing.
        expected_shape = (self.L, self.A, self.L, self.A)
        if current_w.shape != expected_shape:
            raise ValueError(f"Calculated 'current_w' has unexpected shape: {current_w.shape}. Expected {expected_shape}.")

        w_symmetrized = 0.5 * (current_w + current_w.transpose((2, 3, 0, 1)))
        mean_per_pos_pair = w_symmetrized.mean(axis=(1,3), keepdims=True)
        w_normalized = w_symmetrized - mean_per_pos_pair

        return w_normalized

    def predict(self, X=None, **kwargs): # X might not be needed for contact prediction from couplings
        """
        Predicts contacts from the learned MRF model.
        This involves computing the effective coupling matrix W and then applying APC.

        Args:
            X: This argument is currently ignored for MRF contact prediction but kept for API consistency.
            alphabet_size_no_gap (int, optional): Size of alphabet excluding gap (e.g. 20 for AA).
                                                  Defaults to self.A - 1 if last char is gap-like.

        Returns:
            tuple: (raw_contact_map, apc_contact_map)
                   raw_contact_map (L,L): Frobenius norm of couplings per position pair.
                   apc_contact_map (L,L): Average Product Corrected contact map.
        """
        if self.params is None:
            raise ValueError("Model not yet fit.")

        w_couplings = self.get_w() # Get (L,A,L,A) effective couplings

        alphabet_size_no_gap = kwargs.get('alphabet_size_no_gap', self.A -1 if self.A > 0 else 0)
        if alphabet_size_no_gap <=0:
            # Default to using all A if A-1 is not sensible
            alphabet_size_no_gap = self.A


        from ..utils.stats import get_mtx # Local import
        raw_map, apc_map = get_mtx(w_couplings, alphabet_size_no_gap=alphabet_size_no_gap)

        return raw_map, apc_map

    def sample(self, num_samples: int, key: jax.random.PRNGKey,
               burn_in: int = 10, temperature: float = 1.0,
               order: jnp.ndarray = None, # For autoregressive sampling
               return_one_hot: bool = True):
        """
        Generate new sequences from the trained MRF model.

        Args:
            num_samples (int): The number of sequences to generate.
            key (jax.random.PRNGKey): JAX PRNG key for sampling.
            burn_in (int): Number of burn-in steps for Gibbs/sequential sampling.
                           If burn_in=1 and order is provided, performs autoregressive sampling.
                           If burn_in=0 or 1 (and no order/couplings), samples independently from biases.
                           Defaults to 10.
            temperature (float): Temperature for sampling. Defaults to 1.0.
            order (jnp.ndarray, optional): Order of positions for sequential/autoregressive sampling.
                                           If None and sequential sampling is implied by burn_in > 1 & couplings,
                                           a random permutation is used per burn-in step. Defaults to None.
            return_one_hot (bool): If True, returns one-hot encoded sequences. Otherwise, returns integer indices.
                                   Defaults to True.

        Returns:
            jax.numpy.ndarray: Generated sequences, shape (num_samples, L, A) if one-hot,
                               or (num_samples, L) if integer indices.
        """
        if self.params is None:
            raise ValueError("Model not yet fit. Parameters are not available.")

        L, A = self.L, self.A

        # Determine if sequential sampling (Gibbs/AR) is feasible/intended
        has_couplings = ('w' in self.params and self.params['w'] is not None)
        use_sequential_logic = has_couplings and (burn_in > 1 or (burn_in == 1 and order is not None))

        # Prepare parameters for sampling function
        # The sampling function might need to handle k=1 vs k>1 structures differently
        current_params_for_sampling = self.params.copy() # Avoid modifying internal params
        current_params_for_sampling['L'] = L
        current_params_for_sampling['A'] = A
        current_params_for_sampling['k'] = self.k # Pass k for mixture handling

        # Initial MSA: zeros, to be filled by sampling
        msa_shape = (num_samples, L, A)
        msa = jnp.zeros(msa_shape) # Will be one-hot after sampling a position

        # Handle mixture component selection for each sample if k > 1
        if self.k > 1:
            if 'c' not in current_params_for_sampling:
                raise ValueError("Mixture weights 'c' not found for sampling with k > 1.")

            key, labels_key = jax.random.split(key)
            mixture_logits = current_params_for_sampling['c'] # (k,)
            # Sample a component label for each of the num_samples
            sampled_mixture_indices = jax.random.categorical(labels_key, mixture_logits, shape=(num_samples,)) # (num_samples,)
            current_params_for_sampling['sampled_mixture_indices'] = sampled_mixture_indices
            # The internal _sample_step_pll needs to use these indices to select correct w_k, b_k

        # Define a single step of pseudo-likelihood based sampling (one position or all)
        def _sample_step_pll(msa_state, current_key_input, iter_pos_idx=None):
            """ Samples one position (if iter_pos_idx given) or all positions. """
            # iter_pos_idx: if sequential, this is the current position index to sample.
            #               if None, sample all positions independently (only from biases).

            current_logits = jnp.zeros((num_samples, A)) # For a single position if iter_pos_idx
            if iter_pos_idx is None: # For full independent sampling
                 current_logits_full = jnp.zeros((num_samples, L, A))


            if self.k == 1:
                b_eff = current_params_for_sampling['b'] # (L,A)
                w_eff = current_params_for_sampling.get('w') # (L,A,L,A)
            else: # k > 1, select parameters for the sampled mixture component for each sequence
                # sampled_mixture_indices (num_samples,)
                # params['b'] is (k,L,A), params['w'] is (k,L,A,L,A)
                b_k_selected = current_params_for_sampling['b'][current_params_for_sampling['sampled_mixture_indices']] # (num_samples, L, A)
                w_k_selected = current_params_for_sampling.get('w', jnp.zeros((self.k,L,A,L,A)))[current_params_for_sampling['sampled_mixture_indices']] # (num_samples,L,A,L,A)
                b_eff = b_k_selected
                w_eff = w_k_selected


            if iter_pos_idx is not None: # Sequential sampling for a single position `iter_pos_idx`
                site_bias = b_eff[..., iter_pos_idx, :] # (num_samples, A) if k>1, else (A,) -> broadcast
                if b_eff.ndim == 2 and self.k==1: # (L,A) for k=1
                    site_bias = b_eff[iter_pos_idx, :] # (A,)

                current_logits = site_bias # Start with bias

                if w_eff is not None and has_couplings: # Add coupling contributions
                    # msa_state (num_samples, L, A), w_eff (L,A,L,A) or (num_samples,L,A,L,A)
                    # We need field at pos `iter_pos_idx` from other positions in `msa_state`.
                    # Field_ia = sum_{jb} W_{ia,jb} * x_jb
                    # W is (L,A,L,A) or (N,L,A,L,A) for mixtures
                    # x is (N,L,A)
                    # Target: (N,A) for position iter_pos_idx
                    if self.k == 1: # w_eff is (L,A,L,A)
                        # einsum: 'nla,lapq->nq', where l=iter_pos_idx, p=iter_pos_idx
                        # This is field AT pos q, alpha p, from all OTHER pos l, alpha a
                        # Field at pos `i` (iter_pos_idx): sum_{j # i, b} W_{ia,jb} * x_jb
                        # W_slice for pos `i`: w_eff[iter_pos_idx, :, :, :] (A, L, A)
                        # Effective einsum: 'nla, alp -> np' (n=num_samples, l=L, a=A, p=A at pos iter_pos_idx)
                        # Field = jnp.einsum('nla,axla->nx', msa_state, w_eff[iter_pos_idx]) # if w is LA,LA
                        # Field_on_pos_i = einsum('njb,jbIA->nIA', msa_state, w_eff[:,:,iter_pos_idx,:])
                        # Field_ia = sum_jb W_ia,jb x_jb
                        # W_ia,jb is w_eff[i,a,j,b]
                        # field_at_i = jnp.einsum('njb,ajb->na', msa_state, w_eff[iter_pos_idx]) # This was from original sample_pll
                        # This seems to be field at pos `i` from all positions `j` in msa_state, using W_i.
                        # This is: sum_j sum_b msa_state_jb * W_ia,jb
                        # This is correct: field_at_pos_i_alpha_a = sum_{j,beta} W_{i,alpha,j,beta} * msa_state_{j,beta}
                        coupling_field = jnp.einsum('njb,ajb->na', msa_state, w_eff[iter_pos_idx, :, :, :])
                    else: # w_eff is (num_samples, L,A,L,A) for mixtures
                        # Need to select w_eff[n, iter_pos_idx, :, :, :] for each sample n. (A,L,A)
                        # Then einsum('jb,ajb->a', msa_state[n], w_eff_n_i) for each n.
                        # This can be done by vmap or careful batch einsum.
                        # coupling_field = jax.vmap(lambda m, w_i: jnp.einsum('jb,ajb->a', m, w_i))(
                        #    msa_state, w_eff[:, iter_pos_idx, :, :, :]
                        # )
                        # Or direct einsum: 'njb,najb->na'
                        coupling_field = jnp.einsum('njb,najb->na', msa_state, w_eff[:, iter_pos_idx, :, :, :])
                    current_logits += coupling_field

                # Sample for position iter_pos_idx
                sampled_chars_one_hot = jax.nn.one_hot(
                    jax.random.categorical(current_key_input, current_logits / temperature), A
                )
                msa_state = msa_state.at[:, iter_pos_idx, :].set(sampled_chars_one_hot)

            else: # Independent sampling for all positions (only from biases)
                # b_eff is (L,A) for k=1, or (num_samples,L,A) for k>1 (after selection)
                if self.k == 1:
                    # b_eff is (L,A). We need (num_samples,L,A) for categorical sampling over num_samples.
                    # Tile/repeat b_eff num_samples times.
                    current_logits_full = jnp.repeat(b_eff[None, ...], num_samples, axis=0)
                else: # k > 1, b_eff is already (num_samples, L, A)
                    current_logits_full = b_eff

                sampled_chars_one_hot_all_pos = jax.nn.one_hot(
                    jax.random.categorical(current_key_input, current_logits_full / temperature), A # axis=-1 by default
                ) # output shape (num_samples, L, A)
                msa_state = sampled_chars_one_hot_all_pos

            return msa_state, None # Scan expects (carry, output_slice)

        # Main sampling loop (burn-in iterations)
        if use_sequential_logic:
            # Sequential sampling (Gibbs or Autoregressive)
            for _ in range(burn_in):
                key, iter_key = jax.random.split(key)

                current_order = order
                if order is None: # Gibbs: random order per burn-in step
                    current_order = jax.random.permutation(iter_key, jnp.arange(L))

                # Loop over positions in the specified or random order
                # `msa` is carry, `pos_idx_to_sample` is iterated element from current_order
                # Need keys for each position sampling step
                pos_keys = jax.random.split(iter_key, L)

                # Partial function for scan body, fixing X and other static args
                # scan_fn_partial = functools.partial(_sample_step_pll, X=X, params_for_sampling=current_params_for_sampling, temperature=temperature)
                # msa, _ = jax.lax.scan(scan_fn_partial, msa, (current_order, pos_keys))
                # This is not quite right. _sample_step_pll needs msa_state and pos_idx.

                # Correct scan setup for sequential sampling over positions
                def body_seq_sample(carry_msa, pos_and_key_slice):
                    pos_idx, pos_key = pos_and_key_slice
                    # Call _sample_step_pll which modifies one position of carry_msa
                    new_msa, _ = _sample_step_pll(carry_msa, pos_key, iter_pos_idx=pos_idx)
                    return new_msa, None # No per-position output needed from scan here

                msa, _ = jax.lax.scan(body_seq_sample, msa, (current_order, pos_keys))

        else: # Independent sampling from biases (or if no couplings)
            key, final_sample_key = jax.random.split(key)
            msa, _ = _sample_step_pll(msa, final_sample_key, iter_pos_idx=None)

        if return_one_hot:
            return msa
        else:
            return msa.argmax(-1) # Return integer indices

    def get_parameters(self) -> dict:
        """
        Get the learned parameters of the model.

        Returns:
            dict: A dictionary containing model parameters (e.g., 'w', 'b', 'c').
        """
        return self.params

    def load_parameters(self, params: dict):
        """
        Load pre-trained parameters into the model.
        This also re-initializes the optimizer state.

        Args:
            params (dict): A dictionary containing model parameters.
        """
        # Ensure a deep copy of params to avoid any potential modification by reference issues,
        # especially before it's passed to optimizer.init.
        self.params = jax.tree_util.tree_map(lambda x: x, params)

        # Re-initialize optimizer state with the new parameters
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer_state = self.optimizer.init(self.params)
        else:
            self.optimizer_state = None
            # Consider raising an error if optimizer should always exist after __init__
            print("Warning: Optimizer not available or not initialized in load_parameters. Optimizer state not reset.")


    def get_w(self):
        # To be implemented: logic from mrf.py:MRF.get_w()
        # This computes the effective coupling matrix.
        if self.params is None:
            raise ValueError("Model not yet fit. Parameters 'w' (and potentially 'mw') not available.")
        # Placeholder
        return None
