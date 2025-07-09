import jax
import jax.numpy as jnp
import numpy as np # Keep for np.asarray in kmeans_sample if needed, or ensure JAX arrays throughout
from math import log
from .base import Model
import optax # For potential future integration if we make training part of fit more complex

class KMeans(Model):
    """
    KMeans clustering model implemented in JAX.
    """

    def __init__(self, n_clusters: int, n_init: int = 10, max_iter: int = 300, tol: float = 1e-4, seed: int = 0):
        """
        Initialize the KMeans model.

        Args:
            n_clusters (int): The number of clusters to form.
            n_init (int): Number of time the k-means algorithm will be run with different centroid seeds.
                          The final results will be the best output of n_init consecutive runs in terms of inertia.
            max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
            tol (float): Relative tolerance with regards to Frobenius norm of the difference in the cluster
                         centers of two consecutive iterations to declare convergence.
            seed (int): Seed for random number generation.
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.means_ = None # Cluster centers
        self.labels_ = None # Labels of each point
        self.inertia_ = None # Sum of squared distances of samples to their closest cluster center.
        self.cat_ = None # Category proportions

        # JIT compile core methods for performance
        self._dist_jit = jax.jit(self._dist)
        self._kmeans_plus_plus_jit = jax.jit(self._kmeans_plus_plus, static_argnames=("n_clusters",))
        self._core_kmeans_jit = jax.jit(self._core_kmeans, static_argnames=("n_clusters", "n_init", "max_iter", "tol"))

    def _dist(self, a, b):
        sm = a @ b.T
        a_norm = jnp.square(a).sum(-1)
        b_norm = jnp.square(b).sum(-1)
        return jnp.abs(a_norm[:, None] + b_norm[None, :] - 2 * sm)

    def _kmeans_plus_plus(self, key, X, X_weight, n_clusters):
        n_samples, n_features = X.shape
        n_candidates = 2 + int(log(n_clusters))

        def loop_plus_plus(current_m_centers, scan_slice_c):
            # current_m_centers is the array of means being built up by scan.
            # scan_slice_c is a tuple (n_center_to_find_idx, current_prng_key)
            n_center_to_find_idx, current_prng_key = scan_slice_c

            # Distances from all points to the currently selected centers
            # We use current_m_centers directly. JAX handles the slicing for [:n_center_to_find_idx]
            # inside the loop if needed, or we rely on the fact that centers not yet chosen are zero
            # and _dist handles it gracefully or they don't dominate min().
            # The logic from original _kmeans code:
            #   inf_mask = jnp.inf * (jnp.arange(n_clusters) > n)
            #   p = (inf_mask + _dist(X,m)).min(-1)
            # This implies using all `m` (current_m_centers) but masking distances to not-yet-found centers.
            # For scan, n_center_to_find_idx is the index of the *current* center we are determining.
            # So, we need distances to the *previous* n_center_to_find_idx centers.

            # Since indices_to_find = jnp.arange(1, n_clusters),
            # n_center_to_find_idx will always be >= 1 within this scan loop.
            # The initial center (index 0) is set before this scan.
            # Thus, the n_center_to_find_idx == 0 case is not reachable here.

            # Correctly slice current_m_centers up to the number of centers already found
            # (i.e., up to n_center_to_find_idx -1, as n_center_to_find_idx is the *index* of the current one)
            # Use jax.lax.dynamic_slice for slicing with a tracer
            num_features = current_m_centers.shape[1]
            # Ensure n_center_to_find_idx is at least 1 for slicing, or handle empty slice if it can be 0.
            # Since scan is on arange(1, n_clusters), n_center_to_find_idx is always >= 1.
            # So, current_m_centers[:n_center_to_find_idx] should be safe.
            # The error occurs because n_center_to_find_idx is a tracer.

            # If n_center_to_find_idx is dynamic, we must use dynamic_slice.
            # start_indices for the slice current_m_centers[:n_center_to_find_idx] are [0,0]
            # slice_sizes are [n_center_to_find_idx, num_features]

            # However, the direct slicing X[indices] is often allowed if indices is a tracer array of integers.
            # The issue is slice stop being a tracer.
            # Let's try to make the slice stop static if possible, or use a loop if not.
            # The most robust way with scan and dynamic slice extents is often to pass necessary data
            # or use jax.lax.cond for cases that would result in empty slices if not handled.

            # Given n_center_to_find_idx comes from jnp.arange(1, n_clusters), it's a JAX array (tracer).
            # Slicing an array `arr[:tracer]` is problematic.
            # We can use a mask or jax.lax.select.

            # Alternative: if n_clusters is static, we can unroll this part or use jax.lax.fori_loop.
            # For now, let's try to make the slice explicit for JIT

            # Create a boolean mask for rows to select based on n_center_to_find_idx
            row_indices = jnp.arange(current_m_centers.shape[0])
            mask = row_indices < n_center_to_find_idx

            # This is still tricky. The most straightforward way if the slice is truly dynamic
            # is often to use lax.cond to handle the case where n_center_to_find_idx might be 0,
            # and then use dynamic_slice.
            # However, since n_center_to_find_idx >= 1, current_m_centers[:n_center_to_find_idx]
            # should be valid if JAX could trace the slice.
            # The issue is `Traced<int32[]>` as slice index.

            # Let's use dynamic_slice:
            # start_indices should be concrete or tracers of same type.
            # slice_sizes should be a concrete tuple/list of ints. This is the problem for dynamic_slice.
            # `slice_stop` being a tracer is the core issue for standard Python slicing `[:value]`.

            # Let's rewrite using a mask:
            # We need to compute distances to centers 0 to n_center_to_find_idx - 1
            # This means we need to select these centers.

            # Let's assume n_clusters is small enough that vmap over an explicit loop is okay for this part.
            # Or, if n_center_to_find_idx is guaranteed > 0:

            # The error "Array slice indices must have static start/stop/step"
            # refers to Python slice objects `slice(start, stop, step)` where start/stop/step are tracers.
            # `current_m_centers[:n_center_to_find_idx]` is `current_m_centers[slice(None, n_center_to_find_idx, None)]`
            # This is the problem.

            # A common workaround is to compute for all and mask, if feasible.
            # Distances to all potential previous centers (up to n_clusters-1)
            all_dist_to_centers = self._dist_jit(X, current_m_centers) # (n_samples, n_clusters)

            # Create a mask for valid previous centers
            # Indices of centers are 0 to n_clusters-1
            # Previous centers are 0 to n_center_to_find_idx - 1
            valid_prev_centers_mask = jnp.arange(n_clusters) < n_center_to_find_idx # (n_clusters,)

            # Apply mask: distances to invalid centers become jnp.inf
            # Use jnp.where: where mask is False, use inf, else use actual distance.
            masked_distances = jnp.where(valid_prev_centers_mask[None, :], all_dist_to_centers, jnp.inf)
            p_for_choice = masked_distances.min(axis=1)


            # Sample candidate points
            sampled_candidate_indices = jax.random.choice(
                current_prng_key, jnp.arange(n_samples),
                shape=(n_candidates,),
                p=p_for_choice / (p_for_choice.sum() + 1e-8), # Add epsilon for stability
                replace=False
            )
            candidate_points = X[sampled_candidate_indices]

            # Pick the candidate that results in the largest reduction of inertia (or similar heuristic)
            # The original code calculates:
            #   dist = jnp.minimum(p[:,None],_dist(X,X[candidates]))
            #   i = candidates[(X_weight[:,None] * dist).sum(0).argmin()]
            # Here, p is `p_for_choice`.

            # Distances if each candidate point were added
            # potential_min_dists_if_candidate = jnp.minimum(p_for_choice[:, None], self._dist_jit(X, candidate_points))
            # Simpler: a point is a good candidate if it's far from existing centers (p_for_choice is distance)
            # AND other points are close to it.
            # The original provided code's `loop` for kmeans++ was:
            #   inf_mask = jnp.inf * (jnp.arange(n_clusters) > n) (n is n_center_to_find_idx)
            #   p = (inf_mask + _dist(X,m)).min(-1)  (m is current_m_centers)
            #   candidates = jax.random.choice(k, jnp.arange(n_samples), shape=(n_candidates,), p=p/p.sum(), replace=False)
            #   dist = jnp.minimum(p[:,None],_dist(X,X[candidates]))
            #   i = candidates[(X_weight[:,None] * dist).sum(0).argmin()]
            #   return m.at[n].set(X[i]), None
            # This means `p` is calculated based on *all* `m` centers, with future ones masked by inf.
            # This is difficult to do when `n_center_to_find_idx` itself is the loop variable of scan.

            # Let's use the logic from the original `_kmeans_plus_plus` inner scan loop:
            # `m` is `current_m_centers`. `n` is `n_center_to_find_idx`. `k` is `current_prng_key`.

            # p_dist_to_all_current_centers = (jnp.inf * (jnp.arange(n_clusters)[:,None] > n_center_to_find_idx-1) + \
            #                                 self._dist_jit(X, current_m_centers).T).min(0)

            # The logic for `p` in the original _kmeans (not the class method) seems to be:
            # For finding the n-th center, calculate distances to the n-1 already found centers.
            # `inf_mask` was applied to `m` (means), which has shape (n_clusters, n_features).
            # `inf_mask = jnp.inf * (jnp.arange(n_clusters) > n_center_to_find_idx)`
            # This means for the n-th center, distances to centers m_0...m_n are considered (masked for >n).
            # This is still a bit confusing. Let's stick to the simpler "p_for_choice" based on PREVIOUSLY selected centers.

            # Re-evaluating the selection of the best candidate from `candidate_points`:
            # We want to pick the candidate that minimizes the sum of squared distances to the *new* set of centers.
            # For each candidate_point in candidate_points:
            #   temp_centers = current_m_centers.at[n_center_to_find_idx].set(candidate_point)
            #   inertia = sum(min_dist(X, temp_centers)^2 * X_weight) -> too complex for scan body
            # The original code used:
            # dist_to_candidates = self._dist_jit(X, candidate_points) # (n_samples, n_candidates)
            # new_min_distances = jnp.minimum(p_for_choice[:, None], dist_to_candidates) # (n_samples, n_candidates)
            # weighted_sum_new_min_distances = (X_weight[:, None] * new_min_distances).sum(0) # (n_candidates,)
            # best_candidate_local_idx = weighted_sum_new_min_distances.argmin()
            # chosen_point_global_idx = sampled_candidate_indices[best_candidate_local_idx]

            # This seems correct and plausible for a scan body.
            dist_to_candidates = self._dist_jit(X, candidate_points)
            potential_min_dists_if_candidate_chosen = jnp.minimum(p_for_choice[:, None], dist_to_candidates)
            inertia_scores_for_candidates = (X_weight[:, None] * potential_min_dists_if_candidate_chosen).sum(axis=0)

            best_local_idx = inertia_scores_for_candidates.argmin()
            chosen_point_idx = sampled_candidate_indices[best_local_idx]

            updated_m_centers = current_m_centers.at[n_center_to_find_idx].set(X[chosen_point_idx])
            return updated_m_centers, None

        # Initial center choice (center 0)
        key, subkey = jax.random.split(key)
        first_center_idx = jax.random.choice(subkey, jnp.arange(n_samples))
        init_means_scanned = jnp.zeros((n_clusters, n_features)).at[0].set(X[first_center_idx])

        # Prepare carry for scan (indices of centers to find, and keys for each step)
        indices_to_find = jnp.arange(1, n_clusters)
        keys_for_scan = jax.random.split(key, n_clusters - 1)

        final_means = jax.lax.scan(loop_plus_plus, init_means_scanned, (indices_to_find, keys_for_scan))[0]
        return final_means

    def _core_kmeans(self, X, X_weight, n_clusters, n_init, max_iter, tol, seed):
        key = jax.random.PRNGKey(seed)

        def _E_step(means_e):
            return self._dist_jit(X, means_e).argmin(-1) # Return integer labels

        def _M_step(labels_m):
            # Convert integer labels to one-hot for weighted sum
            one_hot_labels = jax.nn.one_hot(labels_m, n_clusters)
            weighted_labels = one_hot_labels * X_weight[:, None]
            new_means = (weighted_labels.T @ X) / (weighted_labels.sum(0)[:, None] + 1e-8)
            return new_means

        def _inertia_calc(means_i, labels_i):
            # Given means and integer labels, calculate inertia
            # This is slightly different from original _inertia that took only means
            chosen_means = means_i[labels_i] # Get the mean for each point
            inertia = jnp.sum(X_weight * jnp.sum(jnp.square(X - chosen_means), axis=-1))
            return inertia
            # Original inertia calc:
            # sco = self._dist_jit(X, means_i).min(-1)
            # return (X_weight * sco).sum()


        def single_run(subkey_run):
            init_means = self._kmeans_plus_plus_jit(subkey_run, X, X_weight, n_clusters)

            if tol == 0:
                def em_scan_body(means_scan, _):
                    labels_scan = _E_step(means_scan)
                    new_means_scan = _M_step(labels_scan)
                    return new_means_scan, None # No loss needed per step for scan if tol=0

                final_means_scan = jax.lax.scan(em_scan_body, init_means, None, length=max_iter)[0]
                final_labels_scan = _E_step(final_means_scan)
                current_inertia = _inertia_calc(final_means_scan, final_labels_scan) # Or use original _inertia
                # current_inertia = (X_weight * self._dist_jit(X, final_means_scan).min(-1)).sum()


            else:
                def em_while_cond(state_while):
                    _, new_sco_while, old_sco_while, n_iter_while, _ = state_while
                    return ((jnp.abs(old_sco_while - new_sco_while) / (old_sco_while + 1e-8)) > tol) & (n_iter_while < max_iter)

                def em_while_body(state_while):
                    old_means_while, _, old_sco_while, n_iter_while, key_while = state_while

                    current_labels_while = _E_step(old_means_while)
                    new_means_while = _M_step(current_labels_while)
                    # new_sco_while = _inertia_calc(new_means_while, _E_step(new_means_while)) # Or use original _inertia
                    new_sco_while = (X_weight * self._dist_jit(X, new_means_while).min(-1)).sum()

                    return new_means_while, new_sco_while, old_sco_while, n_iter_while + 1, key_while

                # Initial state for while_loop
                initial_labels_em = _E_step(init_means)
                # initial_inertia_em = _inertia_calc(init_means, initial_labels_em) # Or use original _inertia
                initial_inertia_em = (X_weight * self._dist_jit(X, init_means).min(-1)).sum()


                # (means, new_inertia, old_inertia, iter_count, key)
                final_means_scan, current_inertia, _, _, _ = jax.lax.while_loop(
                    em_while_cond,
                    em_while_body,
                    (init_means, initial_inertia_em, initial_inertia_em + 2*tol, 0, subkey_run) # Ensure condition is initially true if tol > 0
                )
                final_labels_scan = _E_step(final_means_scan)

            return {"means": final_means_scan,
                    "labels": final_labels_scan, # Store integer labels
                    "inertia": current_inertia}

        if n_init > 0:
            # Vmap over different seeds for multiple initializations
            all_runs_results = jax.vmap(single_run)(jax.random.split(key, n_init))
            best_run_idx = all_runs_results["inertia"].argmin()
            best_results = jax.tree_util.tree_map(lambda x: x[best_run_idx], all_runs_results)
        else:
            best_results = single_run(jax.random.split(key,1)[0]) # Pass a single key

        # Category proportions from integer labels
        final_one_hot_labels = jax.nn.one_hot(best_results["labels"], n_clusters)
        cat_proportions = (final_one_hot_labels * X_weight[:, None]).sum(0) / (X_weight.sum() + 1e-8)

        return {**best_results, "cat": cat_proportions}

    def fit(self, X, X_weight=None, **kwargs):
        """
        Fit KMeans model.

        Args:
            X (jax.numpy.ndarray): Input data of shape (n_samples, n_features) or (n_samples, length, alphabet_size).
            X_weight (jax.numpy.ndarray, optional): Sample weights of shape (n_samples,). Defaults to ones.
                                                     In the original code, this was used for inertia calculation and M-step.
        """
        _X = jnp.asarray(X)
        original_shape = _X.shape
        if _X.ndim > 2:
            _X = _X.reshape(original_shape[0], -1) # Flatten features if X is like (N, L, A)

        if X_weight is None:
            _X_weight = jnp.ones(_X.shape[0])
        else:
            _X_weight = jnp.asarray(X_weight)

        if self.n_clusters == 1:
            # Simplified case for a single cluster
            mean_ = jnp.sum(_X * _X_weight[:, None], axis=0, keepdims=True) / (_X_weight.sum() + 1e-8)
            self.means_ = mean_.reshape((1,) + original_shape[1:]) if X.ndim > 2 else mean_
            self.labels_ = jnp.zeros(original_shape[0], dtype=jnp.int32)
            self.cat_ = jnp.ones((1,))
            # Calculate inertia for single cluster
            chosen_means = self.means_[self.labels_]
            if X.ndim > 2:
                 chosen_means_flat = chosen_means.reshape(original_shape[0],-1)
            else:
                 chosen_means_flat = chosen_means
            self.inertia_ = jnp.sum(_X_weight * jnp.sum(jnp.square(_X - chosen_means_flat), axis=-1))


        else:
            results = self._core_kmeans_jit(_X, _X_weight, self.n_clusters, self.n_init, self.max_iter, self.tol, self.seed)

            self.means_ = results["means"]
            if X.ndim > 2: # Reshape means back if original X was (N, L, A)
                 self.means_ = self.means_.reshape((self.n_clusters,) + original_shape[1:])
            self.labels_ = results["labels"] # Integer labels
            self.inertia_ = results["inertia"]
            self.cat_ = results["cat"]

        return self

    def predict(self, X, **kwargs):
        """
        Predict the closest cluster each sample in X belongs to.

        Args:
            X (jax.numpy.ndarray): New data, shape (n_samples, n_features) or (n_samples, length, alphabet_size).

        Returns:
            jax.numpy.ndarray: Index of the cluster each sample belongs to.
        """
        if self.means_ is None:
            raise ValueError("Model has not been_fit yet.")

        _X = jnp.asarray(X)
        original_shape = _X.shape
        if _X.ndim > 2:
            _X = _X.reshape(original_shape[0], -1)

        # Reshape means_ to be 2D for distance calculation if it was reshaped in fit
        _means_flat = self.means_
        if self.means_.ndim > 2 :
             _means_flat = self.means_.reshape(self.n_clusters, -1)

        return self._dist_jit(_X, _means_flat).argmin(-1)

    def sample(self, num_samples: int, seed: int = None, **kwargs):
        """
        Generate new sequences by sampling from the learned KMeans clusters.
        Assumes means_ are in the shape (k, L, A) for sequence data.

        Args:
            num_samples (int): The number of sequences to generate.
            seed (int, optional): Seed for random number generation. If None, uses object's seed.

        Returns:
            dict: A dictionary containing 'sampled_msa' (generated sequences as one-hot)
                  and 'sampled_labels' (cluster labels from which they were sampled).
                  Returns JAX arrays.
        """
        if self.means_ is None or self.cat_ is None:
            raise ValueError("Model has not been_fit yet or means/cat_ are not available.")
        if self.means_.ndim < 3:
            raise ValueError("Sampling is designed for sequence data where means are (k, L, A).")

        if seed is None:
            # If no seed provided for sampling, use the model's seed to generate a new key for this call
            key = jax.random.PRNGKey(self.seed)
            key, key_for_sampling = jax.random.split(key) # Ensure model's self.seed state isn't directly reused if it were a key
        elif isinstance(seed, int):
            key_for_sampling = jax.random.PRNGKey(seed)
        else: # Assume seed is already a JAX PRNGKey
            key_for_sampling = seed

        N_orig, L, A = self.means_.shape[0], self.means_.shape[1], self.means_.shape[2] # K, L, A from means

        key_labels, key_sample_uniform = jax.random.split(key_for_sampling, 2)

        # Sample labels based on category proportions
        sampled_cluster_labels = jax.random.choice(key_labels, jnp.arange(self.n_clusters),
                                                  shape=(num_samples,), p=self.cat_)

        # Get the means for these sampled_cluster_labels
        sampled_cluster_means_probs = self.means_[sampled_cluster_labels] # (num_samples, L, A), these are probabilities

        # Sample MSA: For each position, sample an amino acid based on the probabilities in the mean
        # This converts probability distributions (means) into one-hot encoded sequences
        # sampled_msa_one_hot = (sampled_cluster_means_probs.cumsum(-1) >= jax.random.uniform(key_sample, shape=(num_samples, L, 1))).argmax(-1)
        # The above line is for converting probabilities to one-hot. If means_ are already probabilities, use categorical.
        # If means_ are actual points (e.g. from protein embeddings), this sampling needs re-evaluation.
        # The original `kmeans_sample` implies means are probabilities for categorical sampling.

        # Sample actual characters (as integers)
        # Ensure probabilities sum to 1 for jax.random.categorical
        probs_normalized = sampled_cluster_means_probs / (jnp.sum(sampled_cluster_means_probs, axis=-1, keepdims=True) + 1e-8)

        # We need to iterate over L positions for each sample to use categorical
        # This is slow. A faster way is to use Gumbel-max trick if means are logits, or sample from cumsum.
        # Using cumsum trick for probabilities:
        uniform_samples = jax.random.uniform(key_sample_uniform, shape=(num_samples, L, 1))
        sampled_msa_indices = (jnp.cumsum(probs_normalized, axis=-1) > uniform_samples).argmax(-1)

        # Convert indices to one-hot encoded MSA
        sampled_msa_one_hot = jax.nn.one_hot(sampled_msa_indices, num_classes=A)

        return {
            "sampled_msa": sampled_msa_one_hot, # (num_samples, L, A)
            "sampled_labels": sampled_cluster_labels # (num_samples,)
        }

    def get_parameters(self):
        """Return learned model parameters."""
        if self.means_ is None:
            return None
        return {
            "means": self.means_,
            "labels": self.labels_, # These are labels for the training data
            "inertia": self.inertia_,
            "cat": self.cat_
        }

    def load_parameters(self, params):
        """Load model parameters."""
        if "means" not in params:
            raise ValueError("Parameters must include 'means'.")
        self.means_ = params["means"]
        self.labels_ = params.get("labels")
        self.inertia_ = params.get("inertia")
        self.cat_ = params.get("cat")

    # Example of how fit used to be called in mrf.py for initialization
    # kms = kmeans(X, X_weight, k=k) -> this is the old top-level function
    # self.inputs["labels"] = kms["labels"] -> these are one-hot in old kmeans.py
    # mb = jnp.log(kms["means"] * self.Neff + (lam+1e-8) * jnp.log(self.Neff))
    # self.params["mb"] = mb - mb.mean(-1,keepdims=True)
    # So, the fit method should store self.means_ and self.labels_ (as integers)
    # and self.cat_
    # The `kmeans` function from original `kmeans.py` returned a dict, this class stores them as attributes.
    # The sampling part also needs to be aligned with how MRF might use it.
    # The original `kmeans_sample` returned a dict with numpy arrays. This one returns JAX arrays.
    # The `kmeans` function in `kmeans.py` had a reshape for means if input was (N,L,A)
    # This is now handled in fit and predict.
    # The `labels` returned by original `_kmeans` (core) were one-hot.
    # The main `kmeans` function converted them using `_E` which did argmin.
    # This refactored version's `_core_kmeans` now returns integer labels directly.
    # The `kmeans_sample` function in the original `kmeans.py` took `msa` and `msa_weights`,
    # ran kmeans, then sampled. Here, `sample` is a method of the fitted KMeans object.
    # It also had an option to use `kms["labels"]` if `samples is None`.
    # This is slightly different; here, we always sample new labels based on `cat_`.
    # If that specific behavior is needed, sample could be adapted or a new method added.
