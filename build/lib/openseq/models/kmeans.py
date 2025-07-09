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

        def loop_plus_plus(carry_m, carry_c):
            m_centers, k_key = carry_m
            n_iter_idx, _ = carry_c # n_iter_idx is the current cluster index being initialized

            # inf_mask = jnp.inf * (jnp.arange(n_clusters) > n_iter_idx) # Original logic, might be complex for scan
            # p = (inf_mask + self._dist_jit(X, m_centers)).min(-1)

            # Calculate distances to existing centers
            distances_to_centers = self._dist_jit(X, m_centers)
            # For clusters not yet picked, their "contribution" to min_dist should be infinite
            # This logic is tricky with scan's fixed shapes. We use a simpler approach for now
            # by considering distances only to valid, already picked centers.
            # A mask for valid centers (those not all-zero, assuming centers are initialized to zero)
            valid_centers_mask = jnp.any(m_centers != 0, axis=1)

            min_dist_sq = jnp.ones(n_samples) * jnp.inf

            # This loop is less efficient than the original but fits scan better
            # It calculates min distances to current valid centers
            for i in range(n_iter_idx): # Only consider up to the centers already chosen
                 min_dist_sq = jnp.minimum(min_dist_sq, distances_to_centers[:, i])

            p = min_dist_sq # Probabilities are proportional to D(x)^2

            # Sample candidates
            candidates_idx = jax.random.choice(k_key, jnp.arange(n_samples),
                                             shape=(n_candidates,),
                                             p=p / (p.sum() + 1e-8), replace=False) # add epsilon for stability

            # Pick sample that decreases inertia the most (approximated)
            # For each candidate, calculate the sum of squared distances if it were chosen as the next center
            candidate_X = X[candidates_idx]

            # Calculate potential new minimum distances if each candidate is chosen
            # dist_if_candidate = jnp.minimum(p[:, None], self._dist_jit(X, candidate_X)) # Original logic
            # Simplified: pick the candidate that is furthest from existing centers (weighted by p)
            # This is not exactly k-means++'s "decrease inertia the most" but a common heuristic.

            # A simpler way for scan: find the candidate that results in the largest sum of minimum distances to it from other points
            # This is also not standard k-means++ but works with scan.
            # The original _kmeans_plus_plus is hard to directly JIT with scan due to dynamic shapes/masking.
            # For now, we'll stick to the original logic which was outside scan in the provided code,
            # meaning _kmeans_plus_plus itself is JITted but its internal loop is Python.
            # The provided code had a jax.lax.scan for the main loop of picking centers in k-means++,
            # so we replicate that structure.

            # Replicating the original provided scan loop for kmeans++
            # `m` here are the centers being built up. `n` is the index of the center to find.
            # `k` is the PRNG key for this step.

            # Distances from all points to the currently selected centers
            current_min_distances = self._dist_jit(X, m_centers[:n_iter_idx])
            if n_iter_idx > 0:
                p_dist = current_min_distances.min(axis=1) # Min distance from each point to any existing center
            else: # First center, all points are equally likely (or use random choice as below)
                p_dist = jnp.ones(n_samples) / n_samples


            # Sample candidates based on p_dist
            candidates = jax.random.choice(k_key, jnp.arange(n_samples),
                                         shape=(n_candidates,),
                                         p=p_dist / (p_dist.sum() + 1e-8), replace=False)

            # Pick sample that decreases inertia the most from these candidates
            # Calculate reduction in inertia for each candidate
            best_candidate_idx = -1
            max_inertia_reduction = -jnp.inf

            # This inner part is tricky for scan. The original code's scan loop for k-means++ was:
            # def loop(m,c): n,k = c; ...; return m.at[n].set(X[i]), None
            # `m` was `init_means`, `n` was `arange(1,n_clusters)`.
            # The logic inside that loop needs to be careful with JAX transformations.

            # For simplicity in this refactor, and given the original was already JITted as a whole,
            # we'll assume the original structure of _kmeans_plus_plus is sound for JIT.
            # The critical part is that the loop inside _kmeans_plus_plus (the `jax.lax.scan` call)
            # correctly updates `m_centers`.

            # The provided `_kmeans_plus_plus` had this structure:
            # i = jax.random.choice(key,jnp.arange(n_samples))
            # init_means = jnp.zeros((n_clusters,n_features)).at[0].set(X[i])
            # carry = (jnp.arange(1,n_clusters), jax.random.split(key, n_clusters-1))
            # return jax.lax.scan(loop_inner_kmeans_plus_plus, init_means, carry)[0]
            # where `loop_inner_kmeans_plus_plus` is the one that finds the next center.
            # We will define that inner loop here.

            # This is the inner loop of the original _kmeans_plus_plus
            # `m_centers` is the current set of means, `n_iter_idx_and_key` is (index of center to find, key)
            m, iter_details = carry_m, carry_c
            n_center_to_find, current_key = iter_details

            current_distances_to_chosen_centers = self._dist_jit(X, m) # Dist to all current centers

            # Mask for valid centers (those already chosen or being chosen)
            # This is complex because m has fixed shape n_clusters.
            # We assume m's rows > n_center_to_find are zero / invalid.
            # A simpler way: p_for_choice should be based on distances to centers 0 to n_center_to_find-1

            dist_to_prev_centers = self._dist_jit(X, m[:n_center_to_find])

            if n_center_to_find == 0: # First center case (handled by initial random choice outside scan)
                 # This path shouldn't be hit if initial center is set before scan
                p_for_choice = jnp.ones(n_samples) / n_samples
            else:
                p_for_choice = dist_to_prev_centers.min(axis=1)


            sampled_candidate_indices = jax.random.choice(current_key, jnp.arange(n_samples),
                                         shape=(n_candidates,),
                                         p=p_for_choice/(p_for_choice.sum() + 1e-8), replace=False)

            candidate_points = X[sampled_candidate_indices]

            # Evaluate reduction in inertia for each candidate
            # Min distances if current centers + one candidate are used
            potential_min_dists = jnp.minimum(p_for_choice[:, None], self._dist_jit(X, candidate_points))

            # Sum of these minimum distances (weighted) for each candidate
            # Note: original code used X_weight here. Assuming it's available in this scope.
            inertia_scores_for_candidates = (X_weight[:, None] * potential_min_dists).sum(0)

            # Pick candidate that minimizes this sum
            best_candidate_local_idx = inertia_scores_for_candidates.argmin()
            chosen_point_idx = sampled_candidate_indices[best_candidate_local_idx]

            updated_m = m.at[n_center_to_find].set(X[chosen_point_idx])
            return updated_m, None

        # Initial center choice
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

        current_seed = self.seed if seed is None else seed
        key = jax.random.PRNGKey(current_seed)

        N_orig, L, A = self.means_.shape[0], self.means_.shape[1], self.means_.shape[2] # K, L, A from means

        key, key_labels, key_sample = jax.random.split(key, 3)

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
        uniform_samples = jax.random.uniform(key_sample, shape=(num_samples, L, 1))
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
```
