import jax
import jax.numpy as jnp
import numpy as np
from openseq.models import KMeans
from openseq.utils.data_processing import ALPHABET # For sequence sampling interpretation

def run_kmeans_example():
    print("Running KMeans Example...")

    # 1. Generate some synthetic data for clustering
    key = jax.random.PRNGKey(42)
    key_data, key_sample = jax.random.split(key)

    # Example 1: Simple 2D feature data
    print("\nExample 1: Clustering 2D feature data")
    X_features = np.random.rand(100, 2) # 100 samples, 2 features
    X_features_jnp = jnp.asarray(X_features)

    # 2. Initialize KMeans model
    n_clusters = 3
    kmeans_model_feat = KMeans(n_clusters=n_clusters, seed=42, tol=1e-5, max_iter=100, n_init=5)

    # 3. Fit the model
    print(f"Fitting KMeans on data with shape: {X_features_jnp.shape}")
    kmeans_model_feat.fit(X_features_jnp)
    print("Fit complete.")

    # 4. Get learned parameters and predict
    params_feat = kmeans_model_feat.get_parameters()
    if params_feat and params_feat.get("means") is not None:
        print(f"Learned cluster centers (Means):\n{params_feat['means']}")
        print(f"Inertia: {params_feat['inertia']}")
        print(f"Category proportions: {params_feat['cat']}")

        labels_feat = kmeans_model_feat.predict(X_features_jnp)
        print(f"Predicted cluster labels for training data:\n{labels_feat}")
    else:
        print("Failed to retrieve parameters or model not fit.")

    # Example 2: Clustering sequence-like data (e.g., embeddings or one-hot MSAs)
    # For this example, let's assume means_ are probability distributions (L, A)
    # and we want to sample one-hot encoded sequences.
    # This part demonstrates the sample() method, assuming means_ have appropriate shape.

    print("\nExample 2: Sampling from KMeans (assuming means are sequence probabilities)")
    # Create a dummy KMeans model that looks like it was fit on sequence data
    # (num_clusters, length, alphabet_size)
    L_seq, A_seq = 10, len(ALPHABET) # 10 positions, full alphabet

    # Generate random probability distributions for cluster means
    dummy_means_probs = jax.random.uniform(key_data, (n_clusters, L_seq, A_seq))
    dummy_means_probs = dummy_means_probs / jnp.sum(dummy_means_probs, axis=-1, keepdims=True)

    dummy_cat_probs = jax.random.uniform(key_data, (n_clusters,))
    dummy_cat_probs = dummy_cat_probs / jnp.sum(dummy_cat_probs)

    kmeans_model_seq = KMeans(n_clusters=n_clusters, seed=0)
    # Manually set parameters as if it was fit
    kmeans_model_seq.load_parameters({
        "means": dummy_means_probs,
        "cat": dummy_cat_probs,
        "labels_": None, # Not needed for sampling only
        "inertia_": 0.0
    })

    num_generated_samples = 5
    print(f"Attempting to sample {num_generated_samples} sequences...")
    try:
        sampled_data = kmeans_model_seq.sample(num_samples=num_generated_samples, seed=123)
        sampled_msa = sampled_data["sampled_msa"]
        sampled_labels = sampled_data["sampled_labels"]
        print(f"Generated {sampled_msa.shape[0]} sequences of length {sampled_msa.shape[1]} with alphabet size {sampled_msa.shape[2]}.")
        print(f"Sampled cluster labels for generated sequences: {sampled_labels}")
        # print(f"First generated sequence (one-hot):\n{sampled_msa[0]}")
    except ValueError as e:
        print(f"Could not sample sequences: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during sampling: {e}")


if __name__ == "__main__":
    run_kmeans_example()
```
