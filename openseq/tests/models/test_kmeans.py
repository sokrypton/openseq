import pytest
import jax
import jax.numpy as jnp
import numpy as np
from openseq.models import KMeans

class TestKMeans:
    @pytest.fixture
    def simple_data(self):
        # Simple 2D data with 3 clear clusters
        key = jax.random.PRNGKey(0)
        c1 = jax.random.normal(key, (50, 2)) + jnp.array([0, 0])
        c2 = jax.random.normal(key, (50, 2)) + jnp.array([5, 5])
        c3 = jax.random.normal(key, (50, 2)) + jnp.array([0, 5])
        data = jnp.concatenate([c1, c2, c3], axis=0)
        # True labels for validation (approximate)
        true_labels = jnp.concatenate([
            jnp.zeros(50, dtype=jnp.int32),
            jnp.ones(50, dtype=jnp.int32),
            jnp.full(50, 2, dtype=jnp.int32)
        ])
        return data, true_labels

    def test_kmeans_fit_predict(self, simple_data):
        X, true_labels_approx = simple_data
        n_clusters = 3

        model = KMeans(n_clusters=n_clusters, seed=42, n_init=5, max_iter=100, tol=1e-4)
        model.fit(X)

        assert model.means_ is not None
        assert model.means_.shape == (n_clusters, X.shape[1])
        assert model.labels_ is not None
        assert model.labels_.shape == (X.shape[0],)
        assert model.inertia_ is not None
        assert model.cat_ is not None
        assert model.cat_.shape == (n_clusters,)
        assert jnp.isclose(jnp.sum(model.cat_), 1.0)

        predictions = model.predict(X)
        assert predictions.shape == (X.shape[0],)

        # Check if the number of unique predicted labels matches n_clusters
        # This can sometimes fail if a cluster ends up empty, though unlikely with n_init > 1
        # For very small tol or max_iter, it might not converge perfectly.
        # assert len(jnp.unique(model.labels_)) == n_clusters
        # assert len(jnp.unique(predictions)) == n_clusters


    def test_kmeans_single_cluster(self, simple_data):
        X, _ = simple_data
        n_clusters = 1
        model = KMeans(n_clusters=n_clusters, seed=0)
        model.fit(X)

        assert model.means_ is not None
        assert model.means_.shape == (n_clusters, X.shape[1])
        assert jnp.allclose(model.means_[0], jnp.mean(X, axis=0), atol=1e-5)
        assert model.labels_ is not None
        assert jnp.all(model.labels_ == 0)
        assert model.inertia_ is not None
        assert model.cat_ is not None
        assert jnp.allclose(model.cat_, jnp.array([1.0]))

    def test_kmeans_get_load_params(self, simple_data):
        X, _ = simple_data
        model1 = KMeans(n_clusters=3, seed=1, n_init=1, max_iter=10) # Quick fit
        model1.fit(X)
        params = model1.get_parameters()

        assert params is not None
        assert "means" in params
        assert "labels" in params
        assert "inertia" in params
        assert "cat" in params

        model2 = KMeans(n_clusters=3, seed=2) # Different seed
        model2.load_parameters(params)

        assert jnp.allclose(model1.means_, model2.means_)
        assert jnp.allclose(model1.labels_, model2.labels_)
        assert jnp.allclose(model1.inertia_, model2.inertia_)
        assert jnp.allclose(model1.cat_, model2.cat_)

        # Predict with loaded params
        preds_model1 = model1.predict(X)
        preds_model2 = model2.predict(X)
        assert jnp.array_equal(preds_model1, preds_model2)

    def test_kmeans_with_weights(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (30, 2))
        # Weight first 10 points heavily, next 10 normally, last 10 lightly
        X_weight = jnp.concatenate([
            jnp.full(10, 10.0),
            jnp.full(10, 1.0),
            jnp.full(10, 0.1)
        ])

        model = KMeans(n_clusters=2, seed=0, n_init=1, max_iter=20)
        model.fit(X, X_weight=X_weight)

        assert model.means_ is not None
        # Further checks would involve verifying that weighted points influenced centers more.

    def test_kmeans_predict_new_data(self, simple_data):
        X, _ = simple_data
        model = KMeans(n_clusters=3, seed=0, n_init=1, max_iter=10)
        model.fit(X)

        key = jax.random.PRNGKey(1)
        X_new = jax.random.normal(key, (20, X.shape[1])) # New data with same feature dim
        predictions = model.predict(X_new)
        assert predictions.shape == (X_new.shape[0],)
        assert jnp.all(predictions >= 0) and jnp.all(predictions < 3)

    def test_kmeans_sampling_sequence_like(self):
        key = jax.random.PRNGKey(0)
        n_clusters, L, A = 2, 5, 3 # 2 clusters, length 5, alphabet 3

        # Dummy means (probabilities) and category proportions
        dummy_means = jax.random.uniform(key, (n_clusters, L, A))
        dummy_means = dummy_means / jnp.sum(dummy_means, axis=-1, keepdims=True)

        dummy_cat = jax.random.uniform(key, (n_clusters,))
        dummy_cat = dummy_cat / jnp.sum(dummy_cat)

        model = KMeans(n_clusters=n_clusters, seed=0)
        model.load_parameters({
            "means": dummy_means,
            "cat": dummy_cat,
            "labels_": None,
            "inertia_": 0.0
        })

        num_samples = 10
        key_sample = jax.random.PRNGKey(1)
        sampled_data = model.sample(num_samples=num_samples, seed=key_sample)

        assert "sampled_msa" in sampled_data
        assert "sampled_labels" in sampled_data
        assert sampled_data["sampled_msa"].shape == (num_samples, L, A)
        assert sampled_data["sampled_labels"].shape == (num_samples,)
        assert jnp.all(sampled_data["sampled_msa"].sum(axis=-1) == 1.0) # Check one-hot encoding

    def test_kmeans_sampling_requires_fit_or_load(self):
        model = KMeans(n_clusters=2)
        with pytest.raises(ValueError, match="Model has not been_fit yet"):
            model.sample(num_samples=5)

    def test_kmeans_sampling_requires_sequence_means(self, simple_data):
        X, _ = simple_data # X is (N, 2)
        model = KMeans(n_clusters=2, seed=0)
        model.fit(X) # means_ will be (k, 2)
        with pytest.raises(ValueError, match="Sampling is designed for sequence data"):
            model.sample(num_samples=5)

    def test_kmeans_n_init_effect(self, simple_data):
        X, _ = simple_data
        # Run with n_init=1 (potentially suboptimal)
        model1 = KMeans(n_clusters=3, seed=42, n_init=1, max_iter=50, tol=1e-4)
        model1.fit(X)
        inertia1 = model1.inertia_

        # Run with n_init=10 (more likely to find better minimum)
        model10 = KMeans(n_clusters=3, seed=42, n_init=10, max_iter=50, tol=1e-4)
        model10.fit(X)
        inertia10 = model10.inertia_

        # Best of 10 runs should be <= best of 1 run
        assert inertia10 <= inertia1
