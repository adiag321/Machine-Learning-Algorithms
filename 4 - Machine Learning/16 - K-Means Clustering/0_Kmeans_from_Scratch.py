#################################################
## K-Means Clustering from Scratch
#################################################
## Approach:
# 1. Choose the number of clusters (K)
# 2. Randomly initialize K centroids from the data
# 3. Assign each data point to the nearest centroid
# 4. Recompute centroids as the mean of assigned points
# 5. Repeat steps 3-4 until centroids stop moving
#    (or max iterations reached)
#################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# ============================================================
# Step 1: Define the KMeans class
# ============================================================
class KMeans:
    """
    K-Means clustering algorithm implemented from scratch.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form (default=3).
    max_iter : int
        Maximum number of iterations to run (default=100).
    random_state : int or None
        Seed for reproducibility (default=None).
    """

    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None       # will hold final centroid positions
        self.labels_ = None         # will hold cluster label for each point

    # --------------------------------------------------------
    # Step 2: Initialize centroids by randomly picking K points
    # --------------------------------------------------------
    def _initialize_centroids(self, X):
        """Pick K random data points as initial centroids."""
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[random_indices]

    # --------------------------------------------------------
    # Step 3: Compute Euclidean distance between two points
    # --------------------------------------------------------
    @staticmethod
    def _euclidean_distance(point1, point2):
        """Return the Euclidean distance between two points."""
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # --------------------------------------------------------
    # Step 4: Assign each point to the nearest centroid
    # --------------------------------------------------------
    def _assign_clusters(self, X):
        """
        For every data point, calculate its distance to each
        centroid and assign it to the closest one.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        labels = np.zeros(X.shape[0], dtype=int)

        for i, point in enumerate(X):
            # compute distance from this point to every centroid
            distances = [self._euclidean_distance(point, centroid)
                         for centroid in self.centroids]
            # the cluster label = index of the nearest centroid
            labels[i] = np.argmin(distances)

        return labels

    # --------------------------------------------------------
    # Step 5: Recompute centroids as the mean of each cluster
    # --------------------------------------------------------
    def _update_centroids(self, X, labels):
        """
        Calculate new centroid positions by averaging all
        data points that belong to each cluster.

        Returns
        -------
        new_centroids : np.ndarray of shape (n_clusters, n_features)
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            # select all points assigned to cluster k
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # if a cluster lost all its points, re-init randomly
                new_centroids[k] = X[np.random.randint(X.shape[0])]

        return new_centroids

    # --------------------------------------------------------
    # Step 6: Main fit_predict loop
    # --------------------------------------------------------
    def fit_predict(self, X):
        """
        Run the full K-Means algorithm:
          1. Initialize centroids
          2. Repeat until convergence or max_iter:
             a. Assign clusters
             b. Update centroids
             c. Check if centroids have changed

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster label for each data point.
        """
        # initialize centroids
        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            # --- assign each point to the nearest centroid ---
            labels = self._assign_clusters(X)

            # --- compute new centroids ---
            new_centroids = self._update_centroids(X, labels)

            # --- check for convergence ---
            # if centroids didn't move, we can stop early
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {iteration + 1}")
                break

            self.centroids = new_centroids
        else:
            print(f"Reached max iterations ({self.max_iter})")

        self.labels_ = labels
        return labels


# ============================================================
# Step 7: Generate sample data and run K-Means
# ============================================================
if __name__ == "__main__":

    # --- create synthetic data with 4 real clusters ---
    true_centers = [(-5, -5), (5, 5), (-2.5, 2.5), (2.5, -2.5)]
    X, y_true = make_blobs(
        n_samples=200,
        centers=true_centers,
        cluster_std=1.0,
        n_features=2,
        random_state=2,
    )

    # --- run our KMeans ---
    km = KMeans(n_clusters=4, max_iter=100, random_state=42)
    predicted_labels = km.fit_predict(X)

    # --------------------------------------------------------
    # Step 8: Visualize the results
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left plot: original (true) labels
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=40, alpha=0.7)
    axes[0].set_title("Original (True) Clusters")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    # right plot: predicted labels + centroids
    axes[1].scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap="viridis", s=40, alpha=0.7)
    axes[1].scatter(
        km.centroids[:, 0], km.centroids[:, 1],
        c="red", marker="X", s=200, edgecolors="black", linewidths=1.5,
        label="Centroids",
    )
    axes[1].set_title("K-Means Predicted Clusters")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].legend()

    plt.tight_layout()
    plt.show()