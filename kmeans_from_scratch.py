"""
K-Means Clustering Algorithm (from scratch)
Description:
    A minimal but complete implementation of K-Means without external ML libraries. Shows vector math, convergence detection, reproducibility, and clean structure.
"""

import random
import numpy as np


class KMeans:
    """Simple KMeans clustering implementation."""
    def __init__(self, k=3, max_iters=100, tol=1e-4, seed=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        random.seed(seed)

    def _initialize_centroids(self, X):
        """Randomly selects k points as initial cluster centers."""
        idx = random.sample(range(len(X)), self.k)
        return X[idx]

    def _assign_clusters(self, X, centroids):
        """Assign each point to the nearest centroid."""
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Recompute centroids as mean of assigned points."""
        new_centroids = []
        for i in range(self.k):
            points = X[labels == i]
            new_centroids.append(points.mean(axis=0) if len(points) else np.zeros(X.shape[1]))
        return np.array(new_centroids)

    def fit(self, X):
        """Performs K-Means clustering."""
        X = np.array(X)
        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            # If centroids barely change → converged
            if np.linalg.norm(centroids - new_centroids) < self.tol:
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

    def predict(self, X):
        """Predict cluster for new data points."""
        X = np.array(X)
        distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    X = np.array([
        [1.0, 2.0], [1.2, 1.9], [0.8, 2.3],
        [7.0, 8.0], [7.5, 7.8], [6.8, 8.1]
    ])

    kmeans = KMeans(k=2)
    kmeans.fit(X)

    print("Final centroids:")
    print(kmeans.centroids)

    print("Labels:", kmeans.labels)
