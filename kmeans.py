import numpy as np


class KMeansFromScratch:
    """
    K-Means clustering implementation from scratch
    """

    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # Sum of squared distances to closest centroid

    def _euclidean_distance(self, X1, X2):
        """
        Calculate Euclidean distance between points

        Args:
            X1: First array of points
            X2: Second array of points

        Returns:
            Array of distances
        """
        return np.sqrt(np.sum((X1 - X2) ** 2, axis=1))

    def _initialize_centroids(self, X):
        """
        Initialize centroids using k-means++ initialization

        Args:
            X: Training data

        Returns:
            Array of initial centroids
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]

        # Choose remaining centroids
        for i in range(1, self.n_clusters):
            # Calculate distances to existing centroids
            distances = np.min([
                self._euclidean_distance(X, centroid)
                for centroid in centroids[:i]
            ], axis=0)

            # Choose next centroid with probability proportional to distance squared
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            centroids[i] = X[np.random.choice(n_samples, p=probabilities)]

        return centroids

    def _get_closest_centroids(self, X):
        """
        Assign each point to nearest centroid

        Args:
            X: Data points

        Returns:
            Array of cluster assignments
        """
        # Calculate distances to all centroids
        distances = np.array([
            self._euclidean_distance(X, centroid)
            for centroid in self.centroids
        ])

        # Return index of closest centroid for each point
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        """
        Update centroids based on mean of assigned points

        Args:
            X: Data points
            labels: Cluster assignments

        Returns:
            Array of new centroids
        """
        new_centroids = np.zeros_like(self.centroids)

        # For each cluster
        for k in range(self.n_clusters):
            # Get points assigned to this cluster
            mask = labels == k
            if np.sum(mask) > 0:  # Avoid empty clusters
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # If cluster is empty, keep old centroid
                new_centroids[k] = self.centroids[k]

        return new_centroids

    def _calculate_inertia(self, X, labels):
        """
        Calculate sum of squared distances to closest centroids

        Args:
            X: Data points
            labels: Cluster assignments

        Returns:
            float: Inertia value
        """
        distances = np.array([
            self._euclidean_distance(X, centroid)
            for centroid in self.centroids
        ])
        return np.sum(distances[labels, np.arange(len(X))] ** 2)

    def fit(self, X):
        """
        Fit K-Means clustering

        Args:
            X: Training data
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)

        # Iterate until convergence or max iterations
        for _ in range(self.max_iters):
            # Assign points to closest centroids
            old_labels = self.labels
            self.labels = self._get_closest_centroids(X)

            # Update centroids based on assignments
            self.centroids = self._update_centroids(X, self.labels)

            # Check for convergence
            if old_labels is not None and np.all(old_labels == self.labels):
                break

        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X, self.labels)

        return self

    def predict(self, X):
        """
        Predict cluster labels for new data

        Args:
            X: Data to cluster

        Returns:
            Array of cluster assignments
        """
        return self._get_closest_centroids(X)

    def fit_predict(self, X):
        """
        Fit and predict in one step

        Args:
            X: Data to cluster

        Returns:
            Array of cluster assignments
        """
        self.fit(X)
        return self.labels


# # Example usage:
# if __name__ == "__main__":
#     # Generate simple example data
#     np.random.seed(42)
#     X = np.concatenate([
#         np.random.normal(0, 1, (100, 2)),  # Cluster 1
#         np.random.normal(4, 1, (100, 2)),  # Cluster 2
#         np.random.normal(-4, 1, (100, 2))  # Cluster 3
#     ])
#
#     # Create and fit model
#     kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
#     labels = kmeans.fit_predict(X)
#
#     print("Cluster assignments:", labels)
#     print("Cluster centers:\n", kmeans.centroids)
#     print("Inertia:", kmeans.inertia_)
