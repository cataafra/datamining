import numpy as np
from collections import Counter


class KNNFromScratch:
    """
    K-Nearest Neighbors implementation from scratch
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two points

        Args:
            x1: First point (array-like)
            x2: Second point (array-like)

        Returns:
            float: Euclidean distance between x1 and x2
        """
        # Convert inputs to numpy arrays for vectorized operations
        x1 = np.array(x1)
        x2 = np.array(x2)

        # Calculate squared differences
        squared_diff = (x1 - x2) ** 2

        # Sum squared differences and take square root
        return np.sqrt(np.sum(squared_diff))

    def fit(self, X, y):
        """
        Store training data - KNN is a lazy learner and doesn't actually 'train'

        Args:
            X: Training features
            y: Training labels
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict_single(self, x):
        """
        Predict the class for a single sample

        Args:
            x: Sample to predict

        Returns:
            predicted_class: Predicted class label
        """
        # Calculate distances between x and all training samples
        distances = []
        for x_train in self.X_train:
            dist = self.euclidean_distance(x, x_train)
            distances.append(dist)

        # Convert to numpy array for easier indexing
        distances = np.array(distances)

        # Get indices of k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of k-nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Return most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        Predict classes for multiple samples

        Args:
            X: Samples to predict

        Returns:
            predictions: Array of predicted class labels
        """
        return np.array([self.predict_single(x) for x in X])

    def score(self, X, y):
        """
        Calculate accuracy score

        Args:
            X: Test features
            y: True labels

        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
