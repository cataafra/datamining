import numpy as np
from collections import Counter


class Node:
    """
    Node class for Decision Tree
    Each node represents a split in the tree
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Which feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # For leaf nodes, stores the predicted class


class DecisionTreeFromScratch:
    """
    Decision Tree Classifier implementation from scratch
    Uses binary splits based on feature thresholds
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree

        Args:
            X: Training features
            y: Training labels
        """
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)

    def _entropy(self, y):
        """
        Calculate entropy of a node

        Args:
            y: Array of class labels

        Returns:
            float: Entropy value
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        # Only consider non-zero probabilities to avoid log(0)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def _information_gain(self, parent, left_child, right_child):
        """
        Calculate information gain for a split

        Args:
            parent: Parent node labels
            left_child: Left child node labels
            right_child: Right child node labels

        Returns:
            float: Information gain
        """
        # Calculate weights for each child
        w_l = len(left_child) / len(parent)
        w_r = len(right_child) / len(parent)

        # Calculate gain = parent entropy - weighted sum of children entropy
        gain = self._entropy(parent)
        gain -= w_l * self._entropy(left_child)
        gain -= w_r * self._entropy(right_child)

        return gain

    def _best_split(self, X, y):
        """
        Find the best split for a node

        Args:
            X: Features
            y: Labels

        Returns:
            Dictionary containing best split information
        """
        best_split = {}
        best_info_gain = -1
        n_samples, n_features = X.shape

        # For each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # Try each value as a threshold
            for threshold in unique_values:
                # Split the data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Skip if split is degenerate
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate information gain
                info_gain = self._information_gain(
                    y,
                    y[left_mask],
                    y[right_mask]
                )

                # Update best split if this is better
                if info_gain > best_info_gain:
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask,
                        'info_gain': info_gain
                    }
                    best_info_gain = info_gain

        return best_split

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree

        Args:
            X: Features
            y: Labels
            depth: Current depth in the tree

        Returns:
            Node: Root node of the (sub)tree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_samples < self.min_samples_split or \
                n_labels == 1:
            # Create leaf node
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find best split
        best_split = self._best_split(X, y)

        # If no valid split found, create leaf node
        if not best_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Create child nodes
        left_mask = best_split['left_mask']
        right_mask = best_split['right_mask']

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_split['feature_idx'],
            threshold=best_split['threshold'],
            left=left,
            right=right
        )

    def _most_common_label(self, y):
        """
        Find most common label in a node

        Args:
            y: Array of labels

        Returns:
            Most common label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict class labels for samples in X

        Args:
            X: Features to predict

        Returns:
            Array of predicted labels
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample

        Args:
            x: Feature vector
            node: Current node

        Returns:
            Predicted class label
        """
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# # Example usage:
# if __name__ == "__main__":
#     # Generate simple example data
#     np.random.seed(42)
#     X = np.random.rand(100, 2)  # 100 samples, 2 features
#     y = np.random.randint(0, 2, 100)  # Binary classification
#
#     # Split into train and test
#     train_idx = np.random.choice([True, False], len(X), p=[0.8, 0.2])
#     X_train, X_test = X[train_idx], X[~train_idx]
#     y_train, y_test = y[train_idx], y[~train_idx]
#
#     # Create and train model
#     dt = DecisionTreeFromScratch(max_depth=5)
#     dt.fit(X_train, y_train)
#
#     # Make predictions
#     predictions = dt.predict(X_test)
#     accuracy = np.mean(predictions == y_test)
#     print("Predictions:", predictions)
#     print("True labels:", y_test)
#     print("Accuracy:", accuracy)