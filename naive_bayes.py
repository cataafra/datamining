import numpy as np


class NaiveBayesFromScratch:
    """
    Gaussian Naive Bayes implementation from scratch
    Assumes features follow a normal distribution within each class
    """

    def __init__(self):
        self.classes = None
        self.parameters = {}

    def _calculate_mean_var(self, X, y):
        """
        Calculate mean and variance for each feature within each class

        Args:
            X: Training features
            y: Training labels

        Returns:
            Dictionary containing mean and variance for each feature per class
        """
        parameters = {}

        # For each class
        for c in self.classes:
            # Get samples belonging to this class
            X_c = X[y == c]

            # Calculate mean and variance for each feature
            parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + 1e-9  # Add small value to avoid division by zero
            }

        return parameters

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier

        Args:
            X: Training features
            y: Training labels
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Calculate prior probabilities P(y)
        self.priors = {}
        for c in self.classes:
            self.priors[c] = np.mean(y == c)

        # Calculate mean and variance for each feature per class
        self.parameters = self._calculate_mean_var(X, y)

    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate likelihood P(x|y) using Gaussian probability density function

        Args:
            x: Feature value
            mean: Mean of the feature for a class
            var: Variance of the feature for a class

        Returns:
            float: Likelihood probability
        """
        # Gaussian probability density function
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _calculate_posterior(self, x):
        """
        Calculate posterior probability for each class

        Args:
            x: Sample to classify

        Returns:
            Dictionary of posterior probabilities for each class
        """
        posteriors = {}

        # Calculate posterior for each class
        for c in self.classes:
            # Start with prior probability
            posterior = np.log(self.priors[c])

            # Multiply by likelihood of each feature
            # Using log probabilities to avoid numerical underflow
            for i, x_i in enumerate(x):
                likelihood = self._calculate_likelihood(
                    x_i,
                    self.parameters[c]['mean'][i],
                    self.parameters[c]['var'][i]
                )
                posterior += np.log(likelihood + 1e-9)

            posteriors[c] = posterior

        return posteriors

    def predict(self, X):
        """
        Predict class labels for samples in X

        Args:
            X: Features to predict

        Returns:
            Array of predicted labels
        """
        predictions = []

        # For each sample
        for x in X:
            # Calculate posterior probability for each class
            posteriors = self._calculate_posterior(x)

            # Choose class with highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))

        return np.array(predictions)

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

# # Example usage:
# if __name__ == "__main__":
#     # Generate simple example data
#     np.random.seed(42)
#     X = np.random.randn(100, 2)  # 100 samples, 2 features
#     y = np.random.randint(0, 2, 100)  # Binary classification
#
#     # Split into train and test
#     train_idx = np.random.choice([True, False], len(X), p=[0.8, 0.2])
#     X_train, X_test = X[train_idx], X[~train_idx]
#     y_train, y_test = y[train_idx], y[~train_idx]
#
#     # Create and train model
#     nb = NaiveBayesFromScratch()
#     nb.fit(X_train, y_train)
#
#     # Make predictions
#     predictions = nb.predict(X_test)
#     accuracy = nb.score(X_test, y_test)
#     print("Predictions:", predictions)
#     print("True labels:", y_test)
#     print("Accuracy:", accuracy)
