import numpy as np


def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNetworkFromScratch:
    """
    Simple Neural Network implementation from scratch
    Architecture: Input Layer -> Hidden Layer -> Output Layer
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights and biases with random values
        Uses He initialization for better training
        """
        # Weights between input and hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))

        # Weights between hidden and output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        """
        Forward pass through the network

        Args:
            X: Input features

        Returns:
            Tuple of activations and cache for backpropagation
        """
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)

        return self.A2

    def backward(self, X, y, output):
        """
        Backward pass to update weights

        Args:
            X: Input features
            y: True labels (one-hot encoded)
            output: Predicted output from forward pass
        """
        m = X.shape[0]

        # Output layer gradients
        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network

        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print progress
        """
        n_samples = X.shape[0]

        # Convert labels to one-hot encoding
        y_onehot = np.zeros((n_samples, self.output_size))
        y_onehot[np.arange(n_samples), y] = 1

        for epoch in range(epochs):
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y_onehot[i:i + batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, output)

            if verbose and (epoch + 1) % 100 == 0:
                predictions = self.predict(X)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        """
        Predict class labels for samples in X

        Args:
            X: Features to predict

        Returns:
            Array of predicted labels
        """
        # Forward pass
        output = self.forward(X)

        # Return class with highest probability
        return np.argmax(output, axis=1)

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
