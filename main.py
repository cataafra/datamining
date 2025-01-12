import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score
)
import time
import seaborn as sns

# Import our implementations
from knn import KNNFromScratch
from decision_tree import DecisionTreeFromScratch
from naive_bayes import NaiveBayesFromScratch
from neural_network import NeuralNetworkFromScratch
from kmeans import KMeansFromScratch


class AlgorithmComparison:
    def __init__(self):
        self.algorithms = {}
        self.results = {}
        self.training_times = {}
        self.prediction_times = {}
        self.detailed_metrics = {}

    def load_data(self):
        """Load and prepare the Iris dataset"""
        print("Loading and preparing data...")

        # Load Iris dataset
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        print("Data preparation completed")

    def initialize_algorithms(self):
        """Initialize all algorithm implementations"""
        print("Initializing algorithms...")

        self.algorithms = {
            'KNN': KNNFromScratch(k=3),
            'Decision Tree': DecisionTreeFromScratch(max_depth=4),
            'Naive Bayes': NaiveBayesFromScratch(),
            'Neural Network': NeuralNetworkFromScratch(
                input_size=4,
                hidden_size=8,
                output_size=3,
                learning_rate=0.01
            ),
            'K-Means': KMeansFromScratch(
                n_clusters=3,
                random_state=42
            )
        }

        print("Algorithms initialized")

    # def plot_confusion_matrix(self, y_true, y_pred, title):
    #     """Plot confusion matrix for a classifier"""
    #     cm = confusion_matrix(y_true, y_pred)
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #     plt.title(f'Confusion Matrix - {title}')
    #     plt.ylabel('True Label')
    #     plt.xlabel('Predicted Label')
    #     plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
    #     plt.close()

    def calculate_supervised_metrics(self, y_true, y_pred):
        """Calculate metrics for supervised learning algorithms"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }

    def calculate_clustering_metrics(self, X, labels):
        """Calculate metrics for clustering"""
        return {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels)
        }

    def train_and_evaluate(self):
        """Train and evaluate all algorithms"""
        print("\nTraining and evaluating algorithms...")

        for name, algorithm in self.algorithms.items():
            print(f"\nEvaluating {name}:")

            # Training time
            train_start = time.time()

            if name != 'K-Means':
                algorithm.fit(self.X_train_scaled, self.y_train)
            else:
                # For K-Means, use only the features
                X_for_kmeans = self.X_train_scaled.copy()
                algorithm.fit(X_for_kmeans)

            train_end = time.time()
            self.training_times[name] = train_end - train_start

            # Prediction time
            predict_start = time.time()

            if name != 'K-Means':
                y_pred = algorithm.predict(self.X_test_scaled)
                self.detailed_metrics[name] = self.calculate_supervised_metrics(self.y_test, y_pred)
                self.results[name] = self.detailed_metrics[name]['accuracy']

                # Plot confusion matrix
                self.plot_confusion_matrix(self.y_test, y_pred, name)
            else:
                # For K-Means, calculate clustering metrics
                X_test_for_kmeans = self.X_test_scaled.copy()
                labels = algorithm.predict(X_test_for_kmeans)
                self.detailed_metrics[name] = self.calculate_clustering_metrics(X_test_for_kmeans, labels)
                self.results[name] = self.detailed_metrics[name]['silhouette']

            predict_end = time.time()
            self.prediction_times[name] = predict_end - predict_start

            # Print results
            print("\nMetrics:")
            for metric, value in self.detailed_metrics[name].items():
                print(f"{metric}: {value:.4f}")
            print(f"Training Time: {self.training_times[name]:.4f} seconds")
            print(f"Prediction Time: {self.prediction_times[name]:.4f} seconds")

    def visualize_results(self):
        """Create visualizations comparing algorithm performance"""
        print("\nCreating visualizations...")

        # Create subplots grid
        fig = plt.figure(figsize=(15, 10))

        # Plot accuracy/silhouette scores
        plt.subplot(2, 2, 1)
        names = list(self.results.keys())
        values = list(self.results.values())
        plt.bar(names, values, color='skyblue')
        plt.title('Primary Metric (Accuracy/Silhouette)')
        plt.xticks(rotation=45)
        plt.ylabel('Score')

        # Plot training times
        plt.subplot(2, 2, 2)
        plt.bar(names, list(self.training_times.values()), color='lightgreen')
        plt.title('Training Time')
        plt.xticks(rotation=45)
        plt.ylabel('Time (seconds)')

        # Plot detailed metrics
        plt.subplot(2, 2, (3, 4))
        metrics_df = pd.DataFrame(self.detailed_metrics)
        bar_positions = np.arange(len(metrics_df.index))
        bar_width = 0.15
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for i, column in enumerate(metrics_df.columns):
            plt.bar(bar_positions + i * bar_width,
                    metrics_df[column],
                    bar_width,
                    label=column,
                    alpha=0.7)

        plt.title('Detailed Metrics Comparison')
        plt.xticks(bar_positions + bar_width * (len(metrics_df.columns) - 1) / 2,
                   metrics_df.index,
                   rotation=45)
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()

        print("Visualizations saved as 'algorithm_comparison.png'")

    def plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot confusion matrix for a classifier"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {title}')
        plt.colorbar()

        # Add number annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.close()

    def run_full_comparison(self):
        """Run complete comparison pipeline"""
        print("Starting algorithm comparison...")

        self.load_data()
        self.initialize_algorithms()
        self.train_and_evaluate()
        self.visualize_results()

        print("\nFinal Summary:")
        print("-" * 50)
        for name in self.algorithms.keys():
            print(f"\n{name}:")
            print("\nDetailed Metrics:")
            for metric, value in self.detailed_metrics[name].items():
                print(f"{metric}: {value:.4f}")
            print(f"Training Time: {self.training_times[name]:.4f} seconds")
            print(f"Prediction Time: {self.prediction_times[name]:.4f} seconds")


if __name__ == "__main__":
    comparison = AlgorithmComparison()
    comparison.run_full_comparison()
