import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegression:
    """
    Logistic Regression implementation using NumPy.
    Optimizer: Gradient Descent
    Loss: Binary Cross-Entropy
    """
    def __init__(self, learning_rate=0.01, n_iters=1000, threshold=0.5):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for _ in range(self.n_iters):
            
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)


            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            eps = 1e-15
            y_pred_safe = np.clip(y_predicted, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_safe) + (1 - y) * np.log(1 - y_pred_safe))
            self.losses.append(loss)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= self.threshold).astype(int)

class KNN:
    """
    K-Nearest Neighbors implementation using NumPy.
    Metric: Euclidean Distance
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels.astype(int)).argmax()
        return most_common


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if (tp + fp) == 0: return 0.0
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if (tp + fn) == 0: return 0.0
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if (prec + rec) == 0: return 0.0
    return 2 * (prec * rec) / (prec + rec)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


class ModelVisualizer:
    """
    Helper class for plotting model performance and comparisons.
    """
    
    @staticmethod
    def plot_loss_curve(losses, title="Training Loss"):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, color='teal', linewidth=2)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    @staticmethod
    def plot_confusion_matrix_heatmap(cm, title="Confusion Matrix", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(['Stay (0)', 'Leave (1)'])
        ax.set_yticklabels(['Stay (0)', 'Leave (1)'])
        
        if ax is None:
            plt.show()

    @staticmethod
    def plot_model_comparison(models, scores_dict, metrics):
        """
        Draws a grouped bar chart comparing multiple models across multiple metrics.
        scores_dict: {'ModelName': [score1, score2...]}
        metrics: ['Accuracy', 'F1', ...]
        """
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (model_name, scores) in enumerate(scores_dict.items()):
            offset = width * i - (width/2)
            rects = ax.bar(x + offset, scores, width, label=model_name)
            
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower center')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()