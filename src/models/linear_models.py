"""Linear models: scikit-learn wrappers plus a from-scratch gradient descent regressor."""
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression

from src.config import RANDOM_STATE


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_lasso(X_train, y_train, alpha=1.0, max_iter=3000):
    model = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


class GradientDescentRegressor:
    """A plain-numpy linear regressor trained with batch gradient descent, kept
    around to show the mechanics behind LinearRegression rather than for production use.
    """

    def __init__(self, learning_rate=0.05, epochs=1000, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and ((epoch + 1) % 200 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1:4d} | Current MSE Loss: {loss:.2f}")

        return self

    def predict(self, X):
        return (np.dot(np.asarray(X), self.weights) + self.bias).flatten()
