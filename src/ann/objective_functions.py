import numpy as np


class ObjectiveFunction:

    def cross_entropy(self, y_pred, y_true):
        eps = 1e-12
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    def cross_entropy_grad(self, y_pred, y_true):
        # Gradient of cross-entropy through softmax: (probs - y_true) / batch
        return (y_pred - y_true) / y_pred.shape[0]

    def mse(self, y_pred, y_true):
        return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    def mse_grad(self, y_pred, y_true):
        # Gradient of MSE through softmax jacobian
        batch  = y_pred.shape[0]
        dL_dA  = 2.0 * (y_pred - y_true) / batch
        s      = np.sum(y_pred * dL_dA, axis=1, keepdims=True)
        return y_pred * (dL_dA - s)

    def get(self, name):
        mapping = {
            "cross_entropy": (self.cross_entropy, self.cross_entropy_grad),
            "mse":           (self.mse,           self.mse_grad),
        }
        if name not in mapping:
            raise ValueError(f"Unknown loss '{name}'. Choose: cross_entropy, mse")
        return mapping[name]
