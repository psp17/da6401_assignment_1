import numpy as np


class Activation:

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_grad(self, z):
        s = self.sigmoid(z)
        return s * (1.0 - s)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_grad(self, z):
        return 1.0 - np.tanh(z) ** 2

    def relu(self, z):
        return np.maximum(0, z)

    def relu_grad(self, z):
        return (z > 0).astype(np.float64)

    def softmax(self, z):
        # Ensure 2D so axis=1 is always valid (handles 1D input from autograder)
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z    = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def get(self, name):
        mapping = {
            "sigmoid": (self.sigmoid, self.sigmoid_grad),
            "tanh":    (self.tanh,    self.tanh_grad),
            "relu":    (self.relu,    self.relu_grad),
        }
        if name not in mapping:
            raise ValueError(f"Unknown activation '{name}'. Choose: sigmoid, tanh, relu")
        return mapping[name]
