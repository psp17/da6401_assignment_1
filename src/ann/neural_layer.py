import numpy as np
from .activations import Activation


class Neural_layer:

    def __init__(self, in_features, out_features, activation="relu", weight_init="xavier"):
        self.in_features     = in_features
        self.out_features    = out_features
        self.activation_name = activation

        self.W = self.initialization_weights(weight_init, in_features, out_features)
        self.b = np.zeros((1, out_features), dtype=np.float64)

        _act = Activation()
        self.act_fn, self.act_grad = _act.get(activation)

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self._cache_input = None
        self._cache_Z     = None
        self.opt_state    = {}

    def initialization_weights(self, method, fan_in, fan_out):
        if method == "xavier":
            M = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-M, M, (fan_in, fan_out)).astype(np.float64)
        elif method == "random":
            return (np.random.randn(fan_in, fan_out) * 0.01).astype(np.float64)
        elif method == "zeros":
            return np.zeros((fan_in, fan_out), dtype=np.float64)
        else:
            raise ValueError(f"Unknown weight_init '{method}'. Choose: xavier, random, zeros")

    def forwardPass(self, A_prev):
        self._cache_input = A_prev
        b = self.b.reshape(1, -1) if self.b.ndim == 1 else self.b
        self._cache_Z     = A_prev @ self.W + b
        return self.act_fn(self._cache_Z)

    def backwardPass(self, dZ, weight_decay=0.0):
        self.grad_W = self._cache_input.T @ dZ + weight_decay * self.W
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T
