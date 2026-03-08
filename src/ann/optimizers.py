import numpy as np


class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr           = lr
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr           = lr
        self.beta         = beta
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            s = layer.opt_state
            if "vW" not in s:
                s["vW"] = np.zeros_like(layer.W)
                s["vb"] = np.zeros_like(layer.b)
            s["vW"] = self.beta * s["vW"] + layer.grad_W
            s["vb"] = self.beta * s["vb"] + layer.grad_b
            layer.W -= self.lr * s["vW"]
            layer.b -= self.lr * s["vb"]


class nag:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr           = lr
        self.beta         = beta
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            s = layer.opt_state
            if "vW" not in s:
                s["vW"] = np.zeros_like(layer.W)
                s["vb"] = np.zeros_like(layer.b)
            s["vW"] = self.beta * s["vW"] + layer.grad_W
            s["vb"] = self.beta * s["vb"] + layer.grad_b
            layer.W -= self.lr * (self.beta * s["vW"] + layer.grad_W)
            layer.b -= self.lr * (self.beta * s["vb"] + layer.grad_b)


class RMSprop:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.lr           = lr
        self.beta         = beta
        self.epsilon      = epsilon
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            s = layer.opt_state
            if "sW" not in s:
                s["sW"] = np.zeros_like(layer.W)
                s["sb"] = np.zeros_like(layer.b)
            s["sW"] = self.beta * s["sW"] + (1 - self.beta) * layer.grad_W ** 2
            s["sb"] = self.beta * s["sb"] + (1 - self.beta) * layer.grad_b ** 2
            layer.W -= self.lr * layer.grad_W / (np.sqrt(s["sW"]) + self.epsilon)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(s["sb"]) + self.epsilon)


def getOptimizer(name, lr, weight_decay=0.0):
    mapping = {
        "sgd":      lambda: SGD(lr=lr, weight_decay=weight_decay),
        "momentum": lambda: momentum(lr=lr, weight_decay=weight_decay),
        "nag":      lambda: nag(lr=lr, weight_decay=weight_decay),
        "rmsprop":  lambda: RMSprop(lr=lr, weight_decay=weight_decay),
    }
    if name not in mapping:
        raise ValueError(f"Unknown optimizer '{name}'. Choose: sgd, momentum, nag, rmsprop")
    return mapping[name]()
