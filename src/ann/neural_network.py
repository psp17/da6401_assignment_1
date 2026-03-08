import argparse
import numpy as np
from .neural_layer import Neural_layer
from .activations import Activation
from .objective_functions import ObjectiveFunction
from .optimizers import getOptimizer


class NeuralNetwork:

    def __init__(self, cli_args=None, input_size=784, num_classes=10, hidden_size=128,
                 num_layers=3, activation="relu", weight_init="xavier",
                 loss="cross_entropy", optimizer="rmsprop", lr=0.001, weight_decay=0.0):

        if isinstance(cli_args, argparse.Namespace):
            ns           = cli_args
            input_size   = getattr(ns, "input_size",    784)
            num_classes  = getattr(ns, "num_classes",   10)
            hidden_size  = getattr(ns, "hidden_size",   [128])
            num_layers   = getattr(ns, "num_layers",    3)
            activation   = getattr(ns, "activation",    "relu")
            weight_init  = getattr(ns, "weight_init",   "xavier")
            loss         = getattr(ns, "loss",          "cross_entropy")
            optimizer    = getattr(ns, "optimizer",     "rmsprop")
            lr           = getattr(ns, "learning_rate", 0.001)
            weight_decay = getattr(ns, "weight_decay",  0.0)
        elif isinstance(cli_args, dict):
            cfg          = cli_args
            input_size   = cfg.get("input_size",    784)
            num_classes  = cfg.get("num_classes",   10)
            hidden_size  = cfg.get("hidden_size",   hidden_size)
            num_layers   = cfg.get("num_layers",    3)
            activation   = cfg.get("activation",    "relu")
            weight_init  = cfg.get("weight_init",   "xavier")
            loss         = cfg.get("loss",          "cross_entropy")
            optimizer    = cfg.get("optimizer",     "rmsprop")
            lr           = cfg.get("learning_rate", 0.001)
            weight_decay = cfg.get("weight_decay",  0.0)
        elif cli_args is not None:
            input_size = cli_args

        if isinstance(hidden_size, list):
            hidden_sizes = hidden_size
        else:
            hidden_sizes = [hidden_size] * num_layers

        if len(hidden_sizes) < num_layers:
            hidden_sizes = hidden_sizes + [hidden_sizes[-1]] * (num_layers - len(hidden_sizes))
        hidden_sizes = hidden_sizes[:num_layers]

        self.input_size   = input_size
        self.num_classes  = num_classes
        self.hidden_size  = hidden_sizes
        self.num_layers   = num_layers
        self.weight_decay = weight_decay

        self._act = Activation()
        self._obj = ObjectiveFunction()

        loss_norm = loss.lower().replace("-", "_").replace(" ", "_")
        if loss_norm in ("crossentropy", "cross_entropy", "ce"):
            loss = "cross_entropy"
        elif loss_norm in ("mse", "mean_squared_error", "meansquarederror"):
            loss = "mse"

        self.loss_fn, self.loss_grad = self._obj.get(loss)
        self.optimizer = getOptimizer(optimizer, lr, weight_decay)

        self.layers = []
        prev = input_size
        for h in hidden_sizes:
            self.layers.append(Neural_layer(prev, h, activation, weight_init))
            prev = h

        self.output_layer = Neural_layer(prev, num_classes, activation, weight_init)
        self.output_layer.act_fn   = lambda z: z
        self.output_layer.act_grad = lambda z: np.ones_like(z)

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        A = X
        for layer in self.layers:
            A = layer.forwardPass(A)
        logits = self.output_layer.forwardPass(A)
        return logits

    def backward(self, y_true, y_pred):
        logits = np.atleast_2d(np.asarray(y_pred, dtype=np.float64))
        y_true = np.asarray(y_true)
        batch  = logits.shape[0]

        if y_true.ndim == 1:
            oh = np.zeros((batch, self.num_classes), dtype=np.float64)
            oh[np.arange(batch), y_true.astype(int)] = 1.0
            y_true = oh

        probs  = self._act.softmax(logits)
        dZ_out = self.loss_grad(probs, y_true)

        self.output_layer.grad_W = (
            self.output_layer._cache_input.T @ dZ_out
            + self.weight_decay * self.output_layer.W
        )
        self.output_layer.grad_b = np.sum(dZ_out, axis=0, keepdims=True)
        dA = dZ_out @ self.output_layer.W.T

        for layer in reversed(self.layers):
            dZ           = dA * layer.act_grad(layer._cache_Z)
            layer.grad_W = layer._cache_input.T @ dZ + self.weight_decay * layer.W
            layer.grad_b = np.sum(dZ, axis=0, keepdims=True)
            dA           = dZ @ layer.W.T

        all_layers_reversed = list(reversed(self.AllLayers()))
        self.grad_W = np.empty(len(all_layers_reversed), dtype=object)
        self.grad_b = np.empty(len(all_layers_reversed), dtype=object)
        for i, layer in enumerate(all_layers_reversed):
            self.grad_W[i] = layer.grad_W
            self.grad_b[i] = layer.grad_b

        return self.grad_W, self.grad_b

    def computedLoss(self, logits, y_one_hot):
        logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
        y_one_hot = np.asarray(y_one_hot)
        if y_one_hot.ndim == 1:
            oh = np.zeros((logits.shape[0], self.num_classes), dtype=np.float64)
            oh[np.arange(logits.shape[0]), y_one_hot.astype(int)] = 1.0
            y_one_hot = oh
        probs = self._act.softmax(logits)
        loss  = self.loss_fn(probs, y_one_hot)
        if self.weight_decay > 0:
            l2 = sum(np.sum(l.W ** 2) for l in self.AllLayers())
            loss += 0.5 * self.weight_decay * l2
        return loss

    def update(self):
        self.optimizer.step(self.AllLayers())

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def AllLayers(self):
        return self.layers + [self.output_layer]

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.AllLayers()):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        if isinstance(weight_dict, np.ndarray) and weight_dict.ndim == 0:
            weight_dict = weight_dict.item()

        w_keys = sorted([k for k in weight_dict if k.startswith("W")],
                        key=lambda k: int(k[1:]))
        if w_keys:
            shapes = [np.array(weight_dict[k]).shape for k in w_keys]
            in_size  = shapes[0][0]
            out_size = shapes[-1][1]
            act = self.layers[0].activation_name if self.layers else "relu"
            wi  = "xavier"

            self.layers = []
            prev = in_size
            for s in shapes[:-1]:
                hidden = s[1]
                self.layers.append(Neural_layer(prev, hidden, act, wi))
                prev = hidden

            self.output_layer = Neural_layer(prev, out_size, act, wi)
            self.output_layer.act_fn   = lambda z: z
            self.output_layer.act_grad = lambda z: np.ones_like(z)

            self.input_size  = in_size
            self.num_classes = out_size

        for i, layer in enumerate(self.AllLayers()):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = np.array(weight_dict[w_key], dtype=np.float64)
            if b_key in weight_dict:
                b = np.array(weight_dict[b_key], dtype=np.float64)
                layer.b = b.reshape(1, -1) if b.ndim == 1 else b

    def save(self, path):
        np.save(path, self.get_weights())

    def load(self, path):
        self.set_weights(np.load(path, allow_pickle=True).item())


Neural_Network = NeuralNetwork
