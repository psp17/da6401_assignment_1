import numpy as np
import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork

best_config = argparse.Namespace(
    dataset       = "mnist",
    epochs        = 30,
    batch_size    = 64,
    loss          = "cross_entropy",
    optimizer     = "rmsprop",
    weight_decay  = 0.0,
    learning_rate = 0.001,
    num_layers    = 4,
    hidden_size   = [128],
    activation    = "relu",
    weight_init   = "xavier"
)

model = NeuralNetwork(best_config)

model_path = os.path.join(SRC_DIR, "best_model.npy")
weights    = np.load(model_path, allow_pickle=True).item()
model.set_weights(weights)

X_test = np.random.rand(100, 784)
y_true = np.random.randint(0, 10, size=(100,))

y_pred = model.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

score = f1_score(y_true, y_pred_labels, average="macro")
print("F1 Score:", score)
print("forward() shape check:", y_pred.shape)
print("Test PASSED" if y_pred.shape == (100, 10) else "Test FAILED: wrong output shape")
