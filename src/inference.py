import argparse
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from ann.neural_network import NeuralNetwork
from utils.data_loader import preprocess, getClassNames


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference — evaluate saved MLP")
    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist",          choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=30)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=64)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",  choices=["cross_entropy", "mse"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=4)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+",               default=[128])
    parser.add_argument("-a",   "--activation",    type=str,   default="relu",           choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",         choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project", type=str,   default="da6401_assignment1")
    parser.add_argument("--model_path",            type=str,   default=os.path.join(SRC_DIR, "best_model.npy"))
    parser.add_argument("--save_cm",               type=str,   default=os.path.join(SRC_DIR, "confusion_matrix.png"))
    return parser.parse_args()


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)

    y_oh = np.zeros((len(y_test), 10), dtype=np.float64)
    y_oh[np.arange(len(y_test)), y_test] = 1.0

    return {
        "logits":    logits,
        "loss":      float(model.computedLoss(logits, y_oh)),
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }


def plotConfusionMatrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args    = parse_arguments()
    model   = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    _, _, _, _, X_test, y_test, _, _ = preprocess(args.dataset)
    class_names = getClassNames(args.dataset)

    results = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 40)
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print("=" * 40 + "\n")

    y_pred = np.argmax(results["logits"], axis=1)
    cm     = confusion_matrix(y_test, y_pred)
    plotConfusionMatrix(cm, class_names, f"Confusion Matrix — {args.dataset}", args.save_cm)

    return results


if __name__ == "__main__":
    main()
