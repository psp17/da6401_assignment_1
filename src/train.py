import argparse
import json
import sys
import os
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from ann.neural_network import NeuralNetwork
from utils.data_loader import preprocess, oneHot, getBatches, getClassNames


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")
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
    parser.add_argument("--wandb_entity",          type=str,   default=None)
    parser.add_argument("--save_model",            type=str,   default=os.path.join(SRC_DIR, "best_model.npy"))
    parser.add_argument("--save_config",           type=str,   default=os.path.join(SRC_DIR, "best_config.json"))
    return parser.parse_args()


def logSampleImages(X_raw, y_raw, class_names):
    table = wandb.Table(columns=["class_id", "class_name", "image"])
    for cls in range(10):
        idxs = np.where(y_raw == cls)[0][:5]
        for idx in idxs:
            table.add_data(cls, class_names[cls], wandb.Image(X_raw[idx]))
    wandb.log({"sample_images": table})


def main():
    args        = parse_arguments()
    hidden_size = args.hidden_size[0] if isinstance(args.hidden_size, list) else args.hidden_size

    config = {
        "dataset":       args.dataset,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "loss":          args.loss,
        "optimizer":     args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay":  args.weight_decay,
        "num_layers":    args.num_layers,
        "hidden_size":   hidden_size,
        "activation":    args.activation,
        "weight_init":   args.weight_init,
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=config,
        name=f"{args.optimizer}_{args.activation}_lr{args.learning_rate}_hl{args.num_layers}_sz{hidden_size}",
    )

    X_train, y_train, X_val, y_val, X_test, y_test, X_raw, y_raw = preprocess(args.dataset)
    class_names = getClassNames(args.dataset)

    logSampleImages(X_raw, y_raw, class_names)

    y_train_oh = oneHot(y_train)
    y_val_oh   = oneHot(y_val)

    model = NeuralNetwork(args)

    best_val_f1  = 0.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []

        for X_b, y_b in getBatches(X_train, y_train_oh, args.batch_size):
            logits   = model.forward(X_b)
            loss_val = model.computedLoss(logits, y_b)
            epoch_losses.append(loss_val)
            model.backward(y_b, logits)
            model.update()

        val_logits = model.forward(X_val)
        val_pred   = np.argmax(val_logits, axis=1)
        val_loss   = model.computedLoss(val_logits, y_val_oh)
        val_acc    = accuracy_score(y_val, val_pred)
        val_f1     = f1_score(y_val, val_pred, average="weighted", zero_division=0)
        train_acc  = accuracy_score(y_train, model.predict(X_train))
        avg_loss   = float(np.mean(epoch_losses))

        print(f"Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  val_loss={val_loss:.4f}"
              f"  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        wandb.log({
            "epoch":      epoch,
            "train_loss": avg_loss,
            "val_loss":   float(val_loss),
            "train_acc":  float(train_acc),
            "val_acc":    float(val_acc),
            "val_f1":     float(val_f1),
        })

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = model.get_weights()

    model.set_weights(best_weights)

    test_logits = model.forward(X_test)
    test_pred   = np.argmax(test_logits, axis=1)
    test_acc    = accuracy_score(y_test, test_pred)
    test_f1     = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    print(f"\nBest Val F1: {best_val_f1:.4f}  |  Test Acc: {test_acc:.4f}  |  Test F1: {test_f1:.4f}")
    wandb.log({"test_accuracy": float(test_acc), "test_f1": float(test_f1), "best_val_f1": float(best_val_f1)})

    np.save(args.save_model, best_weights)
    config.update({"test_accuracy": float(test_acc), "test_f1": float(test_f1), "best_val_f1": float(best_val_f1)})
    with open(args.save_config, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved: {args.save_model}  |  {args.save_config}")
    wandb.finish()


if __name__ == "__main__":
    main()
