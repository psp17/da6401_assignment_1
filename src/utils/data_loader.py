import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


MNIST_CLASSES         = [str(i) for i in range(10)]
FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_raw_data(dataset_name):
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose mnist or fashion_mnist.")
    return X_train, y_train, X_test, y_test


def preprocess(dataset_name, val_split=0.1):
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_raw_data(dataset_name)

    # float64 - gradient precision
    X_train_flat = X_train_raw.reshape(len(X_train_raw), -1).astype(np.float64) / 255.0
    X_test_flat  = X_test_raw.reshape(len(X_test_raw),  -1).astype(np.float64) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_flat, y_train_raw,
        test_size=val_split, random_state=42, stratify=y_train_raw
    )

    return X_train, y_train, X_val, y_val, X_test_flat, y_test_raw, X_train_raw, y_train_raw


def oneHot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes), dtype=np.float64)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def getBatches(X, y_oh, batch_size, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        bi = idx[start:start + batch_size]
        yield X[bi], y_oh[bi]


def getClassNames(dataset_name):
    return MNIST_CLASSES if dataset_name == "mnist" else FASHION_MNIST_CLASSES
