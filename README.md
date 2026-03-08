# DA6401 Assignment 1 — Multi-Layer Perceptron
Roll no: ns26z048 
Name: Prathamesh Sanjay Pednekar

NumPy-only MLP for MNIST and Fashion-MNIST classification.

## Project Structure

```
da6401_assignment_1/
├── models/
├── notebooks/
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── train.py
│   └── inference.py
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py -d mnist -e 20 -b 64 -l cross_entropy -o adam -lr 0.001 -wd 0.0005 -nhl 3 -sz 128 -a relu -w_i xavier
```

### CLI Arguments

| Flag | Description | Default | Choices |
|------|-------------|---------|---------|
| `-d` | Dataset | `mnist` | `mnist`, `fashion_mnist` |
| `-e` | Epochs | `10` | int |
| `-b` | Batch size | `64` | int |
| `-l` | Loss function | `cross_entropy` | `cross_entropy`, `mse` |
| `-o` | Optimizer | `adam` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `-lr` | Learning rate | `0.001` | float |
| `-wd` | Weight decay (L2) | `0.0` | float |
| `-nhl` | Hidden layers | `3` | int (≤ 6) |
| `-sz` | Neurons per layer | `128` | int (≤ 128) |
| `-a` | Activation | `relu` | `sigmoid`, `tanh`, `relu` |
| `-w_i` | Weight init | `xavier` | `random`, `xavier` |

## Inference

```bash
python src/inference.py --model models/best_model.npy --config models/best_config.json --dataset mnist
```
Github repository link : https://github.com/psp17/da6401_assignment_1.git
W&B report link: https://wandb.ai/ns26z048-iitm-india/da6401_assignment1/reports/da6401_assignment1--VmlldzoxNjEzNTYzOA?accessToken=k656xqb0qknw29ueg23itsx2owc8xi2bb99anrvva5hdzf13eihl24up7nrjkym5