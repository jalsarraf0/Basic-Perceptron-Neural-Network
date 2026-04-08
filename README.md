# Basic Perceptron Neural Network

A single-layer perceptron for binary classification, implemented as a structured Python package with training, evaluation, visualization, and a command-line interface.

## Overview

This project demonstrates the fundamentals of a perceptron — the simplest form of a neural network. It uses PyTorch to build and train a linear binary classifier, and provides tooling to visualize decision boundaries and training metrics. The code is organized as a reusable package with unit tests.

## Project Structure

```
Basic-Perceptron-Neural-Network/
├── perceptron/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   ├── model.py         # PerceptronModel (nn.Module)
│   ├── training.py      # Training and evaluation logic
│   └── visualize.py     # Decision boundary and metrics plots
├── tests/
│   ├── test_model.py
│   └── test_training.py
├── requirements.txt
└── LICENSE
```

## How to Run

**1. Clone and set up the environment**

```bash
git clone https://github.com/jalsarraf0/Basic-Perceptron-Neural-Network.git
cd Basic-Perceptron-Neural-Network
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Train**

```bash
python -m perceptron train \
  --data path/to/train.csv \
  --epochs 500 \
  --lr 0.1 \
  --device cpu \
  --visualize \
  --output models/perceptron_model.pth
```

Omitting `--data` uses a built-in example dataset. `--visualize` saves a decision boundary plot and a metrics plot alongside the model file.

**3. Test**

```bash
python -m perceptron test \
  --model models/perceptron_model.pth \
  --data path/to/test.csv \
  --device cpu
```

**4. Visualize (post-training)**

```bash
python -m perceptron visualize \
  --model models/perceptron_model.pth \
  --data path/to/data.csv
```

Only works with 2-feature datasets (generates a 2-D decision boundary plot).

**5. Run tests**

```bash
pytest tests/
```

## Dependencies

- Python 3.8+
- torch >= 1.8.0
- numpy
- matplotlib
- pytest

Install all with `pip install -r requirements.txt`.

## License

Licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
