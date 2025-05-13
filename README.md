# Basic Perceptron Neural Network (Production-Ready Version)

This repository contains a refactored and enhanced implementation of a basic perceptron neural network. The code has been modularized into a Python package and improved with visualization, hardware acceleration support, testing, and documentation, making it ready for production use.

## Features and Improvements

- **Visualizations**  
  - Decision boundary plots (for 2-D data) at the start, midway, and end of training  
  - Training loss vs. epochs line chart  
  - Training accuracy vs. epochs line chart  

- **Modular Code Structure**  
  - The perceptron is implemented as a reusable Python package (`perceptron/`)  
  - Clear separation of concerns: model definition, training & evaluation, visualization, CLI  

- **Command-Line Interface (CLI)**  
  - Train, test, and visualize the perceptron via `python -m perceptron <command>`  
  - Configure data paths, epochs, learning rate, device (`cpu`/`cuda`), output paths, and visualization flag  

- **Hardware Compatibility**  
  - Built on PyTorch: run on CPU or GPU seamlessly (automatically falls back if CUDA not available)  

- **Testing**  
  - Unit tests with `pytest` for the model’s forward pass, prediction logic, and training convergence  

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/jalsarraf0/Basic-Perceptron-Neural-Network.git
   cd Basic-Perceptron-Neural-Network
   ```

2. **Create and activate a virtual environment** (recommended)  
   ```bash
   python3 -m venv .env
   source .env/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install as a package**  
   ```bash
   pip install .
   ```

## Usage

### Train
```bash
python -m perceptron train \
  --data path/to/train.csv \
  --epochs 500 \
  --lr 0.1 \
  --device cuda \
  --visualize \
  --output models/perceptron_model.pth
```

- If `--data` is omitted, a small example dataset is used.  
- Use `--device cpu` or `--device cuda`.  
- `--visualize` saves two plots alongside the model file:
  1. `<basename>_decision_boundaries.png`  
  2. `<basename>_metrics.png`  

### Test
```bash
python -m perceptron test \
  --model models/perceptron_model.pth \
  --data path/to/test.csv \
  --device cpu
```

Prints test accuracy and counts.

### Visualize (Post-Training)
```bash
python -m perceptron visualize \
  --model models/perceptron_model.pth \
  --data path/to/data.csv
```

Generates a decision boundary plot (only for 2-D feature data).

## Running Tests

```bash
pytest tests/
```

## Project Structure

```
Basic-Perceptron-Neural-Network/
├── perceptron/
│   ├── __init__.py
│   ├── __main__.py
│   ├── model.py
│   ├── training.py
│   └── visualize.py
├── tests/
│   ├── test_model.py
│   └── test_training.py
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](LICENSE) file for the full license text.
