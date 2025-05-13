import numpy as np
import torch
from perceptron.training import train

def test_training_improves_accuracy():
    torch.manual_seed(0)
    X = np.array([[0,0],[1,1],[1,0],[0,1]], dtype=np.float32)
    y = np.array([0,1,1,0], dtype=np.float32)
    model, losses, accuracies = train(X=X, y=y, epochs=1000, lr=0.1, device='cpu', visualize=False)
    assert accuracies[-1] == 1.0
    assert losses[-1] < losses[0]
