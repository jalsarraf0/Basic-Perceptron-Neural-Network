import numpy as np
import pytest
import torch
from perceptron.model import PerceptronModel

def test_forward_computation():
    model = PerceptronModel(2)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.linear.bias.copy_(torch.tensor([0.0]))
    x = torch.tensor([[1.0, 2.0]])
    output = model(x)
    assert pytest.approx(output.item(), rel=1e-6) == 0.0

def test_predict_threshold():
    model = PerceptronModel(2)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.linear.bias.copy_(torch.tensor([0.0]))
    inputs = [[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]]
    preds = model.predict(inputs)
    expected = np.array([1, 0, 1])
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
    assert np.array_equal(preds, expected)
