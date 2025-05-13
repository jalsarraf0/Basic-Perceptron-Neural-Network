import torch
import torch.nn as nn

class PerceptronModel(nn.Module):
    """
    A basic perceptron (single-layer neural network) for binary classification.
    """
    def __init__(self, input_dim: int):
        super(PerceptronModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)
        probs = torch.sigmoid(self.forward(x))
        preds = (probs >= 0.5).long().view(-1)
        return preds.cpu().numpy()
