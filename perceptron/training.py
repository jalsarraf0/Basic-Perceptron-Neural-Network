import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from perceptron.model import PerceptronModel

def load_csv_data(file_path: str):
    try:
        data = np.loadtxt(file_path, delimiter=',')
    except Exception:
        data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def train(data_path: str = None, X: np.ndarray = None, y: np.ndarray = None,
          epochs: int = 100, lr: float = 0.1, device: str = 'cpu',
          visualize: bool = False, output_model_path: str = 'perceptron_model.pth'):
    if X is not None and y is not None:
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
    elif data_path is not None:
        X_np, y_np = load_csv_data(data_path)
        X_np = X_np.astype(np.float32)
        y_np = y_np.astype(np.float32)
    else:
        X_np = np.array([[0,0],[1,1],[1,0],[0,1]], dtype=np.float32)
        y_np = np.array([0,1,1,0], dtype=np.float32)

    input_dim = X_np.shape[1]
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU instead.")
        device = 'cpu'
    device = torch.device(device)

    X_tensor = torch.from_numpy(X_np).to(device)
    y_tensor = torch.from_numpy(y_np).view(-1,1).to(device)

    model = PerceptronModel(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses, accuracies = [], []
    initial_w = model.linear.weight.data.cpu().numpy().copy()
    initial_b = float(model.linear.bias.data.cpu().numpy().copy())
    mid_w = mid_b = None
    mid_epoch = epochs // 2

    for epoch in range(1, epochs+1):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs = model(X_tensor)
            loss_val = float(criterion(outputs, y_tensor).item())
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct = (preds == y_tensor.long()).sum().item()
            acc_val = correct / len(y_tensor)

        losses.append(loss_val)
        accuracies.append(acc_val)

        if visualize and epoch == mid_epoch:
            mid_w = model.linear.weight.data.cpu().numpy().copy()
            mid_b = float(model.linear.bias.data.cpu().numpy().copy())

    torch.save({'state_dict': model.state_dict(), 'input_dim': input_dim}, output_model_path)
    print(f"Training complete. Final loss: {losses[-1]:.4f}, Final accuracy: {accuracies[-1]*100:.2f}%")

    if visualize:
        from perceptron.visualize import plot_decision_boundaries, plot_training_metrics
        wb_list = [(initial_w, initial_b)]
        titles = ["Initial"]
        if mid_w is not None:
            wb_list.append((mid_w, mid_b))
            titles.append(f"Epoch {mid_epoch}")
        final_w = model.linear.weight.data.cpu().numpy().copy()
        final_b = float(model.linear.bias.data.cpu().numpy().copy())
        wb_list.append((final_w, final_b))
        titles.append(f"Final (Epoch {epochs})")

        out_dir = os.path.dirname(output_model_path) or '.'
        base = os.path.splitext(os.path.basename(output_model_path))[0]

        db_path = os.path.join(out_dir, f"{base}_decision_boundaries.png")
        plot_decision_boundaries(wb_list, X_np, y_np, titles, filename=db_path)
        tm_path = os.path.join(out_dir, f"{base}_metrics.png")
        plot_training_metrics(losses, accuracies, filename=tm_path)
        print(f"Saved decision boundary plot to {db_path}")
        print(f"Saved training metrics plot to {tm_path}")

    return model, losses, accuracies
