import argparse
import torch
import os
from perceptron.training import train, test_model, load_csv_data
from perceptron.model import PerceptronModel
from perceptron import visualize as viz

def main():
    parser = argparse.ArgumentParser(description="Perceptron CLI")
    sub = parser.add_subparsers(dest='command')
    t = sub.add_parser('train')
    t.add_argument('--data', type=str, default=None)
    t.add_argument('--epochs', type=int, default=100)
    t.add_argument('--lr', type=float, default=0.1)
    t.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    t.add_argument('--visualize', action='store_true')
    t.add_argument('--output', type=str, default='perceptron_model.pth')
    s = sub.add_parser('test')
    s.add_argument('--model', type=str, required=True)
    s.add_argument('--data', type=str, required=True)
    s.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    v = sub.add_parser('visualize')
    v.add_argument('--model', type=str, required=True)
    v.add_argument('--data', type=str, required=True)

    args = parser.parse_args()
    if args.command == 'train':
        train(data_path=args.data, epochs=args.epochs, lr=args.lr,
              device=args.device, visualize=args.visualize, output_model_path=args.output)
    elif args.command == 'test':
        test_model(args.model, args.data, device=args.device)
    elif args.command == 'visualize':
        X_np, y_np = load_csv_data(args.data)
        X_np, y_np = X_np.astype('float32'), y_np.astype('float32')
        if args.device=='cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Using CPU instead.")
            args.device='cpu'
        chk = torch.load(args.model, map_location='cpu')
        input_dim = chk.get('input_dim', None)
        if input_dim is None:
            for k,v in chk['state_dict'].items():
                if k.endswith('weight'):
                    input_dim = v.shape[1]; break
        model = PerceptronModel(input_dim)
        model.load_state_dict(chk['state_dict'])
        model.to(torch.device('cpu'))
        w = model.linear.weight.data.cpu().numpy().copy()
        b = float(model.linear.bias.data.cpu().numpy().copy())
        out = os.path.dirname(args.model) or '.'
        base = os.path.splitext(os.path.basename(args.model))[0]
        path = os.path.join(out, f"{base}_decision_boundaries.png")
        viz.plot_decision_boundaries([(w,b)], X_np, y_np, titles=["Decision Boundary"], filename=path)
        print(f"Saved decision boundary plot to {path}")

if __name__ == "__main__":
    main()
