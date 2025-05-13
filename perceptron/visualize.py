import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(losses, accuracies, filename="training_metrics.png"):
    epochs = range(1, len(losses)+1)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(epochs, losses); axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True)
    axes[1].plot(epochs, accuracies); axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].grid(True)
    plt.tight_layout(); fig.savefig(filename); plt.close(fig)

def plot_decision_boundaries(weight_bias_list, X, y, titles=None, filename="decision_boundaries.png"):
    X = np.array(X); y = np.array(y).flatten()
    n_feat = X.shape[1]
    if n_feat > 2:
        const_cols = [j for j in range(n_feat) if np.all(X[:,j]==X[0,j])]
        if const_cols:
            X_red = np.delete(X, const_cols, axis=1)
            new_list = []
            for w,b in weight_bias_list:
                w = np.array(w).reshape(-1); b = float(b)
                for j in sorted(const_cols,reverse=True):
                    b += w[j]*X[0,j]
                    w = np.delete(w,j)
                new_list.append((w,b))
            weight_bias_list = new_list; X = X_red; n_feat = X_red.shape[1]
        if n_feat > 2:
            print("Cannot plot decision boundary for data with >2 features."); return
    if n_feat != 2:
        print("Cannot plot decision boundary: need 2 features."); return

    num = len(weight_bias_list)
    fig, axes = plt.subplots(1,num,figsize=(5*num,4))
    if num==1: axes=[axes]
    x_vals, y_vals = X[:,0], X[:,1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    xm = 0.1*(x_max-x_min) if x_max!=x_min else 1
    ym = 0.1*(y_max-y_min) if y_max!=y_min else 1
    x_range = (x_min-xm, x_max+xm)

    for idx,(w,b) in enumerate(weight_bias_list):
        w = np.array(w).reshape(-1); b=float(b)
        ax=axes[idx]
        y_i=y.astype(int)
        c0=X[y_i==0]; c1=X[y_i==1]
        l0="Class 0" if idx==0 else None
        l1="Class 1" if idx==0 else None
        ax.scatter(c0[:,0],c0[:,1],marker='o',label=l0)
        ax.scatter(c1[:,0],c1[:,1],marker='s',label=l1)
        if abs(w[1])<1e-6:
            if abs(w[0])<1e-6:
                line_x=[]; line_y=[]
            else:
                x_c=-b/w[0]
                line_x=[x_c,x_c]; line_y=[y_min-ym,y_max+ym]
        elif abs(w[0])<1e-6:
            y_c=-b/w[1]
            line_x=[x_range[0],x_range[1]]; line_y=[y_c,y_c]
        else:
            x_line = np.linspace(x_range[0],x_range[1],100)
            y_line = -(w[0]/w[1])*x_line - (b/w[1])
            line_x, line_y = x_line, y_line

        if line_x!=[]:
            ll="Decision Boundary" if idx==0 else None
            ax.plot(line_x,line_y,linestyle='--',label=ll)
        if titles and idx<len(titles): ax.set_title(titles[idx])
        else: ax.set_title(f"Boundary {idx+1}")
        ax.set_xlim(x_range)
        if line_x!=[]:
            y_min_line=min(min(y_vals), min(line_y))
            y_max_line=max(max(y_vals), max(line_y))
            ax.set_ylim(y_min_line-ym, y_max_line+ym)
        else:
            ax.set_ylim(y_min-ym, y_max+ym)
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        ax.grid(True)
        if idx==0: ax.legend(loc='best')
    plt.tight_layout(); fig.savefig(filename); plt.close(fig)
