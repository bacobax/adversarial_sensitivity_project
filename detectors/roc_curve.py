import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# --- Load CSV ---------------------------------------------------------------
df = pd.read_csv("results.csv")  # adjust path

global_handles = []
global_labels = []


def main():
    real_mask = df["folder"].str.contains("real", case=False)
    real_folders = df.loc[real_mask, "folder"].unique()
    
    if len(real_folders) != 1:
        raise ValueError(f"Expected exactly one 'real' folder, got: {real_folders}")
    
    real_folder = real_folders[0]
    
    # --- ROC per model ------------------------------------------------------------
    models = df["model"].unique()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for model in models:
        mdf = df[df["model"] == model]
        
        # labels: 0 = real, 1 = everything else
        y_true = (mdf["folder"] != real_folder).astype(int).to_numpy()
        y_score = mdf["confidence"].to_numpy()
        
        # need both classes present
        if len(np.unique(y_true)) < 2:
            print(f"Skipping {model}: only one class present.")
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        
        ax.plot(fpr, tpr, label=f"{model} (AUC={auc:.3f})", linewidth=1)
    
    # reference diagonal
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC inpainted vs real")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
