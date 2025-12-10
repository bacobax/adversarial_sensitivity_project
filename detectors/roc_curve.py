from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# --- Load CSV ---------------------------------------------------------------
# read multiple CSVs
df = pd.concat([pd.read_csv(f) for f in [
    "out/results_wr.csv",
    "out/results_fgsm.csv",
    "out/results_pgd.csv",
    "out/results_deepfool.csv",
    # "out/results_all.csv",
]])

global_handles = []
global_labels = []


def main():
    df["category"] = df["folder"].str.split("/").str[-1]
    df["attack"] = df["folder"].str.split("/").str[-2]
    
    models = df["model"].unique()
    attacks = df["attack"].unique()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for model, attack in product(models, attacks):
        mdf = df[(df["model"] == model) & (df["attack"] == attack)]
        
        # labels: 0 = real, 1 = everything else
        y_true = (mdf["category"] != 'real').astype(int).to_numpy()
        y_score = mdf["confidence"].to_numpy()
        
        # need both classes present
        if len(np.unique(y_true)) < 2:
            print(f"Skipping {model}: only one class present.")
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        
        ax.plot(fpr, tpr, label=f"{model} {attack} (AUC={auc:.3f})", linewidth=1)
    
    # reference diagonal
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="gray")
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC inpainted vs real")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.tight_layout()
    plt.show()
    
    # plot average confidence across attacks, models and categories
    fig, ax = plt.subplots(figsize=(6, 6))
    
    
    

if __name__ == "__main__":
    main()
