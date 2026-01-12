from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# --- Load CSV ---------------------------------------------------------------
# read multiple CSVs
df = pd.concat([pd.read_csv(f) for f in [
    "outputs/results.csv",
    # "out/results_all.csv",
]])

global_handles = []
global_labels = []


def main():
    fig, ax = plt.subplots(figsize=(6, 6))
    
    model = 'AnomalyOV'
    attack = 'orig'
    
    mdf = df[(df["detector"] == model) & (df["attack"] == attack)]
    
    # labels: 0 = real, 1 = everything else
    y_true = (mdf["category"] != 'real').astype(int).to_numpy()
    y_score = mdf["logit"].to_numpy()
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    ax.plot(fpr, tpr, label=f"{model} (AUC={auc:.3f})", linewidth=1, c='tab:pink')

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
    # plt.savefig('results_aov.png')
    plt.show()
    
    
    
    

if __name__ == "__main__":
    # real = cv2.imread('datasets/b-free/real')
    samecat = cv2.imread('datasets/b-free/samecat/000000000127.png')
    diffcat = cv2.imread('datasets/b-free/diffcat/000000000127.png')
    mask = cv2.imread('datasets/b-free/mask/000000000127.png', cv2.IMREAD_GRAYSCALE)
    bbox = cv2.imread('datasets/b-free/bbox/000000000127.png', cv2.IMREAD_GRAYSCALE)
    
    contours_same = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_diff = cv2.findContours(bbox, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    samecat = cv2.drawContours(samecat, contours_same, -1, [0,0,255], 3)
    diffcat = cv2.drawContours(diffcat, contours_diff, -1, [0,0,255], 3)
    
    cv2.imshow('samecat', samecat)
    cv2.imshow('diffcat', diffcat)
    cv2.waitKey(0)
    exit(0)
    main()
