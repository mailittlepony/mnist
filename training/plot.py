#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import sys, pandas as pd
import matplotlib.pyplot as plt

def plot_curves(csv_path):
    df = pd.read_csv(csv_path)
    name = csv_path.split('/')[-2] if '/' in csv_path else csv_path
    fig, axes = plt.subplots(4, 1, figsize=(7,10), sharex=True)

    # Train + Val Loss
    if "train_loss" in df and "val_loss" in df:
        axes[0].plot(df["step"], df["train_loss"], label="train_loss")
        axes[0].plot(df["step"], df["val_loss"], label="val_loss")
    else:
        axes[0].plot(df["step"], df["loss"], label="train_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title(f"{name} – Loss")

    # Train + Val Accuracy
    if "train_acc" in df:
        axes[1].plot(df["step"], df["train_acc"], label="train_acc")
    if "val_acc" in df:
        axes[1].plot(df["step"], df["val_acc"], label="val_acc")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].set_title(f"{name} – Accuracy")

    # Generalization Gap
    if "train_acc" in df and "val_acc" in df:
        valid = df.dropna(subset=["val_acc"])
        gap = valid["train_acc"].values[:len(valid)] - valid["val_acc"].values
        axes[2].plot(valid["step"].values[:len(gap)], gap)
        axes[2].set_ylabel("Train–Val Gap (pp)")
        axes[2].set_title(f"{name} – Generalization Gap")

    # Learning Rate
    if "lr" in df:
        axes[3].plot(df["step"], df["lr"])
        axes[3].set_ylabel("Learning rate")
        axes[3].set_xlabel("Step")
        axes[3].set_title(f"{name} – LR Schedule")

    plt.tight_layout()
    plt.savefig(csv_path.replace("_trainlog.csv", "_fit_curves.png"), dpi=150)
    print(f"Saved {csv_path.replace('_trainlog.csv', '_fit_curves.png')}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        plot_curves(path)

