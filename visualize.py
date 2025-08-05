"""
Functions to save results and plot evaluation metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_all_outputs(results: list, method: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    df = pd.DataFrame(results)
    base_name = method.lower()
    
    out_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_file = os.path.join(out_dir, f"{base_name}_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV → {csv_file}")

    # F1 vs Energy (Pareto)
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="energy", y="f1", s=80)
    plt.title(f"{method} – Energy vs Macro-F1")
    plt.xlabel("Energy (kWh)")
    plt.ylabel("Macro-F1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base_name}_results.png"), dpi=200)

    # F1 vs Learning Rate
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="lr", y="f1")
    plt.title(f"{method} – F1 vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Macro-F1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base_name}_f1_vs_lr.png"), dpi=200)

    # F1 vs Batch Size
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="bs", y="f1")
    plt.title(f"{method} – F1 vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Macro-F1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base_name}_f1_vs_bs.png"), dpi=200)

    # F1 vs Epochs
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="ep", y="f1")
    plt.title(f"{method} – F1 vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Macro-F1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base_name}_f1_vs_ep.png"), dpi=200)

    print(f"✅ Saved all outputs for {method} in {out_dir}")
