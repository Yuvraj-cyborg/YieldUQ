import numpy as np
import matplotlib.pyplot as plt

def analyze_uncertainty(mc_preds, y_true):
    preds = np.array(mc_preds)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    plt.figure(figsize=(6,6))
    plt.errorbar(y_true, mean_pred, yerr=std_pred, fmt='o', alpha=0.6)
    plt.xlabel("True Yield")
    plt.ylabel("Predicted Yield ± σ")
    plt.title("Uncertainty in Reaction Yield Prediction")
    plt.grid(True)
    plt.show()

    corr = np.corrcoef(std_pred, np.abs(mean_pred - y_true))[0,1]
    print(f"Correlation (uncertainty vs error): {corr:.3f}")
