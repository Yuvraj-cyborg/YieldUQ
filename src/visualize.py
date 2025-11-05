import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_uncertainty(y_true, mc_preds, save_path=None):
    """
    Visualize true vs predicted yields with uncertainty bars (σ from MC Dropout).
    """
    # Ensure tensor → NumPy
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()

    mean_preds = np.mean(mc_preds, axis=0)
    std_preds = np.std(mc_preds, axis=0)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.errorbar(
        y_true,
        mean_preds,
        yerr=std_preds,
        fmt='o',
        alpha=0.6,
        markersize=4,
        ecolor='tab:blue',
        capsize=2,
        elinewidth=0.8,
    )
    plt.plot([0, 1], [0, 1], 'r--', lw=1.5, label='Ideal')
    plt.title("Reaction Yield Prediction with Uncertainty", fontsize=13)
    plt.xlabel("True Yield", fontsize=11)
    plt.ylabel("Predicted Yield ± σ", fontsize=11)
    plt.grid(alpha=0.25)
    plt.legend()

    # ✅ Optionally save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"✅ Saved high-quality plot to {save_path}")
    else:
        plt.show()


def visualize_error_distribution(y_true, mc_preds, save_path=None):
    """
    Plot scatter of predicted uncertainty (σ) vs. absolute error.
    Shows how uncertainty correlates with model error.
    """
    # ✅ Ensure tensor → NumPy
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()

    mean_preds = np.mean(mc_preds, axis=0)
    std_preds = np.std(mc_preds, axis=0)
    errors = np.abs(mean_preds - np.array(y_true))

    plt.figure(figsize=(7, 5), dpi=150)
    plt.scatter(std_preds, errors, alpha=0.6, s=20, c='tab:purple')
    plt.xlabel("Predicted Uncertainty (σ)", fontsize=11)
    plt.ylabel("Absolute Error |y_pred - y_true|", fontsize=11)
    plt.title("Uncertainty vs. Prediction Error", fontsize=12)
    plt.grid(alpha=0.3)

    # ✅ Optionally save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"📊 Saved error distribution plot to {save_path}")
    else:
        plt.show()


def visualize_calibration(y_true, mc_preds, n_bins=10, save_path=None):
    """
    Optional: Reliability diagram to check calibration between uncertainty and actual error.
    """
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()

    mean_preds = np.mean(mc_preds, axis=0)
    std_preds = np.std(mc_preds, axis=0)
    errors = np.abs(mean_preds - np.array(y_true))

    # Sort by predicted uncertainty
    sort_idx = np.argsort(std_preds)
    std_sorted = std_preds[sort_idx]
    errors_sorted = errors[sort_idx]

    # Bin by uncertainty
    bins = np.array_split(np.arange(len(errors_sorted)), n_bins)
    avg_unc = [np.mean(std_sorted[b]) for b in bins]
    avg_err = [np.mean(errors_sorted[b]) for b in bins]

    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(avg_unc, avg_err, 'o-', color='tab:green', label='Calibration Curve')
    plt.plot([0, max(avg_unc)], [0, max(avg_unc)], 'r--', lw=1, label='Perfect Calibration')
    plt.title("Uncertainty Calibration (Reliability Diagram)", fontsize=12)
    plt.xlabel("Predicted Uncertainty (σ)", fontsize=11)
    plt.ylabel("Actual Mean Error", fontsize=11)
    plt.grid(alpha=0.3)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"📈 Saved calibration plot to {save_path}")
    else:
        plt.show()
