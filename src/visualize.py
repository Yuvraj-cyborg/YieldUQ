import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
sns.set_palette("husl")


def visualize_all_results(y_true, mc_preds, title_suffix=""):
    """
    Visualize model performance with 4 essential diagnostic plots.
    
    Args:
        y_true: True yield values (list or array)
        mc_preds: MC dropout predictions of shape (n_mc_samples, n_test_samples)
                  OR tuple of (mean_preds, std_preds) for heteroscedastic
        title_suffix: Optional suffix to add to plot titles
    """
    # Handle both MC Dropout format and heteroscedastic format
    if isinstance(mc_preds, tuple):
        mean_preds, std_preds = mc_preds
        mean_preds = np.array(mean_preds)
        std_preds = np.array(std_preds)
    else:
        if isinstance(mc_preds, torch.Tensor):
            mc_preds = mc_preds.detach().cpu().numpy()
        mean_preds = np.mean(mc_preds, axis=0)
        std_preds = np.std(mc_preds, axis=0)
    
    errors = np.abs(mean_preds - np.array(y_true))
    residuals = mean_preds - np.array(y_true)
    
    # Calculate key metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((np.array(y_true) - np.mean(y_true))**2)
    correlation = np.corrcoef(std_preds, errors)[0, 1]
    
    # Create 2x2 subplot figure with 4 essential diagnostic plots
    fig = plt.figure(figsize=(14, 11))
    
    # PLOT 1: Predicted vs True Yield (with uncertainty color-coding)
    ax1 = plt.subplot(2, 2, 1)
    scatter1 = ax1.scatter(y_true, mean_preds, c=std_preds, cmap='RdYlGn_r',
                           s=80, alpha=0.75, edgecolors='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'k--', lw=2.5, label='Perfect Prediction', alpha=0.7)
    x_line = np.linspace(0, 1, 100)
    ax1.fill_between(x_line, x_line - 0.1, x_line + 0.1, alpha=0.12, color='gray', label='±10% Error')
    plt.colorbar(scatter1, ax=ax1, pad=0.02, label='Uncertainty')
    ax1.set_xlabel('True Yield', fontsize=11)
    ax1.set_ylabel('Predicted Yield', fontsize=11)
    ax1.set_title(f'Prediction Accuracy {title_suffix}', fontsize=12, weight='bold')
    textstr1 = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
    props1 = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props1, family='monospace')
    ax1.grid(alpha=0.25, linestyle=':', linewidth=0.8)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    
    # PLOT 2: Uncertainty-Error Correlation (diagnostic for calibration quality)
    ax2 = plt.subplot(2, 2, 2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(std_preds, errors)
    line_x = np.linspace(std_preds.min(), std_preds.max(), 100)
    line_y = slope * line_x + intercept
    scatter2 = ax2.scatter(std_preds, errors, alpha=0.65, s=70, c=errors,
                           cmap='viridis', edgecolors='black', linewidth=0.5)
    ax2.plot(line_x, line_y, 'r--', lw=2.5, label=f'Linear Fit (r={correlation:.3f})', alpha=0.9)
    max_val = max(std_preds.max(), errors.max())
    ax2.plot([0, max_val], [0, max_val], 'k:', lw=2.5, label='Ideal Calibration', alpha=0.6)
    plt.colorbar(scatter2, ax=ax2, pad=0.02, label='Error')
    ax2.set_xlabel("Predicted Uncertainty", fontsize=11)
    ax2.set_ylabel("Actual Error", fontsize=11)
    ax2.set_title(f"Uncertainty Calibration {title_suffix}", fontsize=12, weight='bold')
    textstr2 = f'Correlation: {correlation:.4f}\nSlope: {slope:.3f}\nIntercept: {intercept:.4f}'
    props2 = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props2, family='monospace')
    ax2.grid(alpha=0.25, linestyle=':', linewidth=0.8)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # PLOT 3: Residual Distribution (check for systematic biases)
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(residuals, bins=40, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)
    ax3.axvline(0, color='red', linestyle='--', lw=2.5, label='Zero Residual', alpha=0.9)
    mu_res = np.mean(residuals)
    sigma_res = np.std(residuals)
    ax3.axvline(mu_res, color='orange', linestyle=':', lw=2, label=f'Mean={mu_res:.3f}', alpha=0.8)
    
    # Overlay normal distribution
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu_res, sigma_res) * len(residuals) * (residuals.max() - residuals.min()) / 40
    ax3.plot(x_norm, y_norm, 'k-', lw=2, label='Normal Fit', alpha=0.7)
    
    ax3.set_xlabel('Residual (Predicted - True)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Residual Distribution {title_suffix}', fontsize=12, weight='bold')
    textstr3 = f'Mean: {mu_res:.4f}\nStd Dev: {sigma_res:.4f}\nSkewness: {stats.skew(residuals):.3f}'
    props3 = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props3, family='monospace')
    ax3.grid(alpha=0.25, linestyle=':', linewidth=0.8, axis='y')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # PLOT 4: Binned Calibration Curve (expected calibration error visualization)
    ax4 = plt.subplot(2, 2, 4)
    n_bins = 10
    sort_idx = np.argsort(std_preds)
    std_sorted = std_preds[sort_idx]
    errors_sorted = errors[sort_idx]
    bins = np.array_split(np.arange(len(errors_sorted)), n_bins)
    avg_unc = [np.mean(std_sorted[b]) for b in bins]
    avg_err = [np.mean(errors_sorted[b]) for b in bins]
    std_unc = [np.std(std_sorted[b]) for b in bins]
    std_err = [np.std(errors_sorted[b]) for b in bins]
    ece = np.mean(np.abs(np.array(avg_unc) - np.array(avg_err)))
    
    ax4.errorbar(avg_unc, avg_err, xerr=std_unc, yerr=std_err,
                 fmt='o-', color='darkgreen', linewidth=2.5, markersize=9,
                 capsize=5, capthick=2, label='Binned Data', alpha=0.85)
    max_val4 = max(max(avg_unc), max(avg_err)) if avg_unc and avg_err else 1
    ax4.plot([0, max_val4], [0, max_val4], 'r--', lw=2.5, label='Perfect Calibration', alpha=0.8)
    x_line4 = np.linspace(0, max_val4, 100)
    ax4.fill_between(x_line4, 0.85*x_line4, 1.15*x_line4, alpha=0.15, color='red', label='±15% Band')
    ax4.set_xlabel("Predicted Uncertainty", fontsize=11)
    ax4.set_ylabel("Observed Error", fontsize=11)
    ax4.set_title(f"Expected Calibration Error {title_suffix}", fontsize=12, weight='bold')
    textstr4 = f'ECE = {ece:.5f}\nBins: {n_bins}'
    props4 = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax4.text(0.05, 0.95, textstr4, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=props4, family='monospace')
    ax4.grid(alpha=0.25, linestyle=':', linewidth=0.8)
    ax4.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax4.set_aspect('equal')
    
    plt.tight_layout(pad=2.5)
    
    # Print summary to console
    print("\n" + "="*80)
    print(" MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"  R² Score:                    {r2:.6f}")
    print(f"  Mean Absolute Error:         {mae:.6f}")
    print(f"  Root Mean Squared Error:     {rmse:.6f}")
    print(f"  Mean Uncertainty:            {np.mean(std_preds):.6f}")
    print(f"  Uncertainty-Error Corr:      {correlation:.6f}")
    print(f"  Expected Calibration Error:  {ece:.6f}")
    print(f"  Residual Mean:               {np.mean(residuals):.6f}")
    print(f"  Residual Std Dev:            {np.std(residuals):.6f}")
    print("="*80 + "\n")
    
    plt.show()


def compare_uncertainty_methods(y_true, results_list, method_names):
    """
    Compare different uncertainty quantification methods in a compact view.
    
    Args:
        y_true: True yield values
        results_list: List of result dicts from analyze_uncertainty
        method_names: List of method names
    """
    n_methods = len(results_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5.5))
    
    if n_methods == 1:
        axes = [axes]
    
    for i, (results, method_name) in enumerate(zip(results_list, method_names)):
        mean_pred = results['mean_pred']
        calibrated_std = results['calibrated_std']
        
        ax = axes[i]
        scatter = ax.scatter(y_true, mean_pred, c=calibrated_std, cmap='RdYlGn_r',
                           s=70, alpha=0.75, edgecolors='black', linewidth=0.5)
        ax.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Calibrated Uncertainty')
        ax.set_xlabel('True Yield', fontsize=11)
        ax.set_ylabel('Predicted Yield', fontsize=11)
        ax.set_title(f'{method_name}', fontsize=12, weight='bold')
        
        textstr = (f"R² = {results['r2']:.4f}\n"
                  f"MAE = {results['mae']:.4f}\n"
                  f"Corr = {results['correlation']:.3f}\n"
                  f"ECE = {results['calibrated_ece']:.5f}")
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor='black', linewidth=1.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', bbox=props, family='monospace')
        ax.grid(alpha=0.25, linestyle=':', linewidth=0.8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
    
    plt.tight_layout(pad=2.0)
    plt.show()
    
    # Print comparison table
    print("\n" + "="*110)
    print(" UNCERTAINTY METHOD COMPARISON")
    print("="*110)
    print(f"{'Method':<22} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'Correlation':<13} {'ECE (cal.)':<12}")
    print("-"*110)
    for results, method_name in zip(results_list, method_names):
        print(f"{method_name:<22} {results['r2']:<10.4f} {results['mae']:<10.4f} "
              f"{results['rmse']:<10.4f} {results['correlation']:<13.4f} "
              f"{results['calibrated_ece']:<12.5f}")
    print("="*110 + "\n")
