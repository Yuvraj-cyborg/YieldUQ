import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
sns.set_palette("husl")


def visualize_uncertainty(y_true, mc_preds, save_path=None):
    """
    Visualize true vs predicted yields with uncertainty bars (σ from MC Dropout).
    Enhanced with color-coded uncertainty levels and metrics display.
    """
    # Ensure tensor → NumPy
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()

    mean_preds = np.mean(mc_preds, axis=0)
    std_preds = np.std(mc_preds, axis=0)
    
    # Calculate metrics
    mae = np.mean(np.abs(mean_preds - np.array(y_true)))
    rmse = np.sqrt(np.mean((mean_preds - np.array(y_true))**2))
    r2 = 1 - np.sum((mean_preds - np.array(y_true))**2) / np.sum((np.array(y_true) - np.mean(y_true))**2)
    correlation = np.corrcoef(std_preds, np.abs(mean_preds - np.array(y_true)))[0, 1]
    
    # Create figure with color-coded uncertainty
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Color points by uncertainty level
    scatter = ax.scatter(
        y_true,
        mean_preds,
        c=std_preds,
        cmap='YlOrRd',
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        zorder=3
    )
    
    # Add error bars
    ax.errorbar(
        y_true,
        mean_preds,
        yerr=std_preds,
        fmt='none',
        ecolor='gray',
        alpha=0.3,
        capsize=3,
        elinewidth=1.0,
        zorder=2
    )
    
    # Ideal prediction line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Prediction', alpha=0.7, zorder=1)
    
    # Confidence bands (±10% error)
    x_line = np.linspace(0, 1, 100)
    ax.fill_between(x_line, x_line - 0.1, x_line + 0.1, alpha=0.15, color='green', label='±10% Error Band')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Prediction Uncertainty (σ)', fontsize=12, weight='bold')
    
    # Labels and title
    ax.set_xlabel('True Yield', fontsize=14, weight='bold')
    ax.set_ylabel('Predicted Yield', fontsize=14, weight='bold')
    ax.set_title('Reaction Yield Prediction with Uncertainty Quantification', 
                 fontsize=15, weight='bold', pad=20)
    
    # Metrics box
    textstr = f'Metrics:\n' \
              f'R² = {r2:.4f}\n' \
              f'MAE = {mae:.4f}\n' \
              f'RMSE = {rmse:.4f}\n' \
              f'σ-Error Corr. = {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Grid and legend
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()

    # ✅ Optionally save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved high-quality plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_error_distribution(y_true, mc_preds, save_path=None):
    """
    Plot scatter of predicted uncertainty (σ) vs. absolute error with regression line.
    Shows how uncertainty correlates with model error - key for uncertainty calibration.
    """
    # ✅ Ensure tensor → NumPy
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()

    mean_preds = np.mean(mc_preds, axis=0)
    std_preds = np.std(mc_preds, axis=0)
    errors = np.abs(mean_preds - np.array(y_true))
    
    # Calculate correlation
    correlation = np.corrcoef(std_preds, errors)[0, 1]
    
    # Linear regression for trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(std_preds, errors)
    line_x = np.linspace(std_preds.min(), std_preds.max(), 100)
    line_y = slope * line_x + intercept

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    # Density scatter plot
    scatter = ax.scatter(
        std_preds, 
        errors, 
        alpha=0.6, 
        s=50, 
        c=errors,
        cmap='coolwarm',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Regression line
    ax.plot(line_x, line_y, 'r--', lw=2.5, 
            label=f'Linear Fit (R={correlation:.3f})', alpha=0.8)
    
    # Ideal calibration line (uncertainty = error)
    max_val = max(std_preds.max(), errors.max())
    ax.plot([0, max_val], [0, max_val], 'k:', lw=2, 
            label='Perfect Calibration', alpha=0.6)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Absolute Error', fontsize=12, weight='bold')
    
    # Labels and title
    ax.set_xlabel("Predicted Uncertainty (σ)", fontsize=14, weight='bold')
    ax.set_ylabel("Absolute Error |y_pred - y_true|", fontsize=14, weight='bold')
    ax.set_title("Uncertainty-Error Correlation Analysis", fontsize=15, weight='bold', pad=20)
    
    # Statistics box
    textstr = f'Correlation: {correlation:.4f}\n' \
              f'Slope: {slope:.4f}\n' \
              f'p-value: {p_value:.2e}\n' \
              f'Mean σ: {np.mean(std_preds):.4f}\n' \
              f'Mean Error: {np.mean(errors):.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()

    # ✅ Optionally save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"📊 Saved error distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_calibration(y_true, mc_preds, n_bins=10, save_path=None):
    """
    Reliability diagram to check calibration between uncertainty and actual error.
    Well-calibrated model: points should lie on diagonal (predicted σ = actual error).
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
    std_unc = [np.std(std_sorted[b]) for b in bins]
    std_err = [np.std(errors_sorted[b]) for b in bins]
    
    # Calculate calibration error (ECE - Expected Calibration Error)
    ece = np.mean(np.abs(np.array(avg_unc) - np.array(avg_err)))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    
    # Plot calibration curve with error bars
    ax.errorbar(avg_unc, avg_err, xerr=std_unc, yerr=std_err,
                fmt='o-', color='tab:green', linewidth=2.5, 
                markersize=8, capsize=5, capthick=2,
                label='Calibration Curve', alpha=0.8)
    
    # Perfect calibration line
    max_val = max(max(avg_unc), max(avg_err))
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2.5, 
            label='Perfect Calibration', alpha=0.7)
    
    # Shaded region for acceptable calibration (±20% band)
    x_line = np.linspace(0, max_val, 100)
    ax.fill_between(x_line, 0.8*x_line, 1.2*x_line, 
                     alpha=0.15, color='red', label='±20% Band')
    
    # Labels and title
    ax.set_xlabel("Predicted Uncertainty (σ)", fontsize=14, weight='bold')
    ax.set_ylabel("Actual Mean Error", fontsize=14, weight='bold')
    ax.set_title("Uncertainty Calibration - Reliability Diagram", 
                 fontsize=15, weight='bold', pad=20)
    
    # Statistics box
    textstr = f'Expected Calibration Error:\n' \
              f'ECE = {ece:.4f}\n\n' \
              f'Number of bins: {n_bins}\n' \
              f'Samples per bin: ~{len(errors)//n_bins}'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Interpretation guide
    interpretation = 'Good calibration:\nPoints near diagonal'
    props2 = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
    ax.text(0.70, 0.25, interpretation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props2)
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.set_aspect('equal')
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"📈 Saved calibration plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_mc_distribution(mc_preds, sample_indices=None, n_samples=10, save_path=None):
    """
    Visualize the distribution of MC Dropout predictions for selected samples.
    Shows how dropout affects predictions across multiple forward passes.
    
    Args:
        mc_preds: Array of shape (n_mc_samples, n_test_samples)
        sample_indices: Which test samples to visualize (default: first n_samples)
        n_samples: Number of samples to show
        save_path: Path to save the plot
    """
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()
    
    if sample_indices is None:
        sample_indices = list(range(min(n_samples, mc_preds.shape[1])))
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 8), dpi=150)
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(sample_indices[:10]):
        ax = axes[idx]
        predictions = mc_preds[:, sample_idx]
        
        # Histogram with KDE
        ax.hist(predictions, bins=15, alpha=0.6, color='skyblue', 
                edgecolor='black', density=True, label='MC Samples')
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        if len(predictions) > 1:
            kde = gaussian_kde(predictions)
            x_range = np.linspace(predictions.min(), predictions.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        
        # Mean and std lines
        mean_val = np.mean(predictions)
        std_val = np.std(predictions)
        ax.axvline(mean_val, color='green', linestyle='--', lw=2, label=f'μ={mean_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', lw=1.5, alpha=0.7)
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', lw=1.5, alpha=0.7, label=f'σ={std_val:.3f}')
        
        ax.set_title(f'Sample {sample_idx}', fontsize=10, weight='bold')
        ax.set_xlabel('Predicted Yield', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    fig.suptitle('Monte Carlo Dropout: Prediction Distributions for Test Samples', 
                 fontsize=16, weight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"📊 Saved MC distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_residuals(y_true, mc_preds, save_path=None):
    """
    Residual plot to diagnose prediction bias and heteroscedasticity.
    
    Args:
        y_true: True yield values
        mc_preds: MC dropout predictions
        save_path: Path to save the plot
    """
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()
    
    mean_preds = np.mean(mc_preds, axis=0)
    std_preds = np.std(mc_preds, axis=0)
    residuals = mean_preds - np.array(y_true)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # Left: Residuals vs Predicted
    ax1 = axes[0]
    scatter1 = ax1.scatter(mean_preds, residuals, c=std_preds, cmap='plasma',
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.axhline(0, color='red', linestyle='--', lw=2, label='Zero Residual')
    ax1.axhline(0.1, color='orange', linestyle=':', lw=1.5, alpha=0.6)
    ax1.axhline(-0.1, color='orange', linestyle=':', lw=1.5, alpha=0.6, label='±0.1 Band')
    
    cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.02)
    cbar1.set_label('Uncertainty (σ)', fontsize=11, weight='bold')
    
    ax1.set_xlabel('Predicted Yield', fontsize=13, weight='bold')
    ax1.set_ylabel('Residual (Predicted - True)', fontsize=13, weight='bold')
    ax1.set_title('Residual vs Predicted', fontsize=14, weight='bold')
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right: Residual distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, alpha=0.7, color='steelblue', 
             edgecolor='black', density=True)
    
    # Fit normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 
             'r-', lw=2.5, label=f'Normal Fit\nμ={mu:.4f}\nσ={sigma:.4f}')
    
    ax2.axvline(0, color='green', linestyle='--', lw=2, label='Zero Mean')
    ax2.set_xlabel('Residual', fontsize=13, weight='bold')
    ax2.set_ylabel('Density', fontsize=13, weight='bold')
    ax2.set_title('Residual Distribution', fontsize=14, weight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    fig.suptitle('Residual Analysis for Model Diagnostics', 
                 fontsize=16, weight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"📊 Saved residual plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comprehensive_report(y_true, mc_preds, save_dir='plots'):
    """
    Generate a comprehensive visualization report with all plots.
    
    Args:
        y_true: True yield values
        mc_preds: MC dropout predictions
        save_dir: Directory to save all plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("📊 Generating Comprehensive Visualization Report")
    print("="*60)
    
    # 1. Uncertainty plot
    print("\n[1/6] Creating uncertainty visualization...")
    visualize_uncertainty(y_true, mc_preds, 
                         save_path=os.path.join(save_dir, 'yield_uncertainty.png'))
    
    # 2. Error distribution
    print("[2/6] Creating error distribution plot...")
    visualize_error_distribution(y_true, mc_preds,
                                save_path=os.path.join(save_dir, 'error_distribution.png'))
    
    # 3. Calibration
    print("[3/6] Creating calibration plot...")
    visualize_calibration(y_true, mc_preds,
                         save_path=os.path.join(save_dir, 'calibration.png'))
    
    # 4. MC Distribution
    print("[4/6] Creating MC distribution plot...")
    visualize_mc_distribution(mc_preds,
                            save_path=os.path.join(save_dir, 'mc_distributions.png'))
    
    # 5. Residuals
    print("[5/6] Creating residual analysis plot...")
    visualize_residuals(y_true, mc_preds,
                       save_path=os.path.join(save_dir, 'residuals.png'))
    
    # 6. Summary statistics
    print("[6/6] Saving summary statistics...")
    save_summary_statistics(y_true, mc_preds, 
                          save_path=os.path.join(save_dir, 'summary_stats.txt'))
    
    print("\n" + "="*60)
    print(f"✅ Report complete! All plots saved to: {save_dir}/")
    print("="*60 + "\n")


def save_summary_statistics(y_true, mc_preds, save_path='plots/summary_stats.txt'):
    """
    Save comprehensive summary statistics to a text file.
    """
    if isinstance(mc_preds, torch.Tensor):
        mc_preds = mc_preds.detach().cpu().numpy()
    
    mean_preds = np.mean(mc_preds, axis=0)
    std_preds = np.std(mc_preds, axis=0)
    errors = np.abs(mean_preds - np.array(y_true))
    residuals = mean_preds - np.array(y_true)
    
    # Calculate metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((np.array(y_true) - np.mean(y_true))**2)
    correlation = np.corrcoef(std_preds, errors)[0, 1]
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("YIELDUQ-GNN MODEL PERFORMANCE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("PREDICTION METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  R² Score:                  {r2:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"  Root Mean Squared Error:   {rmse:.6f}\n")
        f.write(f"  Mean Prediction:           {np.mean(mean_preds):.6f}\n")
        f.write(f"  Std Prediction:            {np.std(mean_preds):.6f}\n\n")
        
        f.write("UNCERTAINTY METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Mean Uncertainty (σ):      {np.mean(std_preds):.6f}\n")
        f.write(f"  Std Uncertainty:           {np.std(std_preds):.6f}\n")
        f.write(f"  Min Uncertainty:           {np.min(std_preds):.6f}\n")
        f.write(f"  Max Uncertainty:           {np.max(std_preds):.6f}\n")
        f.write(f"  σ-Error Correlation:       {correlation:.6f}\n\n")
        
        f.write("ERROR DISTRIBUTION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Mean Error:                {np.mean(residuals):.6f}\n")
        f.write(f"  Std Error:                 {np.std(residuals):.6f}\n")
        f.write(f"  Max Absolute Error:        {np.max(errors):.6f}\n")
        f.write(f"  Median Absolute Error:     {np.median(errors):.6f}\n")
        f.write(f"  90th Percentile Error:     {np.percentile(errors, 90):.6f}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Number of samples:         {len(y_true)}\n")
        f.write(f"  True yield range:          [{min(y_true):.3f}, {max(y_true):.3f}]\n")
        f.write(f"  True yield mean:           {np.mean(y_true):.6f}\n")
        f.write(f"  True yield std:            {np.std(y_true):.6f}\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"📄 Summary statistics saved to {save_path}")
