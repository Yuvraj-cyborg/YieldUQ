import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy import stats


def expected_calibration_error(uncertainties, errors, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, np.max(uncertainties) * 1.1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_uncertainty_in_bin = uncertainties[in_bin].mean()
            avg_error_in_bin = errors[in_bin].mean()
            ece += np.abs(avg_uncertainty_in_bin - avg_error_in_bin) * prop_in_bin
    
    return ece


def calibrate_uncertainty(uncertainties, errors):
    """Apply isotonic regression for calibration."""
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    calibrated_uncertainties = iso_reg.fit_transform(uncertainties, errors)
    return calibrated_uncertainties, iso_reg


def analyze_uncertainty(mc_preds, y_true, method='mc_dropout'):
    """Enhanced uncertainty analysis with calibration metrics."""
    if isinstance(mc_preds, tuple):
        # Heteroscedastic or ensemble output
        mean_pred, std_pred = mc_preds
        mean_pred = np.array(mean_pred)
        std_pred = np.array(std_pred)
    else:
        # MC Dropout output
        preds = np.array(mc_preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
    
    y_true = np.array(y_true)
    errors = np.abs(mean_pred - y_true)
    
    # Compute metrics
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    r2 = 1 - np.sum((y_true - mean_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    
    # Uncertainty-error correlation
    corr = np.corrcoef(std_pred, errors)[0, 1]
    
    # Expected Calibration Error
    ece = expected_calibration_error(std_pred, errors)
    
    # Calibrate uncertainties
    calibrated_std, calibrator = calibrate_uncertainty(std_pred, errors)
    calibrated_ece = expected_calibration_error(calibrated_std, errors)
    
    print(f"\n{'='*60}")
    print(f"UNCERTAINTY ANALYSIS ({method.upper()})")
    print(f"{'='*60}")
    print(f"Performance Metrics:")
    print(f"  R² Score:                    {r2:.4f}")
    print(f"  Mean Absolute Error (MAE):   {mae:.4f}")
    print(f"  Root Mean Squared Error:     {rmse:.4f}")
    print(f"\nUncertainty Metrics:")
    print(f"  Mean Uncertainty (σ):        {std_pred.mean():.4f}")
    print(f"  Uncertainty-Error Correlation: {corr:.4f}")
    print(f"  Expected Calibration Error:  {ece:.4f}")
    print(f"  Calibrated ECE:              {calibrated_ece:.4f}")
    print(f"  Improvement:                 {((ece - calibrated_ece) / ece * 100):.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'mean_pred': mean_pred,
        'std_pred': std_pred,
        'calibrated_std': calibrated_std,
        'errors': errors,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': corr,
        'ece': ece,
        'calibrated_ece': calibrated_ece,
        'calibrator': calibrator
    }


def compute_prediction_intervals(mean_pred, std_pred, confidence=0.95):
    """Compute prediction intervals with given confidence level."""
    z_score = stats.norm.ppf((1 + confidence) / 2)
    lower = mean_pred - z_score * std_pred
    upper = mean_pred + z_score * std_pred
    return lower, upper


def check_coverage(y_true, mean_pred, std_pred, confidence=0.95):
    """Check if empirical coverage matches theoretical confidence level."""
    lower, upper = compute_prediction_intervals(mean_pred, std_pred, confidence)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    print(f"Prediction Interval Coverage ({confidence*100:.0f}% confidence): {coverage*100:.2f}%")
    print(f"Expected: {confidence*100:.0f}%, Actual: {coverage*100:.2f}%")
    return coverage
