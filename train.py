import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def calculate_metrics(labels, predictions, probabilities):
    """
    Calculates Accuracy, AUC, PPV, NPV.

    Args:
        labels (np.array): True labels (0 or 1).
        predictions (np.array): Predicted labels (0 or 1, thresholded).
        probabilities (np.array): Predicted probabilities (output from sigmoid).

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    acc = accuracy_score(labels, predictions)
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError as e:
        print(f"Warning: Cannot compute AUC ({e}). Setting AUC to NaN.")
        auc = np.nan # Handle cases with only one class in labels or predictions

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0 # Negative Predictive Value

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall / True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate

    metrics = {
        'Accuracy': acc,
        'AUC': auc,
        'PPV': ppv,
        'NPV': npv,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    return metrics

def plot_roc_curves(results_dict, title='ROC Curves', save_path=None):
    """
    Plots ROC curves for multiple models.

    Args:
        results_dict (dict): Dictionary where keys are model names and values are
                             tuples of (true_labels, probabilities).
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, shows plot.
    """
    plt.figure(figsize=(8, 6))
    for model_name, (labels, probs) in results_dict.items():
        if probs is None or labels is None:
            print(f"Skipping ROC curve for {model_name} due to missing data.")
            continue
        try:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = roc_auc_score(labels, probs)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        except ValueError as e:
             print(f"Could not plot ROC for {model_name}: {e}")


    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)') # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()

# --- DeLong Test (Requires careful implementation or external package) ---
# Implementing DeLong's test for comparing AUCs is non-trivial.
# You might find implementations online or use statistical software (like R's pROC).
# Placeholder function:
def compare_auc_delong(labels, probs1, probs2):
    """Placeholder for DeLong's test comparing AUC of two models."""
    # Implementation requires calculating covariance matrices of AUC estimates.
    # See DeLong et al., Biometrics (1988) or look for Python implementations.
    print("DeLong test comparison not implemented in this placeholder.")
    # Example using a hypothetical library function:
    # try:
    #     from some_stats_library import delong_test
    #     p_value = delong_test(labels, probs1, probs2)
    #     return p_value
    # except ImportError:
    #     return np.nan
    return np.nan # Return NaN indicating not implemented