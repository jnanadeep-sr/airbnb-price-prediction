"""Regression evaluation metrics shared across all models."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(name, y_true, y_pred):
    metrics = regression_metrics(y_true, y_pred)
    print(f"--- {name} Performance ---")
    print(f"Mean Absolute Error (MAE):      ${metrics['mae']:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${metrics['rmse']:.2f}")
    print(f"R^2 Score (Variance Explained):  {metrics['r2']*100:.2f}%\n")
    return metrics


def print_metrics_log_target(name, y_true_log, y_pred_log):
    """For models trained on log1p(price): reverse the transform for dollar-denominated
    MAE/RMSE, but compute R^2 in log-space (matches how the model was optimized).
    """
    y_true_actual = np.expm1(y_true_log)
    y_pred_actual = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    r2 = r2_score(y_true_log, y_pred_log)

    print(f"{name:<30} | R^2 Score: {r2*100:.2f}% | MAE: ${mae:.2f} | RMSE: ${rmse:.2f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}
