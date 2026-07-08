"""Plots for comparing trained models."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_importance(model, feature_names, top_n=10, savepath=None):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.nlargest(top_n).plot(kind="barh", color="lightgreen")
    plt.title(f"Top {top_n} Factors Driving Price")
    plt.xlabel("Importance")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def plot_actual_vs_predicted(y_true, predictions_by_model, rolling_window=25, title="Actual vs. Predicted Pricing Trends", savepath=None):
    """`predictions_by_model` is a dict of {model_name: predicted_values} sharing the same
    ordering as `y_true`. Values are sorted by actual price and smoothed with a rolling
    mean so the comparison line chart is readable.
    """
    plot_df = pd.DataFrame({"Actual Price": y_true, **predictions_by_model})
    plot_df = plot_df.sort_values(by="Actual Price").reset_index(drop=True)
    rolling_df = plot_df.rolling(window=rolling_window, min_periods=1).mean()

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    plt.plot(rolling_df["Actual Price"], label="Actual Price ($)", color="black", linewidth=3, linestyle="--")
    for name in predictions_by_model:
        plt.plot(rolling_df[name], label=name, alpha=0.8, linewidth=2)

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Test Set Listings (Sorted from Lowest to Highest Value)", fontsize=11, labelpad=10)
    plt.ylabel("Price per Night ($)", fontsize=11, labelpad=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11, loc="upper left")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()
