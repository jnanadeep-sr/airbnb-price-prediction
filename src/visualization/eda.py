"""Exploratory-analysis plots for the cleaned listings dataframe."""
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import TARGET_COLUMN


def plot_price_percentiles(df, target_column=TARGET_COLUMN, savepath=None):
    percentile_levels = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    percentile_labels = ["10th", "25th", "50th\n(Median)", "75th", "90th", "95th", "99th"]
    values = df[target_column].quantile(percentile_levels).tolist()

    bars = plt.bar(percentile_labels, values, color="#1f77b4", edgecolor="black")
    max_val = max(values)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (max_val * 0.015), f"${yval:,.2f}", ha="center", va="bottom", fontsize=9)

    plt.title("Price Distribution by Percentile", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Percentile Tier", fontsize=10, labelpad=10)
    plt.ylabel("Price per Night ($)", fontsize=10, labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, max_val * 1.15)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def plot_price_histogram(df, target_column=TARGET_COLUMN, price_cap=2500, bins=50, savepath=None):
    df_capped = df[df[target_column] <= price_cap]
    plt.hist(df_capped[target_column], bins=bins, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.title(f"Number of Listings by Price Bucket (Capped at ${price_cap:,})", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Price per Night ($)", fontsize=10, labelpad=10)
    plt.ylabel("Number of Listings (Count)", fontsize=10, labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def plot_numeric_vs_price_grid(df, numeric_features, target_column=TARGET_COLUMN, savepath=None):
    num_cols = 3
    num_rows = math.ceil(len(numeric_features) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 4))
    axes = axes.flatten()

    i = 0
    for i, feature in enumerate(numeric_features):
        if feature in df.columns:
            axes[i].scatter(df[feature], df[target_column], alpha=0.2, color="#1f77b4", edgecolors="none", s=12)
            axes[i].set_title(f"{feature} vs Price", fontsize=11, fontweight="bold", pad=8)
            axes[i].set_xlabel(feature, fontsize=9, labelpad=4)
            axes[i].set_ylabel("Price ($)", fontsize=9, labelpad=4)
            axes[i].grid(True, linestyle="--", alpha=0.5)
            axes[i].tick_params(axis="both", labelsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def plot_categorical_boxplots(df, categorical_features, target_column=TARGET_COLUMN, savepath=None):
    n = len(categorical_features)
    n_cols = 2
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 6))
    axes = axes.flatten()

    for i, col in enumerate(categorical_features):
        order = df.groupby(col)[target_column].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=col, y=target_column, ax=axes[i], order=order, hue=col, legend=False, palette="Set2")
        axes[i].set_title(f"Price Distribution by {col}", fontsize=12, fontweight="bold", pad=10)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Price per Night ($)", fontsize=10)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
        if len(order) > 4:
            axes[i].tick_params(axis="x", rotation=45)

    for j in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def plot_correlation_heatmap(df, target_column=TARGET_COLUMN, full_matrix=False, savepath=None):
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()

    if full_matrix:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.title("Full Correlation Matrix Heatmap", fontsize=16, fontweight="bold", pad=20)
    else:
        price_corr = corr_matrix[[target_column]].drop(labels=[target_column]).sort_values(by=target_column, ascending=False)
        plt.figure(figsize=(6, 10))
        sns.heatmap(price_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label": "Correlation Coefficient"}, linewidths=0.5)
        plt.title("Feature Correlation with Target Price", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()
