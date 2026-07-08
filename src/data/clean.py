"""Cleaning steps applied to the raw listings data before feature engineering."""
import os

import pandas as pd

from src.config import TARGET_COLUMN


def clean_missing_target(input_filename, output_filename, target_column=TARGET_COLUMN):
    """
    Reads a CSV file, removes rows where the target column is missing,
    and saves the cleaned data to a new CSV file.
    """
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found in this directory.")
        return

    print(f"Reading {input_filename}...")
    df = pd.read_csv(input_filename)

    total_rows_before = len(df)
    missing_count = df[target_column].isna().sum()

    if missing_count == 0:
        print(f"No missing values found in the '{target_column}' column. No rows removed.")
        df.to_csv(output_filename, index=False)
        print(f"File saved to {output_filename}")
        return

    df_cleaned = df.dropna(subset=[target_column])
    total_rows_after = len(df_cleaned)

    df_cleaned.to_csv(output_filename, index=False)

    print("\n--- Processing Summary ---")
    print(f"Initial row count:          {total_rows_before}")
    print(f"Rows with missing '{target_column}': {missing_count} (Removed)")
    print(f"Remaining row count:        {total_rows_after}")
    print(f"Cleaned data saved to:      {output_filename}\n")


def clean_price_column(df, column=TARGET_COLUMN):
    """Convert a currency-formatted price column (e.g. "$1,200.00") to float."""
    df = df.copy()
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=[column])


def clean_numerical_features(df, drop_threshold=0.90):
    """Drop sparsely-populated numerical columns and median-impute the rest.
    The 'id' column is always preserved and never imputed.
    """
    df_cleaned = df.copy()

    missing_pct = df_cleaned.isna().mean()
    cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()

    if "id" in cols_to_drop:
        cols_to_drop.remove("id")

    print(f"Dropping columns with >{drop_threshold*100}% missing values:")
    for col in cols_to_drop:
        print(f" - {col}")

    df_cleaned = df_cleaned.drop(columns=cols_to_drop)

    num_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns

    imputed_count = 0
    for col in num_cols:
        if col == "id":
            continue

        if df_cleaned[col].isna().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
            imputed_count += 1

    print(f"\nImputed {imputed_count} numerical columns using their median value.")
    return df_cleaned


def remove_price_outliers_dynamic(df, min_price_threshold=10, percentile_cutoff=0.99, target_column=TARGET_COLUMN):
    """Dynamically identify and remove extreme price outliers using data percentiles."""
    df_cleaned = df.copy()
    initial_count = len(df_cleaned)

    max_price_threshold = df_cleaned[target_column].quantile(percentile_cutoff)

    df_cleaned = df_cleaned[
        (df_cleaned[target_column] <= max_price_threshold)
        & (df_cleaned[target_column] >= min_price_threshold)
    ]

    removed_count = initial_count - len(df_cleaned)

    print("--- Running Dynamic Outlier Removal ---")
    print(f"Calculated {percentile_cutoff*100}th percentile threshold: ${max_price_threshold:,.2f}")
    print(f"Removed {removed_count} listing(s) outside the range [${min_price_threshold} - ${max_price_threshold:,.2f}].")
    print(f"New maximum price in dataset: ${df_cleaned[target_column].max():,.2f}")

    return df_cleaned
