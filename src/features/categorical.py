"""Categorical feature encoding for room_type / property_type / neighbourhood_cleansed.

Encoders are fit on the training split only (to avoid leakage) and then applied
to both train and test via `transform_categorical_features`.
"""
import pandas as pd

DEFAULT_ONE_HOT_COLUMNS = ("room_type",)
DEFAULT_TOP_K_COLUMNS = {"property_type": 15}
DEFAULT_FREQUENCY_COLUMNS = ("neighbourhood_cleansed",)


def fit_categorical_encoders(
    df_train,
    one_hot_columns=DEFAULT_ONE_HOT_COLUMNS,
    top_k_columns=DEFAULT_TOP_K_COLUMNS,
    frequency_columns=DEFAULT_FREQUENCY_COLUMNS,
):
    """Learn category vocabularies / frequency maps from the training split."""
    encoders = {"one_hot_categories": {}, "top_k_categories": {}, "freq_maps": {}}

    for col in one_hot_columns:
        encoders["one_hot_categories"][col] = sorted(df_train[col].dropna().unique().tolist())

    for col, k in top_k_columns.items():
        encoders["top_k_categories"][col] = df_train[col].value_counts().nlargest(k).index.tolist()

    for col in frequency_columns:
        encoders["freq_maps"][col] = df_train[col].value_counts()

    return encoders


def transform_categorical_features(df, encoders):
    """Turn the raw categorical columns into a numeric-only dataframe using
    encoders fit by `fit_categorical_encoders`. Unseen categories at transform
    time fall back to all-zero one-hot rows / a frequency of 0.
    """
    pieces = []

    for col, categories in encoders["one_hot_categories"].items():
        dummies = pd.get_dummies(df[col])
        dummies = dummies.reindex(columns=categories, fill_value=0)
        dummies.columns = [f"{col}_{c}" for c in categories]
        pieces.append(dummies)

    for col, categories in encoders["top_k_categories"].items():
        capped = df[col].where(df[col].isin(categories), other="Other")
        dummies = pd.get_dummies(capped)
        expected_categories = categories + ["Other"]
        dummies = dummies.reindex(columns=expected_categories, fill_value=0)
        dummies.columns = [f"{col}_{c}" for c in expected_categories]
        pieces.append(dummies)

    for col, freq_map in encoders["freq_maps"].items():
        pieces.append(df[col].map(freq_map).fillna(0).rename(f"{col}_frequency"))

    return pd.concat(pieces, axis=1)
