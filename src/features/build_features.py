"""Orchestrates the fast (no-sentiment, no-AutoML) feature engineering pipeline."""
from src.data.clean import clean_numerical_features, clean_price_column, remove_price_outliers_dynamic
from src.features.selection import select_important_features
from src.features.spatial import add_beach_distance
from src.features.text import add_amenity_count, add_description_word_count


def build_feature_matrix(df, min_price_threshold=10, percentile_cutoff=0.99):
    """Run cleaning + feature engineering on a raw (target-cleaned) listings dataframe.

    Steps: numerical imputation -> column selection -> price parsing -> outlier
    removal -> spatial + lightweight text features.
    """
    df = clean_numerical_features(df)
    df = select_important_features(df)
    df = clean_price_column(df)
    df = remove_price_outliers_dynamic(df, min_price_threshold=min_price_threshold, percentile_cutoff=percentile_cutoff)
    df = add_beach_distance(df)
    df = add_amenity_count(df)
    df = add_description_word_count(df)
    return df
