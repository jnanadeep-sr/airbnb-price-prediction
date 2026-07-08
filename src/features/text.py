"""Lightweight text-derived features (no heavy NLP dependencies)."""
import ast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def count_amenities(value):
    """Count amenities from either a stringified list or a comma-separated string."""
    if pd.isna(value):
        return 0
    try:
        return len(ast.literal_eval(str(value)))
    except (ValueError, SyntaxError):
        return len(str(value).split(","))


def add_amenity_count(df, column="amenities"):
    df = df.copy()
    df["amenity_count"] = df[column].apply(count_amenities)
    return df


def add_description_word_count(df, column="description"):
    df = df.copy()
    df["desc_word_count"] = df[column].astype(str).apply(lambda x: len(x.split()))
    return df


def build_tfidf_features(df, column="description", max_features=10):
    """Return a dataframe of TF-IDF columns (prefixed 'text_') for the given text column."""
    descriptions = df[column].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(descriptions)
    return pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"text_{word}" for word in tfidf.get_feature_names_out()],
        index=df.index,
    )
