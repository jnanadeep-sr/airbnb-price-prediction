"""Raw data loading helpers."""
import pandas as pd

from src.config import RAW_LISTINGS_PATH, RAW_REVIEWS_PATH


def load_listings(path=RAW_LISTINGS_PATH):
    """Load the raw Inside Airbnb listings CSV."""
    return pd.read_csv(path)


def load_reviews(path=RAW_REVIEWS_PATH, nrows=None, sample=None, random_state=None):
    """Load the raw Inside Airbnb reviews CSV (gzip). Reviews files are large,
    so `nrows` or `sample` let callers cap how much gets read into memory.
    """
    reviews = pd.read_csv(path, nrows=nrows)
    if sample is not None and sample < len(reviews):
        reviews = reviews.sample(sample, random_state=random_state)
    return reviews
