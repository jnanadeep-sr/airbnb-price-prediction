"""Review sentiment features built with NLTK's VADER analyzer.

Opt-in: requires `nltk` (and the 'vader_lexicon' corpus, downloaded on first use).
Not part of the default fast pipeline because scoring reviews is slow at scale.
"""
import numpy as np
import pandas as pd

ASPECT_KEYWORDS = {
    "host_quality": ["host", "anfitriao", "friendly", "kind", "helpful", "gentil", "ajuda"],
    "cleanliness": ["clean", "limpo", "spotless", "sujo", "dirty", "organized", "arrumado"],
    "location_safety": ["safe", "seguro", "location", "localizacao", "perto", "close", "beach", "praia"],
    "amenities": ["wifi", "ac", "ar condicionado", "pool", "piscina", "kitchen", "cozinha"],
}


def _get_analyzer():
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()


def compute_review_sentiment(reviews, text_column="comments", id_column="listing_id"):
    """Average VADER compound sentiment per listing, from a raw reviews dataframe."""
    sia = _get_analyzer()
    reviews = reviews.dropna(subset=[text_column]).copy()
    reviews["sentiment_score"] = reviews[text_column].apply(lambda text: sia.polarity_scores(str(text))["compound"])
    return reviews.groupby(id_column)["sentiment_score"].mean().reset_index()


def _aspect_sentiment(text, keywords, sia):
    sentences = [s for s in str(text).lower().split(".") if any(k in s for k in keywords)]
    if not sentences:
        return np.nan
    return sia.polarity_scores(" ".join(sentences))["compound"]


def compute_aspect_sentiment(reviews, text_column="comments", id_column="listing_id", aspects=ASPECT_KEYWORDS):
    """Per-listing sentiment broken down by aspect (cleanliness, host quality, etc.),
    plus a `total_reviews_count` of how many reviews contributed to each listing's score.
    """
    sia = _get_analyzer()
    reviews = reviews.dropna(subset=[text_column]).copy()

    for aspect, keywords in aspects.items():
        reviews[f"{aspect}_score"] = reviews[text_column].apply(
            lambda text, kws=keywords: _aspect_sentiment(text, kws, sia)
        )

    agg = {f"{aspect}_score": "mean" for aspect in aspects}
    agg[text_column] = "count"

    host_performance = (
        reviews.groupby(id_column)
        .agg(agg)
        .rename(columns={text_column: "total_reviews_count"})
        .reset_index()
    )
    score_cols = [f"{aspect}_score" for aspect in aspects]
    host_performance[score_cols] = host_performance[score_cols].fillna(0)
    return host_performance
