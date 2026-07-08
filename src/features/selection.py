"""Selecting the columns worth keeping for modeling."""

IMPORTANT_COLUMNS = [
    "id",
    # 1. Target Variable
    "price",
    # 2. Geospatial Coordinates (Crucial for Rio)
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    # 3. Property & Capacity Details
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    # 4. Host Profile Metrics
    "host_is_superhost",
    "host_listings_count",
    "hosts_time_as_host_years",
    # 5. Booking Rules & Availability
    "minimum_nights",
    "maximum_nights",
    "availability_365",
    "has_availability",
    # 6. Review Metadata
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
    # 7. Unstructured Heavy Text Columns (for the text processing step)
    "description",
    "customer_reviews",
]


def select_important_features(df):
    """
    Filters the dataset to retain only high-value predictive features,
    removing metadata, raw URLs, and leaking identifier columns.
    """
    existing_targets = [col for col in IMPORTANT_COLUMNS if col in df.columns]

    print("--- Running Feature Selection ---")
    print(f"Original unique columns: {df.shape[1]}")

    for text_col in ["description", "customer_reviews"]:
        if text_col not in df.columns:
            print(f"Warning: '{text_col}' was not found in the current DataFrame. (Will handle in later text merges)")

    df_filtered = df[existing_targets].copy()
    print(f"Filtered columns retained: {df_filtered.shape[1]}")

    return df_filtered
