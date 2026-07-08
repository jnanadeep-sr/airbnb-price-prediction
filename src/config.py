"""Shared paths and constants for the Airbnb price prediction pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_LISTINGS_PATH = RAW_DIR / "listings.csv"
RAW_REVIEWS_PATH = RAW_DIR / "reviews.csv.gz"
RAW_CALENDAR_PATH = RAW_DIR / "calendar.csv.gz"
CLEANED_TARGET_PATH = PROCESSED_DIR / "airbnb_rio_cleaned_target.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COLUMN = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Reference point used for "distance to the beach" spatial features (Ipanema Beach, Rio).
IPANEMA_LAT, IPANEMA_LON = -22.9836, -43.2045

# Reference point used for "distance to city center" (Centro, Rio de Janeiro).
RIO_CENTER_LAT, RIO_CENTER_LON = -22.9068, -43.1729

# Notable Rio landmarks used for the multi-hotspot centrality feature.
RIO_HOTSPOTS = {
    "dist_luxury_beach": (-22.9836, -43.2045),   # Ipanema/Leblon
    "dist_tourism_hub": (-22.9519, -43.2105),    # Christ the Redeemer
    "dist_nightlife_hub": (-22.9133, -43.1821),  # Lapa/Centro
    "dist_airport": (-22.8132, -43.2494),        # GIG Airport
    "dist_stadium": (-22.9121, -43.2302),        # Maracana
}
