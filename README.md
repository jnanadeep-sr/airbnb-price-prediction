# Airbnb Price Prediction Project

This repository contains the final project for the SoSe 2026 Machine Learning course. The goal of the project is to build a robust, multimodal machine learning pipeline to predict nightly listing prices for Airbnb accommodations using data from the Inside Airbnb open-data initiative (Rio de Janeiro).

## Project Structure

```text
airbnb-price-prediction/
├── data/
│   ├── raw/            # listings.csv, reviews.csv.gz, calendar.csv.gz (not tracked in git)
│   └── processed/      # cleaned/intermediate CSVs produced by the pipeline
├── notebooks/
│   └── archive/        # earlier exploratory notebooks, superseded by src/
├── src/
│   ├── config.py        # shared paths + constants (random seed, Rio landmark coords, ...)
│   ├── data/
│   │   ├── load.py       # read raw listings/reviews CSVs
│   │   └── clean.py      # target/price cleaning, numerical imputation, outlier removal
│   ├── features/
│   │   ├── selection.py    # keep only high-value predictive columns
│   │   ├── spatial.py      # distance-to-beach / distance-to-landmark features
│   │   ├── text.py         # amenity counts, description word counts, TF-IDF
│   │   ├── sentiment.py    # review sentiment via NLTK VADER (opt-in)
│   │   └── build_features.py  # orchestrates the fast feature-engineering pipeline
│   ├── models/
│   │   ├── linear_models.py  # Linear Regression, Lasso, from-scratch Gradient Descent
│   │   ├── tree_models.py    # Random Forest, LightGBM, XGBoost
│   │   ├── ensemble.py       # Stacking (RF + XGBoost -> RidgeCV), opt-in
│   │   ├── automl.py         # AutoGluon multimodal AutoML, opt-in
│   │   └── explain.py        # SHAP explainability, opt-in
│   ├── visualization/
│   │   ├── eda.py         # price distributions, correlation heatmaps, scatter grids
│   │   ├── model_viz.py   # feature importance, actual-vs-predicted trend charts
│   │   ├── geo.py         # Folium price heatmap, opt-in
│   │   └── text_viz.py    # word clouds, opt-in
│   └── evaluate.py       # MAE / RMSE / R^2 helpers
├── main.py               # entry point that runs the end-to-end pipeline
├── requirements.txt        # core dependencies
└── requirements-extra.txt  # optional dependencies for opt-in stages
```

## Data Setup Instructions

Because the raw Airbnb dataset is quite large, it is ignored by Git and will not be pushed to GitHub. To run the project locally, follow these steps to manually set up your directories:

1. **Download the Dataset:**
   * Obtain the `listings.csv`, `reviews.csv.gz`, and `calendar.csv.gz` datasets (e.g., from Inside Airbnb).

2. **Place the Files in the Repository:**
   * Navigate to the project root directory.
   * Drop the raw files directly into the `data/raw/` subdirectory.

Your local directory structure must match this layout exactly for the pipeline scripts to successfully locate the data:

```text
airbnb-price-prediction/
└── data/
    ├── processed/
    └── raw/
        ├── calendar.csv.gz
        ├── listings.csv
        └── reviews.csv.gz
```

## Running the Pipeline

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Run the fast pipeline (cleaning -> feature engineering -> Linear Regression, Lasso, from-scratch Gradient Descent, Random Forest, LightGBM, XGBoost -> comparison table):

```bash
python main.py
```

Optional flags pull in the heavier/opt-in stages (install `requirements-extra.txt` first):

```bash
pip install -r requirements-extra.txt

python main.py --with-plots       # show EDA + model comparison charts
python main.py --with-sentiment   # merge review sentiment features (nltk)
python main.py --with-automl      # also train an AutoGluon multimodal predictor (slow)
```

The `notebooks/archive/` notebooks contain the original exploratory work (EDA, sentiment analysis, model experiments) that this pipeline was refactored from; they're kept for reference but are no longer the source of truth for the pipeline logic.
