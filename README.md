# Airbnb Price Prediction Project

This repository contains the final project for the SoSe 2026 Machine Learning course. The goal of the project is to build a robust, multimodal machine learning pipeline to predict nightly listing prices for Airbnb accommodations using data from the Inside Airbnb open-data initiative (Rio de Janeiro).

## Project Structure

```text
airbnb-price-prediction/
├── data/
│   ├── raw/            # listings.csv, reviews.csv.gz, calendar.csv.gz (not tracked in git)
│   └── processed/      # cleaned/intermediate CSVs produced by the pipeline
├── notebooks/
│   ├── pipeline.ipynb                        # cell-by-cell mirror of main.py, with inline plots
│   ├── spatial_features_experiment.ipynb     # ablation: dist_to_beach / dist_to_city_center
│   ├── categorical_features_experiment.ipynb # ablation: room_type/property_type/neighbourhood encoding
│   ├── sentiment_features_experiment.ipynb   # ablation: review sentiment (basic vs. aspect-based)
│   ├── description_features_experiment.ipynb # ablation: description TF-IDF
│   ├── image_features_experiment.ipynb       # ablation: listing photo (ResNet18) embeddings
│   ├── autogluon_experiment.ipynb            # AutoGluon AutoML, incl. a GPU-trained neural net
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
│   │   ├── categorical.py  # one-hot / frequency encoding for room_type, property_type, neighbourhood
│   │   ├── sentiment.py    # review sentiment via NLTK VADER (opt-in)
│   │   ├── image.py        # listing photo -> ResNet18 embeddings (opt-in, requires torch/torchvision)
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
python main.py --with-sentiment   # merge review sentiment features (nltk) -- see Known Limitations below
python main.py --with-automl      # also train an AutoGluon multimodal predictor (slow)
```

The `notebooks/archive/` notebooks contain the original exploratory work (EDA, sentiment analysis, model experiments) that this pipeline was refactored from; they're kept for reference but are no longer the source of truth for the pipeline logic.

## Results

`python main.py` (Rio de Janeiro listings, 80/20 train/test split, `random_state=42`). Models are trained on `log1p(price)`; R^2 is reported in that same log-space (the scale the models were actually optimized on), MAE/RMSE are converted back to real dollars for interpretability.

| Model | MAE | RMSE | R^2 |
|---|---|---|---|
| **XGBoost** | $248.73 | $502.66 | **65.15%** |
| LightGBM | $249.02 | $501.52 | 65.00% |
| Random Forest | $254.53 | $516.33 | 63.48% |
| Linear Regression | $291.29 | $579.43 | 50.25% |
| Gradient Descent (from scratch) | $291.36 | $579.62 | 50.24% |
| Lasso (alpha=1.0) | $403.43 | $766.11 | -0.07% |

Lasso's negative R^2 isn't a bug -- at `alpha=1.0` its L1 penalty zeroes out nearly every coefficient on this feature set (see `lasso_model.coef_` in `notebooks/pipeline.ipynb`); it's kept in the comparison as an honest example of a model that needs its regularization strength tuned rather than one that "just works" out of the box.

For context: median listing price is ~$500, so an MAE of ~$249 (XGBoost) is a meaningful miss in relative terms even though R^2 looks reasonable in aggregate -- price prediction from listing metadata alone typically tops out around 60-75% R^2 in similar studies, since things no tabular model sees (photo/interior quality, exact street-level location, host pricing psychology) are a large share of the remaining variance.

## Process: What We Tried, and What Actually Helped

The assignment asks for multiple modalities and multiple modeling approaches, compared and discussed -- not just a leaderboard. Each ablation below is a standalone, fully-executed notebook (`notebooks/*_experiment.ipynb`) that trains on the **same train/test split** as the change being tested, so the before/after numbers are apples-to-apples. Only changes that clearly helped were folded into `main.py`; the rest are kept as documented negative results.

| # | Experiment | Result | Folded into `main.py`? |
|---|---|---|---|
| 1 | **Categorical encoding** (`categorical_features_experiment.ipynb`) -- `room_type`, `property_type`, `neighbourhood_cleansed` were previously dropped entirely (`select_dtypes(number)` silently discards them) | XGBoost R^2 **61.74% -> 65.49%** (+3.75pt) -- the single biggest gain found | **Yes** |
| 2 | **Spatial features** (`spatial_features_experiment.ipynb`) -- `dist_to_beach` vs. `dist_to_beach` + `dist_to_city_center`, on top of raw lat/lon | `dist_to_beach` alone: +0.2pt. Adding `dist_to_city_center` too: -0.15pt vs. beach-only. Tree models already infer this from raw lat/lon | `dist_to_beach` only |
| 3 | **Review sentiment** (`sentiment_features_experiment.ipynb`) -- found and fixed a coverage bug (`--with-sentiment` only read the first 50k of 1.19M review rows, covering 474 of ~40k listings); aspect-based sentiment (host quality/cleanliness/safety/amenities) vs. basic average sentiment | Aspect sentiment: R^2 65.15% -> **66.00%**, driven mostly by review *count* (popularity), not sentiment content | No -- see Known Limitations |
| 4 | **Description text (TF-IDF)** (`description_features_experiment.ipynb`) -- 30 TF-IDF word columns from the listing description, on top of aspect sentiment | R^2 dropped 66.00% -> 65.25%; none of the 30 word columns crack the top-20 feature importances -- pure noise to the tree | No (negative result) |
| 5 | **Listing photos** (`image_features_experiment.ipynb`) -- ~3k photos -> frozen, ImageNet-pretrained ResNet18 embeddings (30 PCA components) | R^2 dropped 59.24% -> 57.59% on the tested subset -- off-the-shelf ImageNet features aren't aligned with what makes a listing photo "look expensive" | No (negative result) |
| 6 | **AutoGluon AutoML / neural network** (`autogluon_experiment.ipynb`) -- full model zoo (LightGBM, XGBoost, CatBoost, Random Forest, `NeuralNetTorch`), incl. one run forcing a GPU-trained neural net (Apple Silicon MPS, verified via the model's own `.device`) | R^2 up to **68.61%** (beats manual feature engineering), but the neural net alone was the *weakest* individual model (ensemble weight: 4%) -- trees still win on this tabular data | No (exploratory; see below) |

**Why AutoGluon's result (68.6%) isn't the pipeline default**: it wins by combining an unrestricted AutoML search with its own automatic text/categorical feature generation, at the cost of ~10 minutes of training and an extra dependency most of this repo doesn't otherwise need. `src/models/automl.py` + `main.py --with-automl` exist for anyone who wants to reproduce it, but the manually-engineered XGBoost pipeline was kept as the default for speed and transparency.

## Known Limitations / Possible Improvements

- **`--with-sentiment` still has the coverage bug** described in experiment #3 above (`main.py`'s `Step 2b` reads only the first 50k review rows and uses basic, not aspect-based, sentiment). The fix is proven out in `sentiment_features_experiment.ipynb` but hasn't been ported back into `main.py` yet.
- **`calendar.csv.gz` is downloaded but never used.** Availability/pricing-over-time data could support seasonality or demand features.
- **AutoGluon's multimodal text transformer (`AG_TEXT_NN`) cannot run on Apple Silicon** in this environment -- it hard-requires an NVIDIA GPU internally (via `pynvml`) even just to log GPU info, which has no equivalent on Mac (MPS instead of CUDA). This is an upstream AutoGluon limitation, documented in `autogluon_experiment.ipynb`, not something fixable from this repo.
- **Lasso needs its `alpha` tuned** (or dropped) -- at the default `alpha=1.0` it's not a fair comparison against the other models.
- **Images and description text were tested and didn't help** with off-the-shelf features (ResNet18 embeddings, TF-IDF); a fine-tuned image model or pretrained sentence embeddings (instead of bag-of-words TF-IDF) might do better, but weren't tried.
