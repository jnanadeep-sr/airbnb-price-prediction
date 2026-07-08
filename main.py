"""End-to-end Airbnb price prediction pipeline.

Runs the fast core pipeline by default: clean -> engineer features -> train
Linear Regression, Lasso, from-scratch Gradient Descent, Random Forest,
LightGBM and XGBoost -> print a comparison table.

Heavier, opt-in stages (review sentiment, AutoGluon AutoML, SHAP explainability)
are available as flags but skipped by default since they take minutes and pull
in extra dependencies (nltk, autogluon, shap).
"""
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CLEANED_TARGET_PATH, RANDOM_STATE, RAW_LISTINGS_PATH, TARGET_COLUMN, TEST_SIZE
from src.data.clean import clean_missing_target
from src.data.load import load_reviews
from src.evaluate import print_metrics
from src.features.build_features import build_feature_matrix
from src.models.linear_models import GradientDescentRegressor, train_lasso, train_linear_regression
from src.models.tree_models import train_lightgbm, train_random_forest, train_xgboost


def parse_args():
    parser = argparse.ArgumentParser(description="Airbnb price prediction pipeline")
    parser.add_argument("--with-sentiment", action="store_true", help="Merge review sentiment features (requires nltk)")
    parser.add_argument("--with-plots", action="store_true", help="Show EDA and model comparison plots")
    parser.add_argument("--with-automl", action="store_true", help="Also train an AutoGluon predictor (requires autogluon, slow)")
    return parser.parse_args()


def run_fast_pipeline(df):
    """Train the core, fast model set and return {name: metrics} plus predictions for plotting."""
    # 'id' is a listing identifier, not a predictive feature -- some IDs are 19-digit
    # numbers, and feeding that scale into StandardScaler-based models causes overflow.
    X = df.drop(columns=[TARGET_COLUMN, "id"], errors="ignore").select_dtypes(include=[np.number])
    y_log = np.log1p(df[TARGET_COLUMN])

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Feature matrix shape: {X_train.shape}\n")

    results = {}
    predictions = {}

    print("Training Linear Regression...")
    lr_model = train_linear_regression(X_train_scaled, y_train_log)
    predictions["Linear Regression"] = lr_model.predict(X_test_scaled)

    print("Training Lasso...")
    lasso_model = train_lasso(X_train_scaled, y_train_log)
    predictions["Lasso"] = lasso_model.predict(X_test_scaled)

    print("Training from-scratch Gradient Descent...")
    gd_model = GradientDescentRegressor(learning_rate=0.05, epochs=600)
    gd_model.fit(X_train_scaled, y_train_log)
    predictions["Gradient Descent"] = gd_model.predict(X_test_scaled)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train_log)
    predictions["Random Forest"] = rf_model.predict(X_test)

    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train_log)
    predictions["LightGBM"] = lgb_model.predict(X_test)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train_log)
    predictions["XGBoost"] = xgb_model.predict(X_test)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (trained on log1p(price), metrics in real dollars)")
    print("=" * 70)
    for name, y_pred_log in predictions.items():
        y_true_actual = np.expm1(y_test_log)
        y_pred_actual = np.expm1(y_pred_log)
        results[name] = print_metrics(name, y_true_actual, y_pred_actual)

    return results, predictions, y_test_log


def main():
    args = parse_args()

    print("Step 1/4: Cleaning missing target values...")
    clean_missing_target(RAW_LISTINGS_PATH, CLEANED_TARGET_PATH, target_column=TARGET_COLUMN)

    import pandas as pd

    df = pd.read_csv(CLEANED_TARGET_PATH)

    print("\nStep 2/4: Building feature matrix...")
    df = build_feature_matrix(df)

    if args.with_sentiment:
        print("\nStep 2b: Merging review sentiment features...")
        from src.features.sentiment import compute_review_sentiment

        reviews = load_reviews(nrows=50000)
        df_sentiment = compute_review_sentiment(reviews)
        df = df.merge(df_sentiment, left_on="id", right_on="listing_id", how="left")
        df = df.drop(columns=["listing_id"])
        df["sentiment_score"] = df["sentiment_score"].fillna(0)

    print("\nStep 3/4: Training models...")
    results, predictions, y_test_log = run_fast_pipeline(df)

    if args.with_plots:
        print("\nGenerating plots...")
        from src.visualization.model_viz import plot_actual_vs_predicted

        predictions_actual = {name: np.expm1(preds) for name, preds in predictions.items()}
        plot_actual_vs_predicted(np.expm1(y_test_log).values, predictions_actual)

    if args.with_automl:
        print("\nStep 4/4: Training AutoGluon predictor (this can take several minutes)...")
        from src.models.automl import train_autogluon

        automl_df = df[["accommodates", "bedrooms", "beds", "neighbourhood_cleansed", "description", TARGET_COLUMN]].copy()
        train_data, test_data = train_test_split(automl_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        predictor = train_autogluon(train_data, label=TARGET_COLUMN)
        print(predictor.evaluate(test_data))

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
