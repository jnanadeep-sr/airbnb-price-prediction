"""Stacked ensemble combining Random Forest and XGBoost via a Ridge meta-model."""
from src.config import RANDOM_STATE


def train_stacking_regressor(X_train, y_train, n_estimators=100):
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import RidgeCV
    from xgboost import XGBRegressor

    estimators = [
        ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE)),
        ("xgb", XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE)),
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
    model.fit(X_train, y_train)
    return model
