"""Tree-based regressors: Random Forest, LightGBM, XGBoost."""
from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE


def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, n_estimators=300, learning_rate=0.05, num_leaves=31):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=300, learning_rate=0.05, max_depth=6):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
