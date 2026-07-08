"""AutoGluon multimodal AutoML training. Opt-in: requires the `autogluon` package
and meaningfully more time/compute than the other models (it can spin up a text
transformer for free-text columns like `description`).
"""


def train_autogluon(train_data, label="price", hyperparameters=None, time_limit=600):
    from autogluon.tabular import TabularPredictor

    hyperparameters = hyperparameters or {
        "GBM": {},        # LightGBM
        "XGB": {},        # XGBoost
        "AG_TEXT_NN": {},  # Deep learning text transformer (BERT/DeBERTa)
    }

    predictor = TabularPredictor(label=label, problem_type="regression").fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
    )
    return predictor
