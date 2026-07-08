"""Model explainability via SHAP. Opt-in: requires the `shap` package."""
import matplotlib.pyplot as plt


def explain_with_shap(tree_model, X_sample, savepath=None):
    """Plot a SHAP summary for a fitted tree model (e.g. an XGBoost regressor)."""
    import shap

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=savepath is None)
    if savepath:
        plt.savefig(savepath)
    return shap_values
