
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc


def tweedie_scorer(estimator, X, y, *, distribution):
    """Calculate unweighted CV deviance using the supplied distribution."""
    return -distribution.deviance(y, estimator.predict(X)) / len(y)


# Source: scikit-learn's Tweedie regression insurance claims example.
def lorenz_curve(y_true, y_pred, exposure):
    """Return cumulative claims ordered by predicted risk."""
    ranking = np.argsort(np.asarray(y_pred))
    claims = np.asarray(y_true) * np.asarray(exposure)
    cumulative_claims = np.cumsum(claims[ranking])
    cumulative_claims /= cumulative_claims[-1]
    return np.linspace(0, 1, len(claims)), cumulative_claims


def plot_lorenz_curves(data):
    """Compare the fitted models and oracle on the held-out policies."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for label, column in [
        ("LGBM", "pp_t_lgbm"),
        ("GLM Benchmark", "pp_t_glm1"),
        ("GLM Splines", "pp_t_glm2"),
        ("Oracle", "PurePremium"),
    ]:
        x, claims = lorenz_curve(data["PurePremium"], data[column], data["Exposure"])
        gini = 1 - 2 * auc(x, claims)
        style = {"linestyle": "-.", "color": "gray"} if label == "Oracle" else {}
        ax.plot(x, claims, label=f"{label} (Gini index: {gini:.3f})", **style)
    ax.plot([0, 1], [0, 1], "--", color="black", label="Random baseline")
    ax.set(
        title="Lorenz Curves",
        xlabel="Fraction of policyholders\n(ordered from safest to riskiest)",
        ylabel="Fraction of total claim amount",
    )
    ax.legend(loc="upper left")
    return fig, ax
# %%
