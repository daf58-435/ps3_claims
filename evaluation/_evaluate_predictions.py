import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_tweedie_deviance


def evaluate_predictions(predictions, actuals, sample_weight):
    """Evaluate pure premiums using exposure as sample_weight.

    Bias is the exposure-weighted predicted mean minus the exposure-weighted
    actual mean (positive means overprediction). Deviance is the weighted mean
    Tweedie deviance with power 1.5, matching the fitted models. MAE and RMSE
    also use exposure weights.

    Gini follows ps3_script's Lorenz curve: sort by increasing prediction,
    accumulate actual claim amounts (actuals * exposure), and integrate over
    equally spaced policyholders. It is unnormalized and is undefined (NaN)
    for fewer than two policies or zero total claims. Inputs align by position.
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    arrays = (predictions, actuals, sample_weight)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Inputs must be one-dimensional.")
    if len(predictions) == 0 or not (
        len(predictions) == len(actuals) == len(sample_weight)
    ):
        raise ValueError("Inputs must be nonempty and have the same length.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Inputs must contain only finite values.")
    if np.any(sample_weight < 0) or sample_weight.sum() <= 0:
        raise ValueError("Weights must be nonnegative with a positive total.")
    if np.any(actuals < 0) or np.any(predictions <= 0):
        raise ValueError("Actuals must be nonnegative and predictions positive.")

    errors = predictions - actuals
    ranking = np.argsort(predictions)
    cumulative_claims = np.cumsum((actuals * sample_weight)[ranking])
    gini = np.nan
    if len(actuals) >= 2 and cumulative_claims[-1] > 0:
        cumulative_claims /= cumulative_claims[-1]
        fraction_of_policies = np.linspace(0, 1, len(actuals))
        gini = 1 - 2 * auc(fraction_of_policies, cumulative_claims)

    return pd.DataFrame(
        {
            "value": [
                np.average(errors, weights=sample_weight),
                mean_tweedie_deviance(
                    actuals, predictions, sample_weight=sample_weight, power=1.5
                ),
                np.average(np.abs(errors), weights=sample_weight),
                np.sqrt(np.average(errors**2, weights=sample_weight)),
                gini,
            ]
        },
        index=pd.Index(["Bias", "Deviance", "MAE", "RMSE", "Gini"], name="metric"),
    )
