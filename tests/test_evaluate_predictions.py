import numpy as np
import pytest

from evaluation import evaluate_predictions


def test_weighted_metrics_and_lorenz_gini():
    # Claim amounts in prediction order are [0, 2, 6], so the curve is
    # [0, 1/4, 1] at [0, 1/2, 1], with area 3/8 and Gini 1/4.
    result = evaluate_predictions([1, 2, 4], [0, 1, 6], [1, 2, 1])["value"]
    assert result.index.tolist() == ["Bias", "Deviance", "MAE", "RMSE", "Gini"]
    assert result["Bias"] == pytest.approx(0.25)
    assert result["MAE"] == pytest.approx(1.25)
    assert result["RMSE"] == pytest.approx(np.sqrt(1.75))
    # At power 1.5: d(y, mu) = 4 * (y/sqrt(mu) - 2*sqrt(y) + sqrt(mu)).
    expected_deviance = (
        4 + 8 * (1 / np.sqrt(2) - 2 + np.sqrt(2))
        + 4 * (3 - 2 * np.sqrt(6) + 2)
    ) / 4
    assert result["Deviance"] == pytest.approx(expected_deviance)
    assert result["Gini"] == pytest.approx(0.25)


def test_zero_claims_have_undefined_gini():
    result = evaluate_predictions([1, 2], [0, 0], [1, 1])["value"]
    assert np.isnan(result["Gini"])
    assert np.isfinite(result.drop("Gini")).all()


@pytest.mark.parametrize(
    "predictions, actuals, weights",
    [([], [], []), ([1], [1, 2], [1]), ([0], [1], [1]),
     ([1], [-1], [1]), ([1], [1], [0]), ([1], [1], [-1]),
     ([np.nan], [1], [1]), ([[1]], [1], [1])],
)
def test_invalid_inputs(predictions, actuals, weights):
    with pytest.raises(ValueError):
        evaluate_predictions(predictions, actuals, weights)
