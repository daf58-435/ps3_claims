import runpy
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from plotly.basedatatypes import BaseFigure
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import GridSearchCV

import ps3.data


def test_combined_analysis_runs_through_ps3_and_ps4(monkeypatch):
    """Exercise all three real estimators without network access or a full grid."""
    rng = np.random.default_rng(42)
    size = 300
    data = pd.DataFrame(
        {
            name: rng.choice(["a", "b", "c"], size)
            for name in ["VehBrand", "VehGas", "Region", "Area"]
        }
    )
    for name in ["DrivAge", "VehAge", "VehPower"]:
        data[name] = rng.integers(1, 4, size)
    data["IDpol"] = np.arange(size)
    data["BonusMalus"] = rng.uniform(50, 150, size)
    data["Density"] = rng.uniform(1, 100, size)
    data["Exposure"] = rng.uniform(0.1, 1, size)
    data["ClaimAmountCut"] = rng.gamma(2, 5, size)
    loader = Mock(return_value=data)
    split = Mock(wraps=ps3.data.create_sample_split)
    monkeypatch.setattr(ps3.data, "load_transform", loader)
    monkeypatch.setattr(ps3.data, "create_sample_split", split)
    searches = []

    def small_search(estimator, **kwargs):
        assert kwargs["cv"] == 5
        assert kwargs["param_grid"] == {
            "learning_rate": [0.05, 0.1], "n_estimators": [100, 200]
        }
        kwargs.update(
            param_grid={"learning_rate": [0.1], "n_estimators": [3]},
            cv=2,
        )
        kwargs.update(n_jobs=1, pre_dispatch=1)
        search = GridSearchCV(estimator, **kwargs)
        searches.append(search)
        return search

    monkeypatch.setattr("sklearn.model_selection.GridSearchCV", small_search)
    subplots = Mock(wraps=plt.subplots)
    monkeypatch.setattr(plt, "subplots", subplots)
    show = Mock()
    monkeypatch.setattr(plt, "show", show)
    monkeypatch.setattr(BaseFigure, "show", Mock())
    try:
        results = runpy.run_module("analyses.ps3_script", run_name="__main__")
    finally:
        plt.close("all")

    loader.assert_called_once()
    split.assert_called_once()
    assert subplots.call_count == 3
    assert show.call_count == 3
    assert len(searches) == 2
    assert "BonusMalus" in results["X_train_t"].columns
    assert results["X_train_t"].columns.equals(results["X_test_t"].columns)
    assert not set(results["df_train"]["IDpol"]) & set(results["df_test"]["IDpol"])
    assert len(results["df_train"]) + len(results["df_test"]) == size
    assert len(results["models"]) == 3
    for label, (estimator, X_train, X_test, column) in results["models"].items():
        for sample, X in [("train", X_train), ("test", X_test)]:
            frame = results["df_" + sample]
            predictions = frame[column]
            np.testing.assert_allclose(predictions, estimator.predict(X))
            expected = mean_tweedie_deviance(
                frame["PurePremium"], predictions,
                sample_weight=frame["Exposure"], power=1.5,
            )
            row = results["model_summary"].loc[(label, sample)]
            assert row["deviance"] == pytest.approx(expected)
            assert row["actual_claims"] == pytest.approx(frame["ClaimAmountCut"].sum())
            assert row["predicted_claims"] == pytest.approx(
                np.sum(predictions * frame["Exposure"])
            )

    constrained = results["constrained_lgbm"].best_estimator_
    for sample in ["train", "test"]:
        frame = results["df_" + sample]
        np.testing.assert_allclose(
            frame["pp_t_lgbm_constrained"],
            constrained.predict(results["X_" + sample + "_t"]),
        )
    assert results["monotone_constraints"][
        results["X_train_t"].columns.get_loc("BonusMalus")
    ] == 1
    assert sum(results["monotone_constraints"]) == 1
    assert set(results["curve_model"].evals_result_) == {"train", "test"}
    for name, column in [
        ("explainer_constrained", "pp_t_lgbm_constrained"),
        ("explainer_unconstrained", "pp_t_lgbm"),
    ]:
        explainer = results[name]
        np.testing.assert_allclose(
            explainer.predict(results["X_test_t"]), results["df_test"][column]
        )
        np.testing.assert_allclose(explainer.y, results["y_test_t"])
        np.testing.assert_allclose(explainer.weights, results["w_test_t"])
    for name, column in [
        ("shap_constrained", "pp_t_lgbm_constrained"),
        ("shap_glm", "pp_t_glm1"),
    ]:
        explanation = results[name]
        assert explanation.prediction == pytest.approx(
            results["df_test"][column].iloc[0]
        )
        assert explanation.intercept == pytest.approx(
            results["df_test"][column].mean()
        )
    comparison = results["lgbm_comparison"]
    assert comparison.shape == (5, 2)
    assert np.isfinite(comparison.to_numpy()).all()
    for label, column in [
        ("Unconstrained LGBM", "pp_t_lgbm"),
        ("Constrained LGBM", "pp_t_lgbm_constrained"),
    ]:
        frame = results["df_test"]
        assert comparison.loc["Deviance", label] == pytest.approx(
            mean_tweedie_deviance(
                frame["PurePremium"], frame[column],
                sample_weight=frame["Exposure"], power=1.5,
            )
        )
