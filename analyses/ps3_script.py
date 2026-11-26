# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from evaluation import evaluate_predictions
from ps3.data import create_sample_split, load_transform

TweedieDist = TweedieDistribution(1.5)
categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
numeric_cols = ["BonusMalus", "Density"]
predictors = categoricals + numeric_cols
param_grid = {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]}


def tweedie_scorer(estimator, X, y):
    """Use the same unweighted CV score for both LightGBM searches."""
    return -TweedieDist.deviance(y, estimator.predict(X)) / len(y)


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
# Load data and create the train/test split
print("PS3: loading OpenML data...", flush=True)
df = load_transform()

df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
df = create_sample_split(df, id_column="IDpol", training_frac=0.8)
train = np.flatnonzero(df["sample"] == "train")
test = np.flatnonzero(df["sample"] == "test")
df_train, df_test = df.iloc[train].copy(), df.iloc[test].copy()
y = df["PurePremium"]
weight = df["Exposure"].to_numpy()
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

# %%
# Encode predictors for the benchmark GLM and LightGBM
feature_encoder = ColumnTransformer(
    [
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            categoricals,
        ),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")
X_train_t = feature_encoder.fit_transform(df_train[predictors])
X_test_t = feature_encoder.transform(df_test[predictors])

# %%
# Fit the benchmark GLM
print("PS3: fitting benchmark and spline GLMs...", flush=True)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)

t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)

# %%
# Fit the spline GLM
preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("spline", SplineTransformer(knots="quantile", include_bias=False)),
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            categoricals,
        ),
    ]
).set_output(transform="pandas")
spline_glm = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True),
        ),
    ]
)

spline_glm.fit(df_train[predictors], y_train_t, estimate__sample_weight=w_train_t)

# %%
# Tune the unconstrained LightGBM
print("PS3: tuning LightGBM (4 candidates, 5 folds, then refit)...", flush=True)
cv = GridSearchCV(
    LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5, n_jobs=1, verbosity=-1
    ),
    param_grid=param_grid,
    cv=5,
    scoring=tweedie_scorer,
    n_jobs=2,
    verbose=2,
    pre_dispatch=2,
    error_score="raise",
)

cv.fit(X_train_t, y_train_t, sample_weight=w_train_t)
lgbm = cv.best_estimator_

# %%
# Compare fitted models on the train and test samples
models = {
    "LGBM": (lgbm, X_train_t, X_test_t, "pp_t_lgbm"),
    "GLM Benchmark": (t_glm1, X_train_t, X_test_t, "pp_t_glm1"),
    "GLM Splines": (
        spline_glm, df_train[predictors], df_test[predictors], "pp_t_glm2"
    ),
}
summary_rows = []

for model_name, (estimator, X_train, X_test, column) in models.items():
    for sample, data, X, actuals, exposure in [
        ("train", df_train, X_train, y_train_t, w_train_t),
        ("test", df_test, X_test, y_test_t, w_test_t),
    ]:
        predictions = estimator.predict(X)
        data[column] = predictions
        summary_rows.append(
            {
                "model": model_name,
                "sample": sample,
                "deviance": (
                    TweedieDist.deviance(actuals, predictions, sample_weight=exposure)
                    / exposure.sum()
                ),
                "actual_claims": np.sum(actuals * exposure),
                "predicted_claims": np.sum(predictions * exposure),
            }
        )
model_summary = pd.DataFrame(summary_rows).set_index(["model", "sample"])

print("PS3: models ready.", flush=True)

print(model_summary)

# %%
# Compare risk ranking on the held-out policies
fig, ax = plot_lorenz_curves(df_test)
plt.show()

# %%
# Problem Set 4
# Exercise 1: Monotonicity Constraints

# Step 1: Calculate exposure-weighted average claims

bonus_malus_avg = df.groupby("BonusMalus", as_index=False).agg(
    total_claims=("ClaimAmountCut", "sum"),
    total_exposure=("Exposure", "sum"),
)

bonus_malus_avg["AverageClaims"] = (
    bonus_malus_avg["total_claims"] / bonus_malus_avg["total_exposure"]
)

print(bonus_malus_avg.head())

# %%
# Plot exposure-weighted average claims by BonusMalus

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    bonus_malus_avg["BonusMalus"],
    bonus_malus_avg["AverageClaims"],
    marker="o",
)

ax.set(
    title="Exposure-weighted Average Claims by BonusMalus",
    xlabel="BonusMalus",
    ylabel="Average Claims per Unit of Exposure",
)

plt.show()

# %%
# Examine groups with the smallest exposure

print(bonus_malus_avg.sort_values("total_exposure").head(10))

# %%
# What will/could happen if we do not include a monotonicity constraint?
# Without a monotonicity constraint, LightGBM may produce non-monotonic
# predictions due to complex feature interactions and data noise. Consequently,
# a higher BonusMalus score could sometimes lead to a lower predicted premium,
# contradicting the expected relationship.

# %%
# Step 2: Create a constrained LightGBM

print(X_train_t.columns.tolist())

monotone_constraints = [
    1 if col == "BonusMalus" else 0 for col in X_train_t.columns
]

print(monotone_constraints)

constrained_lgbm = GridSearchCV(
    estimator=LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_jobs=1,
        monotone_constraints=monotone_constraints,
    ),
    param_grid=param_grid,
    cv=5,
    scoring=tweedie_scorer,
    n_jobs=2,
    pre_dispatch=2,
    verbose=2,
    error_score="raise",
)

# %%
# Fit the constrained LightGBM using 5-fold CV

constrained_lgbm.fit(X_train_t, y_train_t, sample_weight=w_train_t)

print("Best parameters:", constrained_lgbm.best_params_)
print("Best CV score:", constrained_lgbm.best_score_)
print(constrained_lgbm.best_estimator_)

# %%
# Step 3: Predict using the best constrained estimator

df_test["pp_t_lgbm_constrained"] = constrained_lgbm.best_estimator_.predict(X_test_t)

df_train["pp_t_lgbm_constrained"] = constrained_lgbm.best_estimator_.predict(X_train_t)

print(df_test[["PurePremium", "pp_t_lgbm_constrained"]].head())

# %%
# Compare the CV scores of the unconstrained and constrained models

print("Unconstrained CV score:", cv.best_score_)
print("Constrained CV score:", constrained_lgbm.best_score_)

print("Unconstrained:", cv.best_params_)
print("Constrained:", constrained_lgbm.best_params_)

# %%
# Exercise 2 – Learning curve
import lightgbm as lgb
from sklearn.base import clone


def cv_tweedie_metric(y_true, y_pred):
    score = -TweedieDist.deviance(y_true, y_pred) / len(y_true)
    return "neg_mean_tweedie_deviance", score, True  # Higher is better


curve_model = clone(constrained_lgbm.best_estimator_)
curve_model.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t,
    eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    eval_names=["train", "test"],
    eval_sample_weight=[w_train_t, w_test_t],
    eval_metric=cv_tweedie_metric,
)

lgb.plot_metric(
    curve_model,
    metric="neg_mean_tweedie_deviance",
    title="Constrained LightGBM learning curve",
    xlabel="Boosting iteration (0 = first tree)",
    ylabel="Negative mean Tweedie deviance",
)
plt.show()
# Both the training and test scores improve in the beginning. The test score
# achieves the best using 38 trees, then declines while the training score
# continues to improve. This suggests overfitting in later iterations, so we
# should explore fewer than 100 trees using cross-validation.

# %%
# Exercise 3: Evaluate constrained and unconstrained LightGBM models.
# Each column uses the same held-out policies and exposure weights.

lgbm_comparison = pd.concat(
    {
        "Unconstrained LGBM": evaluate_predictions(
            df_test["pp_t_lgbm"], df_test["PurePremium"], df_test["Exposure"]
        )["value"],
        "Constrained LGBM": evaluate_predictions(
            df_test["pp_t_lgbm_constrained"],
            df_test["PurePremium"],
            df_test["Exposure"],
        )["value"],
    },
    axis=1,
)
print("Exposure-weighted test metrics (Gini follows the Lorenz plot):")
print(lgbm_comparison)
# Prefer bias close to zero, lower deviance/MAE/RMSE, and higher Gini.
