# %%

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from evaluation import evaluate_predictions
from ps3.data import create_sample_split, load_transform
from tools import lorenz_curve, plot_lorenz_curves, tweedie_scorer

TweedieDist = TweedieDistribution(1.5)
tweedie_cv_scorer = partial(tweedie_scorer, distribution=TweedieDist)
categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
numeric_cols = ["BonusMalus", "Density"]
predictors = categoricals + numeric_cols
param_grid = {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]}

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
    scoring=tweedie_cv_scorer,
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

#reminder - at this point, we have just built 3 models using our training data
#and have now tested them on our test split.

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
'''
constrained_lgbm = GridSearchCV(
    estimator=LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_jobs=1,
        monotone_constraints=monotone_constraints,
    ),
    param_grid=param_grid,
    cv=5,
    scoring=tweedie_cv_scorer,
    n_jobs=2,
    pre_dispatch=2,
    verbose=2,
    error_score="raise",
)'''

constrained_lgbm = clone(cv).set_params(
    estimator__monotone_constraints = monotone_constraints
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

#the point of this section is that you want to determine whether you are overfitting your model
#you take your best model (at this stage you have selected the best hyperparameters but not estimated the model)
#you allow the lightgbm to build trees repeating to minimize loss
#you see at what stage the hyperparameters that you chose based on your training set begin overfitting for 
#for training set. 


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
            df_test["pp_t_lgbm"], 
            df_test["PurePremium"], 
            df_test["Exposure"]
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

# %%
# Exercise 4: Explain the fitted LightGBM models on the held-out policies.

import dalex as dx

explainer_constrained = dx.Explainer(
    model=constrained_lgbm.best_estimator_,
    data=X_test_t,
    y=y_test_t,
    weights=w_test_t,
    label="Constrained LGBM",
)

explainer_unconstrained = dx.Explainer(
    model=lgbm,
    data=X_test_t,
    y=y_test_t,
    weights=w_test_t,
    label="Unconstrained LGBM",
)

pdp_constrained = explainer_constrained.model_profile(
    type="partial",
    variable_type="numerical",
    N=300,
    random_state=42,
)

pdp_unconstrained = explainer_unconstrained.model_profile(
    type="partial",
    variable_type="numerical",
    N=300,
    random_state=42,
)
# %%
pdp_constrained.plot(pdp_unconstrained)

# %%
# Exercise 5: Compare SHAP explanations for constrained LightGBM and benchmark GLM.

# Reuse explainer_constrained from Exercise 4.
explainer_glm = dx.Explainer(
    model=t_glm1,
    data=X_test_t,
    y=y_test_t,
    weights=w_test_t,
    model_type="regression",
    label="GLM Benchmark",
)

# Select the first observation from the test set
observation = X_test_t.iloc[[0]]

# Calculate SHAP values for constrained LGBM
shap_constrained = explainer_constrained.predict_parts(
    new_observation=observation,
    type="shap",
    B=25,
    random_state=42,
)

# Calculate SHAP values for GLM
shap_glm = explainer_glm.predict_parts(
    new_observation=observation,
    type="shap",
    B=25,
    random_state=42,
)

# Plot and compare both decompositions
shap_constrained.plot(shap_glm, max_vars=X_test_t.shape[1])

# Compare predictions and baseline values
print("GLM prediction:", shap_glm.prediction)
print("GLM baseline:", shap_glm.intercept)

print("LGBM prediction:", shap_constrained.prediction)
print("LGBM baseline:", shap_constrained.intercept)

# %%
