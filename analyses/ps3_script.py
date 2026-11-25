# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# Why do you think, we divide by exposure here to arrive at our outcome variable?
# We divide ClaimAmountCut by Exposure to obtain PurePremium, which represents the expected claim amount per unit of exposure. Exposure is also used as a sample weight when fitting the model, so observations with greater exposure contribute more to the estimation.

#%%
# Create a reproducible training and testing sample split
df = create_sample_split(df, id_column="IDpol", training_frac=0.8)

# Create a reproducible training and testing sample split
df = create_sample_split(df, id_column="IDpol", training_frac=0.8)

train = np.where(df["sample"] == "train")[0]
test = np.where(df["sample"] == "test")[0]
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

#%%

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

#%%

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)

#%%

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

#%%

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "spline",
            SplineTransformer(
                knots="quantile",
                include_bias=False,
            ),
        ),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist,
                l1_ratio=1,
                fit_intercept=True,
            ),
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

model_pipeline = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    n_jobs=1,
)
model_pipeline.fit(X_train_t, y_train_t, sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
def tweedie_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return -TweedieDist.deviance(y, y_pred) / len(y)

param_grid = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
}

cv = GridSearchCV(
    estimator=LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_jobs=1,
    ),
    param_grid=param_grid,
    cv=5,
    scoring=tweedie_scorer,
    n_jobs=-1,
)

cv.fit(X_train_t, y_train_t, sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()
# %%
# ============================================================
# Problem Set 4
# ============================================================
# %%
# Exercise 1: Monotonicity Constraints

# Step 1: Calculate exposure-weighted average claims

bonus_malus_avg = (
    df.groupby("BonusMalus", as_index=False)
    .agg(
        total_claims=("ClaimAmountCut", "sum"),
        total_exposure=("Exposure", "sum")
    )
)

bonus_malus_avg["AverageClaims"] = (
    bonus_malus_avg["total_claims"]
    / bonus_malus_avg["total_exposure"]
)

print(bonus_malus_avg.head())

# %%

# Plot exposure-weighted average claims by BonusMalus

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    bonus_malus_avg["BonusMalus"],
    bonus_malus_avg["AverageClaims"],
    marker="o"
)

ax.set(
    title="Exposure-weighted Average Claims by BonusMalus",
    xlabel="BonusMalus",
    ylabel="Average Claims per Unit of Exposure"
)

plt.show()

# %%
# Examine groups with the smallest exposure

print(
    bonus_malus_avg
    .sort_values("total_exposure")
    .head(10)
)
# %%
# What will/could happen if we do not include a monotonicity constraint?
# Without a monotonicity constraint, LightGBM may produce non-monotonic predictions due to complex feature interactions and data noise. Consequently, a higher BonusMalus score could sometimes lead to a lower predicted premium, contradicting the expected relationship.

# %%

# Step 2: Create a constrained LightGBM

print(X_train_t.columns.tolist())

monotone_constraints = [
    1 if col == "BonusMalus" else 0
    for col in X_train_t.columns
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
    n_jobs=-1,
)
# %%
# Fit the constrained LightGBM using 5-fold CV

constrained_lgbm.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t
)

print("Best parameters:", constrained_lgbm.best_params_)
print("Best CV score:", constrained_lgbm.best_score_)
print(constrained_lgbm.best_estimator_)
# %%

# Step 3: Predict using the best constrained estimator

df_test["pp_t_lgbm_constrained"] = (
    constrained_lgbm.best_estimator_.predict(X_test_t)
)

df_train["pp_t_lgbm_constrained"] = (
    constrained_lgbm.best_estimator_.predict(X_train_t)
)

print(
    df_test[
        ["PurePremium", "pp_t_lgbm_constrained"]
    ].head()
)
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
    X_train_t, y_train_t,
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
# should explore less thn 100 trees using cross validation.
# # %%
