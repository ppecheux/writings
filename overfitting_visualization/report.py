#%%
import altair as alt
import datapane as dp
import pandas as pd
from vega_datasets import data as vega_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

np.random.seed(2022)
#%%
cars = pd.read_json(vega_data.cars.url)[["Horsepower", "Miles_per_Gallon"]]
cars.dropna(inplace=True)

#%%
mask = np.random.rand(len(cars)) < 0.1
cars["train"] = mask
cars["sample set"] = [
    "point in training set" if m else "point in test set" for m in mask
]
train_df = cars[mask]
X_train = np.array(cars[mask]["Horsepower"]).reshape(-1, 1)
y_train = np.array(cars[mask]["Miles_per_Gallon"]).reshape(-1, 1)
X_test = np.array(cars[~mask]["Horsepower"]).reshape(-1, 1)
y_test = np.array(cars[~mask]["Miles_per_Gallon"]).reshape(-1, 1)
#%%
MAX_COEFF = 8


def alt_linear_reg_with_coeff(X: np.array, y: np.array, coeff=1):

    poly = PolynomialFeatures(coeff)
    X = poly.fit_transform(X)
    reg = LinearRegression().fit(X, y)

    x_plot = np.arange(40, 240).reshape(-1, 1)
    x_plot_poly = poly.fit_transform(x_plot)
    y_hat_plot = reg.predict(x_plot_poly)
    return y_hat_plot


pred_df = pd.concat(
    [
        pd.DataFrame(
            {
                "Horsepower": list(range(40, 240)),
                f"Miles_per_Gallon": alt_linear_reg_with_coeff(
                    X_train, y_train, coeff
                ).reshape(1, -1)[0],
                "number_of_parameters": coeff,
            }
        )
        for coeff in range(MAX_COEFF)
    ]
)

#%%


def score_linear_reg_with_coeff(X_train, X_test, y_train, y_test, coeff=1):
    poly = PolynomialFeatures(coeff)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)
    reg = LinearRegression().fit(X_train, y_train)
    score_train = reg.score(X_train, y_train)
    score_test = reg.score(X_test, y_test)
    return score_train, score_test


scores = [
    score_linear_reg_with_coeff(X_train, X_test, y_train, y_test, coeff)
    for coeff in range(MAX_COEFF)
]
scores_train, scores_test = zip(*scores)


score_df = pd.DataFrame(
    {
        "nb_coeff": list(range(MAX_COEFF)),
        "R²_train": scores_train,
        "R²_test": scores_test,
    }
)

#%%
alt_score_chart = (
    alt.Chart(score_df)
    .transform_fold(["R²_test", "R²_train"], as_=["score", "R²"])
    .mark_line()
    .encode(
        alt.X("nb_coeff:O", title="number of model parameters"),
        y="R²:Q",
        color="score:N",
    )
)

alt_score_chart

#%%
selection_train_test = alt.selection_multi(fields=["train"])

color = alt.condition(
    selection_train_test, alt.Color("train:N", legend=None), alt.value("lightgray")
)

alt_chart = (
    alt.Chart(cars)
    .mark_point(filled=True)
    .encode(x="Horsepower:Q", y="Miles_per_Gallon:Q", tooltip="Name:N", color=color)
)
#%%
select_pred = alt.selection_single(
    name="select",
    fields=["number_of_parameters"],
    init={"number_of_parameters": 1},
    bind=alt.binding_range(min=0, max=MAX_COEFF - 1, step=1),
)


legend = (
    alt.Chart(cars)
    .mark_point()
    .encode(y=alt.Y("sample set:N", axis=alt.Axis(orient="right")), color=color)
    .add_selection(selection_train_test)
)

alt_pred_chart = (
    (
        alt.Chart(pred_df)
        .mark_line()
        .encode(
            alt.X("Horsepower", scale=alt.Scale(zero=False)),
            alt.Y("Miles_per_Gallon", scale=alt.Scale(zero=False)),
        )
    )
    .add_selection(select_pred)
    .transform_filter(select_pred)
)

# %%
alt_pred_chart + alt_chart | legend
# %%
dp.Report(
        dp.Plot(
            alt_pred_chart + alt_chart | legend,
            name="linear_regression_with_PolynomialFeatures",
        ),
#        dp.Plot(alt_score_chart, name="R2ScoreComparison")
).publish(name="interactive report to visualize overfitting linear_regression_with_PolynomialFeatures")

dp.Report(
    dp.Plot(
        alt_score_chart, name="R2ScoreComparison"
    )
).publish(name="interactive report to visualize overfitting R2ScoreComparison")
# %%
