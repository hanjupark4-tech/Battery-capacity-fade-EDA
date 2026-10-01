"""Data + model logic for the app (reads the precomputed CSV, no .mat needed)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

CSV = Path(__file__).parent / "data" / "fade_raw.csv"
FEATURES = ["Cycle", "Initial_Loss"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Percentage_Change"] = df.groupby("Cell")["Capacity_Fade"].pct_change()
    return df


def flag_outliers(df, low=-0.05, high=0.005) -> pd.DataFrame:
    df = df.copy()
    df["is_outlier"] = (df["Percentage_Change"] < low) | (df["Percentage_Change"] > high)
    return df


def add_initial_loss(df, early_cycles=500) -> pd.DataFrame:
    df = df.copy()
    early = df[df["Cycle"] <= early_cycles]
    g = early.groupby("Cell")["SOH"]
    df["Initial_Loss"] = df["Cell"].map(g.first() - g.last())
    return df


def fit_predict(train, test, n_estimators=100, seed=42):
    out = {}
    lr = LinearRegression().fit(train[FEATURES], train["SOH"])
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed).fit(train[FEATURES], train["SOH"])
    out["Linear Regression"] = lr.predict(test[FEATURES])
    out["Random Forest"] = rf.predict(test[FEATURES])
    return out, rf


def metrics(y, pred):
    return {"RMSE": float(mean_squared_error(y, pred) ** 0.5), "R2": float(r2_score(y, pred))}


def leave_one_cell_out(df, n_estimators=100, early_cycles=500):
    rows = []
    for cell in sorted(df["Cell"].unique()):
        tr, te = df[df["Cell"] != cell], df[df["Cell"] == cell]
        preds, _ = fit_predict(tr, te, n_estimators)
        for name, p in preds.items():
            late = te["Cycle"] > early_cycles
            rows.append({"Held-out cell": cell, "Model": name,
                         "RMSE (all cycles)": metrics(te["SOH"], p)["RMSE"],
                         "RMSE (cycle > early)": metrics(te["SOH"][late], p[late.values])["RMSE"]})
    return pd.DataFrame(rows)
