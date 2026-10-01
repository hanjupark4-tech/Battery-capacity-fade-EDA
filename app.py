import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="Battery SOH Prediction", page_icon="🔋", layout="wide")

PALETTE = px.colors.qualitative.D3
ALL_CELLS = [f"Cell{i}" for i in range(1, 9)]
CELL_COLOR = {c: PALETTE[i] for i, c in enumerate(ALL_CELLS)}


@st.cache_data
def get_raw():
    return core.load_raw()


raw = get_raw()

# ---------------------------------------------------------------- header
st.title("Li-ion Battery Capacity Fade and SOH Prediction")
st.caption(
    "Oxford Battery Degradation Dataset 1 (8 Kokam 740 mAh pouch cells, 40 °C, "
    "urban driving profile). Explore capacity fade, clean outliers, and compare "
    "models that predict State of Health (SOH) from early-life behaviour."
)

tab_over, tab_eda, tab_model, tab_about = st.tabs(
    ["Overview", "Explore and clean", "Predict SOH", "Method and limits"]
)

# ---------------------------------------------------------------- sidebar: cleaning
with st.sidebar:
    st.header("Outlier filter")
    st.write("A cycle is dropped if capacity changes by more than these thresholds versus the previous check-up.")
    low = st.slider("Max drop between check-ups", 0.01, 0.20, 0.05, 0.01, format="%.2f")
    high = st.slider("Max rise between check-ups", 0.000, 0.050, 0.005, 0.001, format="%.3f")

flagged = core.flag_outliers(raw, low=-low, high=high)
clean = flagged[~flagged["is_outlier"]].copy()

# ---------------------------------------------------------------- overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells", raw["Cell"].nunique())
    c2.metric("Check-up points", len(raw))
    c3.metric("Outliers removed", int(flagged["is_outlier"].sum()))
    c4.metric("Max cycle count", int(raw["Cycle"].max()))

    fig = px.line(clean, x="Cycle", y="SOH", color="Cell", color_discrete_map=CELL_COLOR,
                  category_orders={"Cell": ALL_CELLS}, template="plotly_white")
    fig.update_layout(title="Capacity fade vs cycle number", yaxis_title="State of Health (SOH)",
                      xaxis_title="Cycle number", height=480, legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.info(
        "SOH = discharge capacity at a check-up divided by the first check-up capacity. "
        "All eight cells follow a similar knee-shaped decay, which is what makes early-life prediction plausible."
    )

# ---------------------------------------------------------------- explore
with tab_eda:
    left, right = st.columns([1, 3])
    with left:
        cells = st.multiselect("Cells", ALL_CELLS, default=ALL_CELLS)
        show_removed = st.checkbox("Show removed outliers", value=True)
        y_axis = st.radio("Y axis", ["SOH", "Capacity_Fade"],
                          format_func=lambda s: "SOH" if s == "SOH" else "Capacity (mAh)")
    with right:
        sub_clean = clean[clean["Cell"].isin(cells)]
        fig = go.Figure()
        for c in cells:
            g = sub_clean[sub_clean["Cell"] == c]
            fig.add_scatter(x=g["Cycle"], y=g[y_axis], mode="lines+markers", name=c,
                            line=dict(color=CELL_COLOR[c]), marker=dict(size=5))
        if show_removed:
            bad = flagged[flagged["is_outlier"] & flagged["Cell"].isin(cells)]
            fig.add_scatter(x=bad["Cycle"], y=bad[y_axis], mode="markers", name="removed",
                            marker=dict(symbol="x", size=10, color="#d62728"))
        fig.update_layout(template="plotly_white", height=480, xaxis_title="Cycle number",
                          yaxis_title="SOH" if y_axis == "SOH" else "Capacity (mAh)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Removed points")
    removed = flagged[flagged["is_outlier"]][["Cell", "Cycle", "Capacity_Fade", "Percentage_Change"]]
    st.dataframe(removed.rename(columns={"Capacity_Fade": "Capacity (mAh)", "Percentage_Change": "Change vs previous"}),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------- model
with tab_model:
    st.write(
        "Features: **cycle number** and **Initial_Loss** (SOH lost during the first N cycles of each cell). "
        "Train on some cells, predict the SOH curve of unseen cells."
    )
    c1, c2, c3 = st.columns(3)
    test_cells = c1.multiselect("Test cells (unseen)", ALL_CELLS, default=["Cell6", "Cell7", "Cell8"])
    early = c2.slider("Early-life window for Initial_Loss (cycles)", 300, 1500, 500, 100)
    n_est = c3.slider("Random Forest trees", 10, 300, 100, 10)

    train_cells = [c for c in ALL_CELLS if c not in test_cells]
    if len(test_cells) == 0 or len(train_cells) < 2:
        st.warning("Pick at least one test cell and leave at least two cells for training.")
        st.stop()

    df = core.add_initial_loss(clean, early_cycles=early)
    train, test = df[df["Cell"].isin(train_cells)], df[df["Cell"].isin(test_cells)]
    preds, rf = core.fit_predict(train, test, n_estimators=n_est)
    late = (test["Cycle"] > early).values

    rows = []
    for name, p in preds.items():
        m_all = core.metrics(test["SOH"], p)
        m_late = core.metrics(test["SOH"][late], p[late])
        rows.append({"Model": name, "RMSE": m_all["RMSE"], "R2": m_all["R2"],
                     "RMSE (after window only)": m_late["RMSE"], "R2 (after window only)": m_late["R2"]})
    res = pd.DataFrame(rows)

    st.subheader("Test-set performance")
    st.dataframe(res.style.format({c: "{:.4f}" for c in res.columns if c != "Model"}),
                 width="stretch", hide_index=True)
    st.caption("The 'after window only' columns score only cycles beyond the early-life window, "
               "so the model is judged purely on extrapolation, not on cycles used to build Initial_Loss.")

    model_choice = st.radio("Model to plot", list(preds.keys()), index=1, horizontal=True)
    t = test.copy()
    t["Predicted"] = preds[model_choice]
    fig = go.Figure()
    for c in test_cells:
        g = t[t["Cell"] == c]
        fig.add_scatter(x=g["Cycle"], y=g["SOH"], mode="markers", name=f"{c} actual",
                        marker=dict(color=CELL_COLOR[c], size=7, opacity=0.6))
        fig.add_scatter(x=g["Cycle"], y=g["Predicted"], mode="lines", name=f"{c} predicted",
                        line=dict(color=CELL_COLOR[c], width=3))
    fig.add_vline(x=early, line_dash="dot", line_color="gray",
                  annotation_text="end of early-life window", annotation_position="top right")
    fig.update_layout(template="plotly_white", height=500, xaxis_title="Cycle number",
                      yaxis_title="State of Health (SOH)", title=f"{model_choice}: predicted vs actual")
    st.plotly_chart(fig, width="stretch")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Feature importance (Random Forest)")
        imp = pd.DataFrame({"Feature": core.FEATURES, "Importance": rf.feature_importances_})
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", template="plotly_white",
                               height=260), width="stretch")
    with colB:
        st.subheader("Residuals")
        t["Residual"] = t["Predicted"] - t["SOH"]
        st.plotly_chart(px.scatter(t, x="Cycle", y="Residual", color="Cell", color_discrete_map=CELL_COLOR,
                                   template="plotly_white", height=260), width="stretch")

    st.subheader("Leave-one-cell-out cross-validation")
    st.caption("With only 8 cells, a single train/test split is noisy. Here each cell is held out in turn.")
    if st.button("Run cross-validation"):
        with st.spinner("Training 16 models..."):
            cv = core.leave_one_cell_out(df, n_estimators=n_est, early_cycles=early)
        piv = cv.pivot(index="Held-out cell", columns="Model", values="RMSE (cycle > early)")
        fig = px.bar(piv.reset_index().melt(id_vars="Held-out cell", var_name="Model", value_name="RMSE"),
                     x="Held-out cell", y="RMSE", color="Model", barmode="group", template="plotly_white", height=350)
        st.plotly_chart(fig, width="stretch")
        st.write(cv.groupby("Model")[["RMSE (all cycles)", "RMSE (cycle > early)"]].mean().round(4))

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
**Pipeline**
1. Load per-cycle capacity from the raw `.mat` file (`C1dc` discharge curve, maximum of `q`).
2. Compute SOH = capacity / first capacity for each cell.
3. Remove check-ups where capacity jumps by more than a threshold (measurement glitches).
4. Build `Initial_Loss` = SOH drop over the first N cycles of each cell.
5. Train Linear Regression and Random Forest on training cells, evaluate on unseen cells.

**Limits worth stating honestly**
- Only 8 cells and about 500 check-up points, so results are indicative, not production grade.
- Random Forests cannot extrapolate beyond the SOH range seen in training.
- `Initial_Loss` uses the test cell's own early data, which is realistic for early-life prediction
  but means the model is not predicting from zero information. The 'after window only' metrics address this.
- All cells share one temperature and drive profile, so generalisation to other conditions is untested.

**Data**: Birkl, C. (2017). Oxford Battery Degradation Dataset 1. University of Oxford.
        """
    )
