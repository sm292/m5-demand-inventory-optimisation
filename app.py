
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from inventory import build_recommendations

OUT = Path("models/outputs")

st.set_page_config(
    page_title="Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Demand Forecasting & Inventory Optimisation")
st.caption(
    "Retail demand planning | 28-day forecasts | "
    "Safety stock | Reorder recommendations"
)

required_files = [
    OUT / "future_forecasts.csv",
    OUT / "validation_predictions.csv",
    OUT / "metrics.json",
]

if not all(path.exists() for path in required_files):
    st.error("Forecast files are missing. Run: python train_model.py")
    st.stop()

forecasts = pd.read_csv(OUT / "future_forecasts.csv", parse_dates=["date"])
validation = pd.read_csv(
    OUT / "validation_predictions.csv", parse_dates=["date"]
)

with open(OUT / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

st.sidebar.header("Planning assumptions")

lead_time = st.sidebar.slider(
    "Supplier lead time (days)", 1, 30, 7
)
review_period = st.sidebar.slider(
    "Review period (days)", 1, 30, 7
)
service_level = st.sidebar.select_slider(
    "Target service level",
    options=[0.90, 0.95, 0.975, 0.99],
    value=0.95,
    format_func=lambda x: f"{x:.1%}",
)

recommendations = build_recommendations(
    forecasts,
    validation,
    lead_time_days=lead_time,
    review_period_days=review_period,
    service_level=service_level,
)

st.subheader("Forecast performance")

model_metrics = metrics["model"]
baseline_metrics = metrics["seasonal_naive"]

a, b, c, d = st.columns(4)
a.metric("Series forecast", f"{metrics['series_count']:,}")
b.metric("Model WAPE", (
    f"{model_metrics['wape']:.1%}"
    if model_metrics["wape"] is not None else "N/A"
))
c.metric("Model MAE", f"{model_metrics['mae']:.2f}")
d.metric("Model RMSE", f"{model_metrics['rmse']:.2f}")

if metrics["model_improves_wape"]:
    st.success("The model improved on the seasonal-naive WAPE in validation.")
else:
    st.warning(
        "The model did not improve on the baseline WAPE. "
        "Use the baseline until the model is improved."
    )

st.subheader("Forecast and inventory overview")

left, right = st.columns(2)

with left:
    daily = forecasts.groupby("date", as_index=False)["forecast"].sum()
    fig = px.line(
        daily,
        x="date",
        y="forecast",
        markers=True,
        title="Total predicted daily demand across sampled series",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.histogram(
        recommendations,
        x="daily_forecast",
        nbins=25,
        title="Distribution of average daily demand",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Inventory recommendations")

sort_column = st.selectbox(
    "Rank recommendations by",
    ["recommended_order_qty", "reorder_point", "forecast_28d_total"],
)

ranked = recommendations.sort_values(
    sort_column, ascending=False
)

st.dataframe(ranked, use_container_width=True)

st.download_button(
    "Download inventory recommendations",
    data=ranked.to_csv(index=False).encode("utf-8"),
    file_name="inventory_recommendations.csv",
    mime="text/csv",
)

st.subheader("Inspect an individual product")

series_ids = sorted(forecasts["id"].unique().tolist())
selected_id = st.selectbox("Select item-store series", series_ids)

selected_forecast = forecasts[
    forecasts["id"] == selected_id
].sort_values("date")

selected_validation = validation[
    validation["id"] == selected_id
].sort_values("date")

selected_row = recommendations[
    recommendations["id"] == selected_id
].iloc[0]

st.write(
    f"**Item:** {selected_row['item_id']}  |  "
    f"**Store:** {selected_row['store_id']}"
)

inventory_position = st.number_input(
    "Current inventory position (units)",
    min_value=0,
    value=0,
    step=1,
)

one_item_recommendation = build_recommendations(
    selected_forecast,
    selected_validation,
    lead_time_days=lead_time,
    review_period_days=review_period,
    service_level=service_level,
    inventory_positions={selected_id: inventory_position},
).iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("28-day forecast", f"{selected_row['forecast_28d_total']:.0f} units")
k2.metric("Safety stock", f"{one_item_recommendation['safety_stock']:.1f}")
k3.metric("Reorder point", f"{one_item_recommendation['reorder_point']:.1f}")
k4.metric("Suggested order", f"{one_item_recommendation['recommended_order_qty']} units")

history_chart = selected_validation[["date", "actual", "forecast"]].copy()
history_chart = history_chart.rename(
    columns={"actual": "Actual", "forecast": "Predicted"}
)

fig = px.line(
    history_chart,
    x="date",
    y=["Actual", "Predicted"],
    markers=True,
    title="Chronological validation: actual vs predicted",
)
st.plotly_chart(fig, use_container_width=True)

future_chart = selected_forecast.rename(
    columns={"forecast": "Predicted demand"}
)
fig = px.line(
    future_chart,
    x="date",
    y="Predicted demand",
    markers=True,
    title="Next 28 days of predicted demand",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Forecasts and inventory recommendations are experimental. "
    "The dataset is historical retail data; recommendations require "
    "real stock, supplier lead-time and cost information before use."
)

