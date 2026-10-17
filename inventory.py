
from statistics import NormalDist

import numpy as np
import pandas as pd


def build_recommendations(
    forecasts,
    validation,
    lead_time_days=7,
    review_period_days=7,
    service_level=0.95,
    inventory_positions=None,
):
    """
    Create approximate inventory recommendations.

    inventory_positions: optional dict mapping series id to
    inventory position (on-hand + on-order - backorders).
    """
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1.")

    if lead_time_days < 1 or review_period_days < 1:
        raise ValueError("Lead time and review period must be positive.")

    z_score = NormalDist().inv_cdf(service_level)
    inventory_positions = inventory_positions or {}

    residuals = validation.copy()
    residuals["residual"] = (
        residuals["actual"] - residuals["forecast"]
    )

    error_std = residuals.groupby("id")["residual"].std()
    global_std = float(residuals["residual"].std())
    if not np.isfinite(global_std):
        global_std = 0.0

    recommendations = []

    for series_id, group in forecasts.groupby("id"):
        group = group.sort_values("date")

        daily_demand = max(
            0.0, float(group["forecast"].mean())
        )

        sigma = float(error_std.get(series_id, global_std))
        if not np.isfinite(sigma):
            sigma = global_std

        # Approximation: forecast error is treated as a proxy for
        # daily demand uncertainty. Validate this assumption with real data.
        safety_stock = z_score * sigma * np.sqrt(lead_time_days)

        reorder_point = (
            daily_demand * lead_time_days + safety_stock
        )

        protection_period = lead_time_days + review_period_days
        target_stock = (
            daily_demand * protection_period
            + z_score * sigma * np.sqrt(protection_period)
        )

        inventory_position = float(
            inventory_positions.get(series_id, 0.0)
        )

        order_quantity = max(
            0.0, target_stock - inventory_position
        )

        recommendations.append({
            "id": series_id,
            "item_id": group["item_id"].iloc[0],
            "store_id": group["store_id"].iloc[0],
            "daily_forecast": daily_demand,
            "lead_time_days": lead_time_days,
            "service_level": service_level,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "target_stock": target_stock,
            "inventory_position": inventory_position,
            "recommended_order_qty": int(np.ceil(order_quantity)),
            "forecast_28d_total": float(group["forecast"].sum()),
        })

    return pd.DataFrame(recommendations)

