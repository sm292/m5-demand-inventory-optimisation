
# Demand Forecasting & Inventory Optimisation

## Business objective
Forecast daily retail product demand and translate predictions
into inventory replenishment recommendations.

## Dataset
M5 Forecasting - Accuracy, Walmart retail sales.

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

## Methods
- Seasonal-naive baseline
- HistGradientBoostingRegressor
- Lag and rolling-demand features
- Calendar and event features
- Chronological 28-day validation
- MAE, RMSE, WAPE and forecast bias
- Safety stock, reorder point and target stock calculations

## Run in GitHub Codespaces

```bash
python -m pip install -r requirements.txt
python download_data.py
python train_model.py
streamlit run app.py --server.address 0.0.0.0
```

## Limitations
This prototype samples 100 item-store series and uses
historical sales for validation. It is not a full-scale
reproduction of the M5 competition scoring system.

Inventory recommendations rely on simplified assumptions
about demand variability, supplier lead time and inventory
position. Validate these assumptions before operational use.

