# Online Marketplace Forensic Investigation System
Detect -> Investigate -> Explain. Evidence profiles for sellers in the Olist dataset.

## Run (VS Code terminal)
1. `python -m venv venv` then activate it (`venv\Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Put the 9 Olist CSVs (from your `archive.zip`) in `data/` (names like `olist_orders_dataset.csv`)
4. `python pipeline.py` (builds `output/`; ~1 min) then `python app.py`
5. Open http://127.0.0.1:5000

## Layers
- `pipeline.py`: your notebook's preprocessing (order/seller tables, features) + z-score, IQR, Isolation Forest, Benford (first digit MAD/chi-square), regression residuals, correlation matrix, daily time series with onset/peak detection, repeat-customer and review-burst checks.
- Each seller gets weighted signals, minus legitimate explanations (Black Friday/holiday peak, established large seller) -> investigation priority High/Medium/Low.
- `app.py` (Flask API) + `static/index.html` (dashboard, Chart.js via CDN).
Thresholds are in `pipeline.py` - tune them.
