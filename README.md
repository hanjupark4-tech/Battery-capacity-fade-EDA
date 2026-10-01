# Li-ion Battery Capacity Fade and SOH Prediction

Interactive Streamlit app built on the Oxford Battery Degradation Dataset 1.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Regenerate the CSV (only if you change the cleaning logic)
    python build_data.py path/to/OxfordBatteryData.mat

The 266 MB `.mat` file is not in the repo. `data/fade_raw.csv` (36 KB) holds the extracted per-cycle capacities.

## Files
- `app.py`: Streamlit UI (4 tabs)
- `core.py`: cleaning, features, models, leave-one-cell-out CV
- `data.py`, `model.py`: original exploratory scripts (need the .mat)
- `build_data.py`: .mat to CSV conversion
