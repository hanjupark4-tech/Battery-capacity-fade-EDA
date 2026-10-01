"""Run once locally: converts the 266MB .mat into a small CSV for the app."""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from data import load_fade_data, clean_data

mat = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("OxfordBatteryData.mat")
raw = load_fade_data(mat)
raw["Percentage_Change"] = raw.groupby("Cell")["Capacity_Fade"].pct_change()
clean = clean_data(raw.copy())
raw.to_csv("data/fade_raw.csv", index=False)
print(raw.shape, clean.shape)
print(raw.groupby("Cell").agg(n=("Cycle","size"), maxcyc=("Cycle","max"), q0=("Capacity_Fade","first")))
