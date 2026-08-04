import numpy as np 
import scipy.io as sp 
import pandas as pd  
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent   
path = BASE / "OxfordBatteryData.mat"
FIGURES = Path(__file__).resolve().parent.parent / "figures"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Helvetica'   
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.edgecolor'] = '#cccccc'  
plt.rcParams['legend.frameon'] = False     

def load_fade_data(path):
    raw = sp.loadmat(path, squeeze_me=True, struct_as_record=False)
    Cell_list = []
    capacity_fade_list = []

    for k in raw.keys():
        if k.startswith('Cell'):
            Cell_list.append(k)

    for i in range(len(Cell_list)):
        cell_data = raw[Cell_list[i]]
        for j in cell_data._fieldnames: 
            Cycle_data= getattr(cell_data, j)
            capacity_data = getattr(Cycle_data, 'C1dc')
            fade_data = getattr(capacity_data, 'q')
            append_tuple = (Cell_list[i],int(j.replace('cyc', '')), np.max(np.abs(fade_data)))
            capacity_fade_list.append(append_tuple)

    capa_df = pd.DataFrame(capacity_fade_list, columns=['Cell', 'Cycle', 'Capacity_Fade'])
    soh = capa_df['Capacity_Fade']/capa_df.groupby('Cell')['Capacity_Fade'].transform('first')
    capa_df['SOH'] = soh
    return capa_df


def clean_data(capa_df):
    capa_df['Percentage_Change'] = capa_df.groupby('Cell')['Capacity_Fade'].pct_change()
    capa_df['is_outlier'] = (capa_df['Percentage_Change'] < -0.05) | (capa_df['Percentage_Change'] > 0.005)
    capa_df_clean = capa_df[capa_df['is_outlier'] == False].copy()
    return capa_df_clean

def plot_data(capa_df_clean):
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=capa_df_clean, x='Cycle', y='SOH', hue='Cell')
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SOH)')
    plt.title('Capacity Fade vs Cycle Number')
    plt.tight_layout()
    plt.savefig(FIGURES / "Capacity_Fade_vs_Cycle_Number.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_data(clean_data(load_fade_data(path)))