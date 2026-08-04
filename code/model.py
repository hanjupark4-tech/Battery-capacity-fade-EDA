import pandas as pd
from data import load_fade_data, clean_data, path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / "figures"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Helvetica'   
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.edgecolor'] = '#cccccc'  
plt.rcParams['legend.frameon'] = False       

def prepare_data():
    df = clean_data(load_fade_data(path))
    early = df[df['Cycle'] <= 500]
    initial_loss = early.groupby('Cell')['SOH'].first() - early.groupby('Cell')['SOH'].last()
    df['Initial_Loss'] = df['Cell'].map(initial_loss)
    return df

def split_data(df):
    train_df = df[df['Cell'].isin(['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5'])]
    test_df = df[df['Cell'].isin(['Cell6', 'Cell7', 'Cell8'])]
    X_train = train_df[['Cycle', 'Initial_Loss']]
    y_train = train_df['SOH']
    X_test = test_df[['Cycle', 'Initial_Loss']]
    y_test = test_df['SOH']
    return X_train, y_train, X_test, y_test, test_df

def train_and_evaluate(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    lr_rmse = mean_squared_error(y_test, lr_pred)**0.5
    lr_r2 = r2_score(y_test, lr_pred)
    rf_rmse = mean_squared_error(y_test, rf_pred)**0.5
    rf_r2 = r2_score(y_test, rf_pred)
    return lr_rmse, lr_r2, rf_rmse, rf_r2, rf_pred  

def plot_predictions(test_df):
    plt.figure(figsize=(9, 6))
    colors = plt.cm.tab10.colors   
    for i, (cell, g) in enumerate(test_df.groupby('Cell')):
        c = colors[i]
        plt.scatter(g['Cycle'], g['SOH'], s=15, color=c, alpha=0.6, label=f'{cell} actual')
        plt.plot(g['Cycle'], g['Predicted'], color=c, linewidth=2, label=f'{cell} predicted')
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SOH)')
    plt.title('Battery Capacity Fade: Predicted vs Actual')
    plt.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig(FIGURES / 'Battery_fade_Predicted_vs_Actual.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    df = prepare_data()
    X_train, y_train, X_test, y_test, test_df = split_data(df)
    lr_rmse, lr_r2, rf_rmse, rf_r2, rf_pred = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"Linear Regression RMSE: {lr_rmse:.4f}, R^2: {lr_r2:.4f}")
    print(f"Random Forest RMSE: {rf_rmse:.4f}, R^2: {rf_r2:.4f}")
    test_df = test_df.copy()
    test_df['Predicted'] = rf_pred
    plot_predictions(test_df)
