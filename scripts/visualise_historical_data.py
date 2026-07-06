"""
this script visualises the historical data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

MC_top = 250
ends = '2026-07-05'
interval = '5m'

# get price data
data = pd.read_parquet(os.path.join(abs_dir, f'../data/raw/prices/historical-prices_interval-{interval}_MC-top{MC_top}_ends-{ends}.parquet.gz'))
nodes = pd.read_csv(os.path.join(abs_dir, '../data/raw/nodes.csv'))

# preprocess data
data = data[data['name'].isin(nodes['name'])][['time', 'ticker', 'name', 'price_usd']]   # slice only relevant coins
data['time'] = pd.to_datetime(data['time'])
data["log_return"] = data.groupby("ticker")["price_usd"].transform(lambda x: np.log(x).diff())

# TLCC (#TODO: make systematic)
y1 = data[data['ticker'] == 'SOL']['log_return'].dropna()
y2 = data[data['ticker'] == 'JUP']['log_return'].dropna()

def numpy_tlcc(x, y, max_lag):
    """
    Computes cross-correlation for 1D arrays over a range of lags.
    Normalizes the arrays to compute Pearson correlation coefficient.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        x_star = x.shift(lag)
        y_star = y[~x_star.isna().values]
        x_star = x_star.dropna()

        # directional correlation
        pos_mask = x_star > 0
        x_star = x_star.loc[pos_mask.values]
        y_star = y_star.loc[pos_mask.values]

        # Assuming x and y are 1D arrays of log returns
        x_star = (x_star - np.mean(x_star)) / np.std(x_star)
        y_star = (y_star - np.mean(y_star)) / np.std(y_star)

        correlations.append(np.mean(x_star.values * y_star.values))
    return lags, np.array(correlations)

max_lag = 7
lags, correlations = numpy_tlcc(y1, y2, max_lag)

best_lag = lags[np.argmax(np.abs(correlations))]
best_corr = correlations[np.argmax(np.abs(correlations))]

print(f"Highest correlation: {best_corr:.2f} at Lag: {best_lag}")

# visualise data
os.makedirs(os.path.join(abs_dir, '../data/raw/prices/fig'), exist_ok=True)

for ticker,name in zip(data['ticker'].unique(), data['name'].unique()):
    vis = data[data['ticker'] == ticker]
    fig,ax = plt.subplots(nrows=2, figsize=(8.1,11.7/2))
    ax[0].plot(vis['time'], vis['price_usd'], color='black')
    ax[1].plot(vis['time'], vis['log_return'], color='black')
    ax[0].set_title(f'{name} ({ticker})')
    ax[0].set_ylabel('Price (USD)')
    ax[1].set_ylabel('Log. Return. (-)')
    plt.tight_layout()
    plt.savefig(os.path.join(abs_dir, f'../data/raw/prices/fig/{ticker}.pdf'))
    plt.close()