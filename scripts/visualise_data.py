"""
this script visualises the data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

# get price data
data = pd.read_parquet(os.path.join(abs_dir, '../data/raw/prices/historical-prices_retrieved_2026-07-05.parquet.gz'))
nodes = pd.read_csv(os.path.join(abs_dir, '../data/raw/nodes.csv'))

# preprocess data
data = data[data['name'].isin(nodes['name'])][['time', 'ticker', 'name', 'price_usd']]   # slice only relevant coins
data['time'] = pd.to_datetime(data['time'])
data["log_return"] = data.groupby("ticker")["price_usd"].transform(lambda x: np.log(x).diff())

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