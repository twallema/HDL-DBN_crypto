"""
this script retrieves historical data (up to 36 months) from CoinMarketCap for all currently active crypto coins
"""

import os
import glob
import json
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

abs_dir = os.path.dirname(__file__)

# settings
count = 1096
interval = '24h'
time_end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

################################# 
## retrieve the latest mapping ##
#################################

search_pattern = os.path.join(abs_dir, '../raw/CMC_mappings/CMC-mapping_retrieved_*')
matching_files = glob.glob(search_pattern)

if matching_files:
    latest_file = max(matching_files)
    mapping = pd.read_csv(latest_file)
else:
    raise FileNotFoundError("no mapping found. make sure to run `get_mapping.py` first.")

##################
## get the data ##
##################

load_dotenv(dotenv_path=os.path.join(abs_dir, '../../CMC_API_KEY.env'))

ids = mapping['id'].unique()[:250]  # limited to 250 biggest coins
headers = {"X-CMC_PRO_API_KEY": os.getenv("API_KEY")}
data_collect = []

for id in ids:

    url = f"https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/historical?id={id}&time_end={time_end}T23%3A59%3A00.000Z&count={count}&interval={interval}"

    response = requests.get(url, headers=headers)

    parsed_dict = json.loads(response.text)

    data = parsed_dict["data"][str(id)]['quotes']
    
    timestamps = [datetime.fromisoformat(data[i]['timestamp']) for i in range(len(data))]
    prices = [data[i]['quote']['USD']['price'] for i in range(len(data))]
    volumes = [data[i]['quote']['USD']['volume_24h'] for i in range(len(data))]
    market_caps = [data[i]['quote']['USD']['market_cap'] for i in range(len(data))]
    total_supplies = [data[i]['quote']['USD']['total_supply'] for i in range(len(data))]
    circulating_supplies = [data[i]['quote']['USD']['circulating_supply'] for i in range(len(data))]

    data = pd.DataFrame(index=pd.Index(timestamps, name='time'),
                        data=np.transpose(np.stack([prices, volumes, market_caps, total_supplies, circulating_supplies])),
                        columns=["price_usd", "volume_24h", "market_cap", "total_supply", "circulating_supply"]).reset_index()

    data['name'] = mapping[mapping['id'] == id]['name'].values[0]
    data['ticker'] = mapping[mapping['id'] == id]['symbol'].values[0]

    data = data[['time', 'ticker', 'name', 'price_usd', 'volume_24h', 'market_cap', 'total_supply', 'circulating_supply']]

    data_collect.append(data)

data = pd.concat(data_collect, axis=0)

os.makedirs(os.path.join(abs_dir, '../raw/prices'), exist_ok=True)
data.to_parquet(os.path.join(abs_dir, f'../raw/prices/historical-prices_interval-{interval}_retrieved_{datetime.today().strftime('%Y-%m-%d')}.parquet.gz'), index=False, compression='gzip')