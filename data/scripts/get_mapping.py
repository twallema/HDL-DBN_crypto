"""
this script retrieves a mapping of the CoinMarketCap IDs to crypto coins
ranked by market cap, and with information on the "parent" coin
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

abs_dir = os.path.dirname(__file__)

load_dotenv(dotenv_path=os.path.join(abs_dir, '../../CMC_API_KEY'))

url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map?listing_status=active&sort=cmc_rank"

headers = {"x-cmc_pro_api_key": os.getenv("API_KEY")}

response = requests.get(url, headers=headers)

parsed_dict = json.loads(response.text)

crypto_list = parsed_dict["data"]

symbols = [crypto["symbol"] for crypto in crypto_list]
names = [crypto["name"] for crypto in crypto_list]
ids = [crypto["id"] for crypto in crypto_list]
ranks = [crypto["rank"] for crypto in crypto_list]
platforms = [crypto["platform"]["name"] if crypto["platform"] else None for crypto in crypto_list]

crypto_map = pd.DataFrame(data=np.transpose(np.stack([ranks, ids, symbols, names, platforms])), columns=["rank_market_cap", "id", "symbol", "name", "platform"])

os.makedirs(os.path.join(abs_dir, '../raw'), exist_ok=True)
crypto_map.to_csv(os.path.join(abs_dir, f'../raw/crypto_map_{datetime.today().strftime('%Y-%m-%d')}.csv'), index=False)

