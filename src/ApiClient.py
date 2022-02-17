import alpaca_trade_api as alpaca 
from pathlib import Path  
from os.path import exists
import pandas as pd
import os

BASE_DIR ="data"

class ApiClient:
    def __init__(self, api_key_Id, api_key_secret, api_url="https://paper-api.alpaca.markets"):
           self.serviceClient=alpaca.REST(api_key_Id, api_key_secret, api_url)

    def get_closing_price(self,stock_ticker,limit=90):
        # if data exists in file, load from file otherwise call from the api, to avoid hitting the api limit
        filepath= self.get_path(stock_ticker=stock_ticker)
        if exists(filepath):
            return self.read_csv(filepath=filepath)            
        else:
            barset = self.serviceClient.get_barset(stock_ticker, 'day', limit=limit)
            bars = barset[stock_ticker]
            self.save_to_csv(bars.df,filepath=filepath)
            return bars.df

    def save_to_csv(self,df, filepath):
        filepath = Path(filepath)          
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)

    def read_csv(self, filepath):
            print("reading from file")
            print(filepath)
            df = pd.read_csv(filepath)
            df.time = pd.to_datetime(df.time)   # converting the time column to DatetimeIndex
            df.set_index('time',inplace=True)    
            return df   
    def get_path(self, stock_ticker):        
      dirname = os.path.dirname(__file__)
      filepath = os.path.join(dirname, f'{BASE_DIR}\{stock_ticker}.csv')
      return filepath
