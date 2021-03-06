import alpaca_trade_api as alpaca
from pathlib import Path
from os.path import exists
import pandas as pd
import os

BASE_DIR = "../data"


class ApiClient:
    def __init__(self, api_key_Id, api_key_secret, api_url="https://paper-api.alpaca.markets"):
        self.api = alpaca.REST(api_key_Id, api_key_secret, api_url)

    def get_closing_price(self, stock_ticker, limit=90):
        # if data exists in file, load from file otherwise call from the api, to avoid hitting the api limit       
            barset = self.api.get_barset(
                stock_ticker, 'day', limit=limit)
            bars = barset[stock_ticker]
            df = bars.df
            # converting the time column to DatetimeIndex
            time = pd.to_datetime(df.index)
            df.set_index(time, inplace=True)     
            return df
        
    def get_last_trade(self, STOCK):
        return self.api.get_last_trade(STOCK)

    def get_account(self):
        return self.api.get_account()
    def list_orders(self):
        return self.api.list_orders()

    def list_positions(self):
        return self.api.list_positions()
   
    def submit_order(self, STOCK, qty, side, type, time_in_force, order_class, limit_price=None):
        if limit_price == None:
          return self.api.submit_order(STOCK,  qty=qty,  side=side,  type=type, time_in_force=time_in_force, order_class=order_class)
        else:
          return  self.api.submit_order(STOCK,  qty=qty,  side=side,  type=type, time_in_force=time_in_force, order_class=order_class,limit_price=limit_price)
    def save_to_csv(self, df, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)

    def read_csv(self, filepath):
        df = pd.read_csv(filepath)
        # converting the time column to DatetimeIndex
        df.time = pd.to_datetime(df.time)
        df.set_index('time', inplace=True)
        # df.index.name = ['time']
        return df

    def get_path(self, stock_ticker):
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, f'{BASE_DIR}\{stock_ticker}.csv')
        return filepath
