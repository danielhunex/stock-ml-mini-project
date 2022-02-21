import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])   

import_or_install('importlib')
import_or_install('sys')
import_or_install('os')
import_or_install('pandas')
import_or_install('importlib')
import_or_install("numpy")
import_or_install("matplotlib")

import importlib
import sys
import os,sys
sys.path.insert(1, os.path.join(os.getcwd()  , '..'))

import TradingStrategy as ts
import ApiClient as ac
import ExponentialMovingAverageStrategy as ema
import SimpleMovingAverageStrategy as sma

importlib.reload(ts)
importlib.reload(ac)
importlib.reload(ema)
importlib.reload(sma)

Api_Key ='PKPWLUH9TXMBKRAZ27MH'
Secret_Key='sUJp72yNbrkKICNGhtPRwFX0bViJsP4dw94YpaOn'
endpoint='https://paper-api.alpaca.markets'

client = ac.ApiClient(api_key_Id=Api_Key,api_key_secret=Secret_Key)

for ticker in ["FB","MSFT","NFLX","AMD","GOOG"]:
  df= client. get_closing_price(ticker,365)

  ema_instance = ema.ExponentialMovingAverageStrategy(df=df,ticker=ticker) # you can replace this with SimpleMovingAverage

  #print(df.head(10))
  df= ema_instance.create_trading_strategy(long_period=50,short_period=20,column='close') 

   #calculate the profits
  df = ema_instance.calculate_profit()

  # The returns of the Buy and Hold strategy:
  hold_strategy_profit = df["daily_profit"].sum() * 100

  # The returns of the algorithm
  ema_strategy_profit = df["strategy_profit"].sum() * 100
  

  print(f'Percentage return of Buy and Hand algorithm for {ticker} for 365 day period:  {hold_strategy_profit}%') 
  print(f'Percentage return of {ema_instance.mvType} algorithm for {ticker} for 365 day period:  {ema_strategy_profit}%') 
  
  df.to_csv("data-processed-1.csv")
  