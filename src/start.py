from TradingStrategy import TradingStrategy
from ApiClient import ApiClient
from ExponentialMovingAverage import ExponentialMovingAverage

Api_Key =''
Secret_Key=''
endpoint='https://paper-api.alpaca.markets'

ts = ApiClient(api_key_Id=Api_Key,api_key_secret=Secret_Key)
df= ts. get_closing_price("AMZN",365)

ema = ExponentialMovingAverage(df);

ema.create_trading_strategy(long_period=50,short_period=20,column='close')

print(df[df["positions"]==1].head(10))
