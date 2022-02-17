from TradingStrategy import TradingStrategy
from ApiClient import ApiClient

Api_Key =''
Secret_Key=''
endpoint=''

ts = ApiClient(api_key_Id=Api_Key,api_key_secret=Secret_Key)
df= ts. get_closing_price("MSFT",365)

print(df.head(10))