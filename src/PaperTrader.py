import pandas as pd
import numpy as np
import TradingStrategy as tStrategy
import Common.DatetimeUtility as du
from datetime import datetime
from dateutil import tz
from STL.StlMl import STL_strategy
import time

class PaperTrader:
    def __init__(self, STOCKs=["FB","MSFT","NFLX","AMD","GOOG"]):
        money = 100000
        self.stratgies={}
        self.STOCKs_money = {}
        self.STOCKs = STOCKs
        for stock in self.STOCKs:
            self.stratgies[stock]=tStrategy.TradingStrategy(stock)
            self.STOCKs_money[stock]= float(money/5)
        self.DatetimeUtility = du.DatetimeUtility()
        
        self.buy_price_stockes={}
        
        
        

    def run_trading(self):
        while True:
            now = datetime.now(tz=tz.gettz('America/New_York'))
            while not self.DatetimeUtility.is_market_open_now(now):
                 time.sleep(30)
            for stock,tradingStrategy in self.Strategies.items():
                tradingStrategy.market_buy_strategy()
                
                
# Choice a model and predict sign for each stock               
    def model2trade(self, name='stl'):
        if name.lower() == 'stl':        
            for stock,tradingStrategy in self.Strategies.items():
                df = tradingStrategy.get_past255_closing_prices()
                df.index=pd.to_datetime(df.index,utc=True)
                tradingStrategy.trained_model = "stl"
                stocks_predict = STL_strategy(stock,df,'close',10,3).strategy()
                tradingStrategy.get_positions_quantity()
                hold_count = tradingStrategy.EXISTING_QUANTITY
                current_price = tradingStrategy.get_current_price()
                loss = np.log(current_price/self.buy_price_stockes[stock])
                # Sell
                if (stocks_predict ==-1 or loss <-0.02) and hold_count!=0:
                    hold_count = 0 
                    tradingStrategy.client.submit_order(self.STOCK,
                                     qty=hold_count,
                                     side="sell",
                                     type="market",
                                     time_in_force="day",
                                     order_class=None)
                # Buy    
                elif stocks_predict == 1 and hold_count==0:
                    hold_count = int(self.STOCKs_money[stock]/current_price)
                    self.buy_price_stockes[stock] = tradingStrategy.get_buy_price()
                    tradingStrategy.client.submit_order(self.STOCK,
                                     qty=hold_count,
                                     side="buy",
                                     type="market",
                                     time_in_force="day",
                                     order_class=None)
        
        else:
            pass
                
                
                
                
            