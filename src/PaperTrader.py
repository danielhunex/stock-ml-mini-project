import TradingStrategy as tStrategy
import Common.DatetimeUtility as du
from datetime import datetime
from dateutil import tz
import time

class PaperTrader:
    def __init__(self, STOCKs=["FB","MSFT","NFLX","AMD","GOOG"]):

        stratgies={}
        self.STOCKs = STOCKs
        for stock in self.STOCKs:
            stratgies[stock]=tStrategy.TradingStrategy(stock)
        self.Strategies = stratgies
        self.DatetimeUtility = du.DatetimeUtility()

    def run_trading(self):
        while True:
            now = datetime.now(tz=tz.gettz('America/New_York'))
            while not self.DatetimeUtility.is_market_open_now(now):
                 time.sleep(30)
            for stock,tradingStrategy in self.Strategies.items():
                tradingStrategy.market_buy_strategy()