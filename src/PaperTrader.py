import pandas as pd
import numpy as np
import TradingStrategy as tStrategy
import Common.DatetimeUtility as du
from datetime import datetime
from dateutil import tz
import time
import schedule
import time


class PaperTrader:
    def __init__(self, API_KEY_ID, SECRET_KEY, model='ema', STOCKs=["FB", "MSFT", "NFLX", "AMD", "GOOG"]):
        self.model = model
        self.Strategies = {}
        self.STOCKs_money = {}
        for stock in STOCKs:
            self.Strategies[stock] = tStrategy.TradingStrategy(
                stock, API_KEY_ID, SECRET_KEY, self.model)
        self.DatetimeUtility = du.DatetimeUtility()

    def run_trading(self):
        schedule.every().day.at("12:30").do(self.trade)
        while True:
            now = datetime.now(tz=tz.gettz('America/New_York'))
            #while not self.DatetimeUtility.is_market_open_now(now):
                # check if the market is open or not every 1 mins
                #time.sleep(60)
            schedule.run_pending()
            print("Finished running trading for the day...")
            time.sleep(60)

    def trade(self):
        for stock, tradingStrategy in self.Strategies.items():
            tradingStrategy.market_buy_strategy()


# Choice a model and predict sign for each stock

    def model2trade(self,):
        if self.model.lower() == 'stl':
            for stock, tradingStrategy in self.Strategies.items():
                df = tradingStrategy.get_past255_closing_prices()
                df.index = pd.to_datetime(df.index, utc=True)
                tradingStrategy.trained_model = "stl"
                stocks_predict = STL_strategy(
                    stock, df, 'close', 10, 3).strategy()
                tradingStrategy.get_positions_quantity()
                hold_count = tradingStrategy.EXISTING_QUANTITY
                current_price = tradingStrategy.get_current_price()
                loss = np.log(current_price/self.buy_price_stockes[stock])
                # Sell
                if (stocks_predict == -1 or loss < -0.02) and hold_count != 0:
                    hold_count = 0
                    tradingStrategy.client.submit_order(self.STOCK,
                                                        qty=hold_count,
                                                        side="sell",
                                                        type="market",
                                                        time_in_force="day",
                                                        order_class=None)
                # Buy
                elif stocks_predict == 1 and hold_count == 0:
                    hold_count = int(self.STOCKs_money[stock]/current_price)
                    self.buy_price_stockes[stock] = tradingStrategy.get_buy_price(
                    )
                    tradingStrategy.client.submit_order(self.STOCK,
                                                        qty=hold_count,
                                                        side="buy",
                                                        type="market",
                                                        time_in_force="day",
                                                        order_class=None)

        else:
            pass
