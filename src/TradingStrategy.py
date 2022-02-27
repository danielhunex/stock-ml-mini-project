from ctypes import util
from logging.config import valid_ident
import time
import Common.ApiClient as ac
import Common.DatetimeUtility as utility
import alpaca_trade_api as alpaca
import MA.ExponentialMovingAverageStrategy as ema
from datetime import datetime
from dateutil import tz
import pandas as pd


ENDPOINT = "https://paper-api.alpaca.markets"
# Put in yours here - Needed for paper trading
API_KEY_ID = ""
# Put in yours here - Needed for paper trading
SECRET_KEY = ""


class TradingStrategy:
    def __init__(self, STOCK):
        self.Datetime_Utility = utility.DatetimeUtility()
        self.STOCK = STOCK
        self.SELL_LIMIT_FACTOR = 1.01  # 1 percent margin

        self.client = ac.ApiClient(
            api_key_Id=API_KEY_ID, api_key_secret=SECRET_KEY)

        # Get past one year closing data
        self.df = self.get_past255_closing_prices()
        self.ema_instance = ema.ExponentialMovingAverageStrategy(df=self.df.copy(
            deep=True), ticker=STOCK)  # you can replace this with SimpleMovingAverage
        trained_model, predicted = self.ema_instance.generate_train_model(
            ticker=STOCK, plot=False)
        self.trained_model = trained_model
        self.trained_label = predicted

    def update_strategy_model(self):

        now = datetime.now(tz=tz.gettz('America/New_York'))
        if self.Datetime_Utility.is_market_open_now(now):

            date_str = now.date().strftime("%Y-%m-%d")
            last_date_str = f'{date_str} 05:00:00+00:00'
            last_price, volume = self.get_current_price()

            # create a row with the current price as close price for updating the training models,
            # the rest are filled with the same value but what we care is for close

            data = {'open': last_price, 'high': last_price,
                    'low': last_price, 'close': last_price, 'volume': volume}

            today_df = pd.DataFrame(
                data, index=[pd.to_datetime(last_date_str)])
            time = pd.to_datetime(today_df.index)
            today_df.set_index(time, inplace=True)
            today_df.index.name = "time"

            # append the current price, so that we can predicated on updated model
            df = self.df.copy()
            df = pd.concat([df, today_df])
            df.reset_index()
            time = pd.to_datetime(df.index)
            df.set_index(time, inplace=True)
            df.index.name = "time"

            # retrain with the new df
            self.ema_instance = ema.ExponentialMovingAverageStrategy(
                df=df, ticker=self.STOCK)
            trained_model, predicted = self.ema_instance.generate_train_model(
                self.STOCK, plot=False)
            self.trained_model = trained_model
            self.trained_label = predicted
            return df
        return self.ema_instance.df

    def get_past255_closing_prices(self):
        df = self.client.get_closing_price(self.STOCK, 255)
        return df

    def get_current_price(self):
        return float(self.client.get_last_trade(self.STOCK).price), float(self.client.get_last_trade(self.STOCK).size)

    def get_quantity_buy(self):
        if int(float(self.client.get_account().cash)) > 0:
            current_price, size = self.get_current_price()
            return int((float(self.client.get_account().cash)/5)   # cash should be divided between 5 stocks
                       / current_price)
        else:
            return 0

    def exists_buy_order(self):
        # Identifies if a buy order exists for a stock
        orders = self.client.list_orders()
        for order in orders:
            if order.side == "buy" and order.symbol == self.STOCK:
                return True

        return False

    def have_bought_stock(self):
        positions = self.client.list_positions()
        for position in positions:
            if position.symbol == self.STOCK and int(position.qty) == self.NEW_QUANTITY + self.EXISTING_QUANTITY:
                return True
        return False

    def get_positions_quantity(self):
        self.EXISTING_QUANTITY = 0
        positions = self.client.list_positions()
        for position in positions:
            if position.symbol == self.STOCK:
                # add the quantities for  stock
                self.EXISTING_QUANTITY += int(position.qty)
        return self.EXISTING_QUANTITY

    def get_buy_price(self):
        # Identify the buying price for a stock
        positions = self.client.list_positions()
        for position in positions:
            if position.symbol == self.STOCK:
                return float(position.cost_basis)/int(position.qty)

    def buy_market_order(self):
        # Buy the stock at market price (This is for paper-trading)
        if self.NEW_QUANTITY > 0:
            self.client.submit_order(self.STOCK,
                                     qty=self.NEW_QUANTITY,
                                     side="buy",
                                     type="market",
                                     time_in_force="day",
                                     order_class=None)

    def buy_limit_order(self, base_price):

        self.NEW_QUANTITY = self.get_quantity_buy()

        if self.NEW_QUANTITY > 0:
            self.client.submit_order(self.STOCK,
                                     qty=self.NEW_QUANTITY,
                                     side="buy",
                                     type="limit",
                                     time_in_force="day",
                                     order_class=None,
                                     limit_price=base_price)

    def sell_limit_order(self):
        # (This is for paper-trading)
        # only sell if we have positions, and the model says sell
        # label -1 is sell, 1 is buy, we check the last label( todays label)
        self.get_positions_quantity()

        if self.EXISTING_QUANTITY > 0:
            self.client.submit_order(self.STOCK, qty=self.EXISTING_QUANTITY, side='sell', type='limit', time_in_force='day',
            order_class=None, limit_price=self.SELL_LIMIT_PRICE)  # make this factor to the actual bought price

    def does_strategy_suggest_buy(self):
        self.update_strategy_model()
        return self.trained_label[len(self.trained_label)-1] == 1;
    
    def does_strategy_suggest_sell(self):
        self.update_strategy_model()
        return self.trained_label[len(self.trained_label)-1] == -1; 

    def market_buy_strategy(self):
        # Providing a simple trading strategy here:
        # Buy at market price if conditions are favorable for buying
        # Sell at a limit price that is determined based on buying price
        # This strategy doesn't use any ML here - You may want to use
        # appropriate libraries to train models + use the trained strategy
        # here

        # Get existing quantity
        self.get_positions_quantity()

        # MARKET BUY order
        self.NEW_QUANTITY = self.get_quantity_buy()

        if self.NEW_QUANTITY == 0:
            return "ZERO EQUITY"

        if not self.exists_buy_order():
           if self.does_strategy_suggest_buy():  # we only buy if the strategy says suggests buy
               self.buy_market_order()

        # BRACKET SELL order
        # Initiate sell order if stock has been bought
        # If not, wait for it to be bought
        while not self.have_bought_stock():
            # print(self.api.positions)
            #print(self.NEW_QUANTITY + self.EXISTING_QUANTITY)
            time.sleep(1)

        if self.have_bought_stock():           
            # Initiate Sell order
            if self.does_strategy_suggest_sell():
               buy_price = self.get_buy_price()
               self.SELL_LIMIT_PRICE = int(
               float(buy_price))*self.SELL_LIMIT_FACTOR
               self.sell_limit_order()

    