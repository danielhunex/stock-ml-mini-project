import time 
import alpaca_trade_api as alpaca 
ENDPOINT="https://paper-api.alpaca.markets"
API_KEY_ID="" # Put in yours here - Needed for paper trading
SECRET_KEY="" # Put in yours here - Needed for paper trading

class TradingStrategy:
    def __init__(self,STOCK):
        self.api = alpaca.REST(API_KEY_ID, SECRET_KEY, ENDPOINT)
        self.STOCK = STOCK
        self.SELL_LIMIT_FACTOR = 1.01 # 1 percent margin
        
        # Anything else you want to initialize or a method you want to
        # call during initialization of class - Feel free to add
        
        # Get past 90 days closing prices
        self.get_past90_closing_prices()
    def try_me(self):
        print("IT works") 
    
    def get_past90_closing_prices(self):  
        barset = self.api.get_barset(self.STOCK, 'day', limit=90)
        bars = barset[self.STOCK]
        self.past90_closing_prices = [bars[index].c for index in range(len(bars))]
    
    def get_current_price(self):
        return float(self.api.get_last_trade(self.STOCK).price)
    
    def get_quantity_buy(self):
        if int(float(self.api.get_account().cash)) > 0:
            return int((float(self.api.get_account().cash)/2) \
                       /self.get_current_price())
        else:
            return 0
        
    def exists_buy_order(self):
        # Identifies if a buy order exists for a stock
        orders = self.api.list_orders()
        for order in orders:
            if order.side=="buy" and order.symbol==self.STOCK:
                return True
        
        return False
    
    def have_bought_stock(self):
        positions=self.api.list_positions()
        for position in positions:
            if position.symbol==self.STOCK and int(position.qty)==self.NEW_QUANTITY + self.EXISTING_QUANTITY:
                return True
        return False
        
        
    def get_buy_price(self):
        # Identify the buying price for a stock
        positions=self.api.list_positions()
        for position in positions:
            if position.symbol==self.STOCK:
                return float(position.cost_basis)/int(position.qty)
    
    
    def buy_market_order(self):
        # Buy the stock at market price (This is for paper-trading)
        if self.NEW_QUANTITY > 0:
            self.api.submit_order(self.STOCK, \
                        qty=self.NEW_QUANTITY,\
                        side="buy",\
                        type="market", \
                        time_in_force="day",
                        order_class=None)
        
    def buy_limit_order(self,base_price):
        pass
        
    def sell_limit_order(self):
        # (This is for paper-trading)
        pass
        # Your code if you want to sell at limit
        # Check Alpaca docs on selling at limit
        
    def identify_strategy_for_selling(self):
        # If you have multiple strategies
        # Pick between them here - Or use ML to help identify 
        # your strategy
        pass
        
        
    def market_buy_strategy(self):
        # Providing a simple trading strategy here:
        # Buy at market price if conditions are favorable for buying
        # Sell at a limit price that is determined based on buying price
        # This strategy doesn't use any ML here - You may want to use
        # appropriate libraries to train models + use the trained strategy 
        # here
        
        # Get existing quantity
        positions = self.api.list_positions()
        self.EXISTING_QUANTITY = 0
        for position in positions:
            if position.symbol == self.STOCK:
                self.EXISTING_QUANTITY += int(position.qty)
                
        # MARKET BUY order
        self.NEW_QUANTITY=self.get_quantity_buy()
        
        if self.NEW_QUANTITY == 0:
            return "ZERO EQUITY"
        
        if not self.exists_buy_order():
            self.buy_market_order()
            
        
        # BRACKET SELL order
        # Initiate sell order if stock has been bought
        # If not, wait for it to be bought
        while not self.have_bought_stock():
            #print(self.api.positions)
            #print(self.NEW_QUANTITY + self.EXISTING_QUANTITY)
            time.sleep(1)
        
        if self.have_bought_stock():
            buy_price=self.get_buy_price()
            self.SELL_LIMIT_PRICE=int(float(buy_price))*self.SELL_LIMIT_FACTOR
            
            # Initiate Sell order
            self.sell_limit_order()
      
    def your_best_strategy(self):
        # Implement here or add other methods to do the same
        pass
        
 