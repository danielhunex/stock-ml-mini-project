import pandas as pd
import numpy as np
from MA.BaseStrategy import BaseStrategy
from Common.MovingAverageCalculator import MovingAverageCalculator

class SimpleMovingAverageStrategy(BaseStrategy):
    def __init__(self, df, ticker):
        BaseStrategy.__init__(self, df, "SMA")
        self.TICKER = ticker
        self.movingAverage = MovingAverageCalculator(df)


    def create_trading_strategy(self, long_period=50, short_period=20, column='close'):
         # generate the short exponetial moving average
         self.movingAverage.SMA(period=short_period,column=column)
         # generate the long exponential moving average
         self.movingAverage.SMA(period=long_period,column=column)
         self._generate_signal_position(long_period=long_period, short_period=short_period)

         self._plot(self.TICKER, short_period=short_period,long_period=long_period)
         return self.df

   
  
    


 



        