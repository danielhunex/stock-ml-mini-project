from signal import signal
import pandas as pd
import numpy as np
from MovingAverage import MovingAverage

class ExponentialMovingAverage:
    def __init__(self, df):
        self.dataFrame =df
        self.movingAverage = MovingAverage(df)

    def __generate_signal_position(self, long_period, short_period):
          df = pd.DataFrame(index=self.dataFrame.index)  
          df["signal"] =0.0
             #generate the signal for buy (1) and sell (0)
          df['signal'][short_period:] = np.where(self.dataFrame[f'ema_{short_period}'][short_period:] > self.dataFrame[f'ema_{long_period}'][short_period:], 1.0, 0.0) 
         
          # trading orders points based on signal
          df['positions'] = df['signal'].diff()  

          #copy the columns from df to dateframe
          self.dataFrame["signal"] = df["signal"]
          self.dataFrame["positions"] = df["positions"]

    def create_trading_strategy(self, long_period=50, short_period=20, column='close'):
         # generate the short exponetial moving average
         self.movingAverage.EMA(period=short_period,column=column)
         # generate the long exponential moving average
         self.movingAverage.EMA(period=long_period,column=column)
         self.__generate_signal_position(long_period=long_period, short_period=short_period)
         return self.dataFrame

  
    


 



        