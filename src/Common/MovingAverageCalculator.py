import pandas as pd

class MovingAverageCalculator:
    def __init__(self,df):
        self.DataFrame = df
    def SMA(self, period=20, column='close'):
         df = pd.DataFrame(index=self.DataFrame.index)         
         df[f"sma_{period}"] = self.DataFrame[column].rolling(window=period,min_periods=1, center=False).mean()
         self.DataFrame[f"sma_{period}"] = df[f"sma_{period}"]
         return self.DataFrame
    def EMA (self,period=20, column='close'):
         df = pd.DataFrame(index=self.DataFrame.index)         
         df[f"ema_{period}"] = self.DataFrame[column].ewm(span=period,min_periods=1, adjust=False).mean()
         self.DataFrame[f"ema_{period}"] = df[f"ema_{period}"]
         return self.DataFrame
