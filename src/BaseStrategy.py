import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
class BaseStrategy:

    def __init__(self, df, mv_type):
        self.df = df
        self.mvType = mv_type

    #function to generate buy and sell signals
    def _generate_signal_position(self, long_period, short_period):
       
        mva_short = f'{self.mvType}_{short_period}'.lower()
        mva_long = f'{self.mvType}_{long_period}'.lower()
       
        df = pd.DataFrame(index=self.df.index)  
        df["signal"] =0.0
            #generate the signal for buy (1) and sell (0)
        df['signal'][short_period:] = np.where(self.df[mva_short][short_period:] > self.df[mva_long][short_period:], 1.0, 0.0) 
         
          # trading orders points based on signal
        df['positions'] = df['signal'].diff()  

          #copy the columns from df to dateframe
        self.df["signal"] = df["signal"]
        self.df["positions"] = df["positions"]

     # function to plot mv, actual data, buy and sell signal points
    def _plot (self,ticker_stock,short_period, long_period ):
        plt.figure(figsize = (20,10))
        mva_short = f'{self.mvType}_{short_period}'.lower()
        mva_long = f'{self.mvType}_{long_period}'.lower()

        # plot close price
        self.df['close'].plot(color = 'k', label= 'Close Price') 

        # plot short term mv
        self.df[mva_short].plot(color = 'b',label = f'{short_period}-day {self.mvType}') 

        #plot long term mv
        self.df[mva_long].plot(color = 'g',label = f'{long_period}-day {self.mvType}') 

        # plot sell
        plt.plot(self.df[self.df['positions'] == -1.0].index, self.df[mva_short][self.df["positions"] == -1.0],'v', markersize=15, color='r', label="Sell")

        #plot buy
        plt.plot(self.df[self.df['positions']  == 1.0].index, self.df[mva_short][self.df['positions'] == 1.0],  '^', markersize=15, color='g', label="Buy")         
          
        plt.ylabel('Price($)', fontsize = 15 )
        plt.xlabel('Date', fontsize = 15 )
        plt.title(ticker_stock, fontsize = 20)
        plt.legend()
        plt.grid()
        plt.show()