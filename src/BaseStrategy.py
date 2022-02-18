import  matplotlib.pyplot as plt
class BaseStrategy:

    def __init__(self):
        pass

    def _plot (self,ticker_stock, df,short_period, long_period, mv_type="EMA", ):
        plt.figure(figsize = (20,10))
        mva_short = f'{mv_type}_{short_period}'.lower()
        mva_long = f'{mv_type}_{long_period}'.lower()

        # plot close price
        df['close'].plot(color = 'k', label= 'Close Price') 

        # plot short term mv
        df[mva_short].plot(color = 'b',label = f'{short_period}-day {mv_type}') 

        #plot long term mv
        df[mva_long].plot(color = 'g',label = f'{long_period}-day {mv_type}') 

        # plot sell
        plt.plot(df[df['positions'] == -1.0].index, df[mva_short][df["positions"] == -1.0],'v', markersize=15, color='r', label="Sell")

        #plot buy
        plt.plot(df[df['positions']  == 1.0].index, df[mva_short][df['positions'] == 1.0],  '^', markersize=15, color='g', label="Buy")         
          
        plt.ylabel('Price($)', fontsize = 15 )
        plt.xlabel('Date', fontsize = 15 )
        plt.title(ticker_stock, fontsize = 20)
        plt.legend()
        plt.grid()
        plt.show()