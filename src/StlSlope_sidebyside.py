# !pip install talib-binary
# !pip install alpaca_trade_api
# !pip install stldecompose

from sklearn.preprocessing import MinMaxScaler
import ApiClient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stldecompose import decompose
from sklearn.cluster import KMeans,AgglomerativeClustering
from stldecompose.forecast_funcs import mean

class StlSlope():
    def __init__ (self,ticker,df,column='close',cycle=20,period=5):
        self.ticker = ticker
        self.df = df
        self.df.index=pd.to_datetime(df.index,utc=True)
        self.column = column
        self.period = period
        self.cycle = cycle
        self.stl()
    
    def stl (self):
        self.decomp = decompose(self.df[[self.column]], self.cycle)
        self.residual = self.decomp.resid
    
    def slope_strategy (self)   :
        # MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-10, 10))
        scaled_data = scaler.fit_transform(self.residual)
        self.residual[self.column]=scaled_data
        
        # Slope
        lis=[]
        position=0
        middle = -mean(self.residual[self.column])
        mem = 1
        
        self.kmeans = KMeans(n_clusters=3,max_iter=1000).fit_predict(self.residual[['close']])
        order = np.zeros_like(self.kmeans)
        order[self.kmeans==self.kmeans[np.argmax(self.residual.values)]]=-1
        order[self.kmeans==self.kmeans[np.argmin(self.residual.values)]]=1
        
        for i in order:  
            # sell for save
            if self.df['close'][i]/mem <0.92 and position==1 :
                lis.append(-1)
                position=0
                continue
            # buy
            elif i==1 and position==0:
                lis.append(1)
                position=1 
                mem = self.df['close'][i]     
                continue 
            # sell
            elif i==0 and position==1:
                lis.append(-1)
                position=0
                continue
            else:
                lis.append(0)
        self.position  = np.array(lis)      
        return self.position 
    
    def publish_trading_strategy (self,principal=10000):
        self.slope_strategy()
        temp = principal
        print('Principal: $',temp)
        sell = self.df.close[self.position == -1.0]
        buy = self.df.close[self.position == 1.0]
        for i in range(sell.shape[0]):
            temp=temp/buy[i]*sell[i]
        print("get: $",temp-principal,"\nThat's ",temp/principal*100,"%")
        self._plot()
        
    def  _plot (self):
        plt.figure(figsize = (20,10))
        self.df[self.column].plot(color = 'k', label= self.column) 
        #plot sell
        plt.plot(self.df.loc[self.position == -1.0].index, 
                 self.df.close[self.position == -1.0],'v', 
                 markersize=15, color='r', label="Sell")

        #plot buy
        plt.plot(self.df.loc[self.position == 1.0].index, 
                 self.df.close[self.position == 1.0],'^', 
                 markersize=15, color='g', label="Buy") 
        plt.ylabel('Price($)', fontsize = 15 )
        plt.xlabel('Date', fontsize = 15 )
        plt.title(self.ticker, fontsize = 20)
        plt.legend()
        plt.grid()
        plt.show()