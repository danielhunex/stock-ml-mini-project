import math
import pandas as pd
import numpy as np
from MA.BaseStrategy import BaseStrategy
from Common.MovingAverageCalculator import MovingAverageCalculator
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import math
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


class ExponentialMovingAverageStrategy(BaseStrategy):
    def __init__(self, df, ticker):
        BaseStrategy.__init__(self, df, "EMA")
        self.TICKER = ticker
        self.movingAverage = MovingAverageCalculator(df)

    def create_trading_strategy(self, long_period=50, short_period=20, column='close'):
        # generate the short exponetial moving average
        self.movingAverage.EMA(period=short_period, column=column)
        # generate the long exponential moving average
        self.movingAverage.EMA(period=long_period, column=column)
        self._generate_signal_position(
            long_period=long_period, short_period=short_period)

        self._plot(self.TICKER, short_period=short_period,
                   long_period=long_period)
        return self.df

    def create_feature_label(self, period=20, column='close', order=4):

        # create the short period exponetial moving average
        self.movingAverage.EMA(period=period, column=column)

        mva_column = f'ema_{period}'

        # get local min and max points, which will be also features
        self.df['min'] = self.df.iloc[argrelextrema(
            self.df[mva_column].values, np.less_equal, order=order)[0]]['close']
        self.df['max'] = self.df.iloc[argrelextrema(
            self.df[mva_column].values, np.greater_equal, order=order)[0]]['close']

        # based on local min and max create a label
        self.df["signal"] = 0
        self.df["signal"] = np.where(self.df['min'] > 0, 1, np.where(self.df['max'] > 0, -1, 0))

        self.df.fillna(0, inplace=True)

        # sanitize, no sell before buy   
        self.df = self.__sanitize(self.df)    

        label = self.df["signal"]
        self.df.drop(["signal"], inplace=True, axis=1)

        #scaler transfrom
        scalar_transformer = MinMaxScaler()

        features = ["close", 'min', 'max', mva_column]

        df_transformed = self.df[features]

        df_scaled = scalar_transformer.fit_transform(df_transformed)

        # convert back to dataframe
        self.df = pd.DataFrame(columns=df_transformed.columns, data=df_scaled, index=df_transformed.index)

        return self.df, label, scalar_transformer

        # function to create train test data, default 7 month train, 5 month

      # split data to train and test, default last 5 month data to be test
    def split_train_test_data(self, feature_transform, target, split_ratio=5/12):
        test_size = math.ceil(feature_transform.shape[0]*split_ratio)
        train_size = feature_transform.shape[0]-test_size
        # return the splits
        return feature_transform[:train_size], target[0:train_size], feature_transform[train_size:], target[train_size:train_size+test_size]

    def fit(self, X_train, y_train):
        # Create and train the model

        forest = RandomForestClassifier(random_state=7)
        forest.fit(X_train, y_train)

        return forest

    def plot_buy_sell_point(self, df_train, mva_column, period, df_test, ticker):
        plt.figure(figsize=(30, 10))

        
        # plot the train
        df_train['close'].plot(color='k', label='Close Price')
        df_train[mva_column].plot(color='m', label=f'EMA {period} Price')

        # plot sell
        plt.plot(df_train[df_train['signal'] == -1].index, df_train["close"]
                 [df_train["signal"] == -1], 'v', markersize=15, color='r', label="Sell")

        # plot buy
        plt.plot(df_train[df_train['signal'] == 1].index, df_train["close"]
                 [df_train['signal'] == 1],  '^', markersize=15, color='g', label="Buy")


        # ----------------------

        # plot the test
        
        df_test['close'].plot(color='b', label='Backtest Close Price')
        df_test[mva_column].plot(color='c', label=f'Backtest EMA {period} Price')

        plt.plot(df_test[df_test['signal'] == -1].index, df_test["close"]
                 [df_test["signal"] == -1], 'v', markersize=15, color='#cc6600', label="Sell - backtest")

        # plot buy
        plt.plot(df_test[df_test['signal'] == 1].index, df_test["close"]
                 [df_test['signal'] == 1],  '^', markersize=15, color='#99cc00', label="Buy - backtest")


        plt.title(ticker, fontsize=20)
        plt.legend()
        plt.grid()
        plt.show()

    def generate_train_model(self, ticker,plot=True):
        feature_transform, label, scalar_transformer=self.create_feature_label(20,'close',3)
        X_train ,y_train, X_test,y_test= self.split_train_test_data(feature_transform, label)
       
        predictor = self.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        
        train= pd.DataFrame(columns=X_train.columns, data=scalar_transformer.inverse_transform(X_train), index=X_train.index)
        test=  pd.DataFrame(columns=X_test.columns, data=scalar_transformer.inverse_transform(X_test), index=X_test.index) 

        train["signal"] = y_train
        test["signal"] = y_pred
        test = self.__calc_profit(self.__sanitize(test))
        if plot:
           self.plot_buy_sell_point(train,"ema_20",20,test,ticker)
        return test, y_pred

    #remove sell before buy, or no successive buy signal,
    def __sanitize (self,df):
        prevSignal = 0
        for i in range(0, df.shape[0]):
            if prevSignal == -1:     
                df["signal"][i-1] = 0   # no sell before buy for the first time in trading
                prevSignal = df["signal"][i-1]
            elif prevSignal == 1:
                break
            else:
                prevSignal = df["signal"][i]
        #add positions
        bought = False
        df["positions"] =0
        for i in range(0, df.shape[0]):
            if df["signal"][i] == 1:
                bought =True
                df["positions"][i]=1
            elif df["signal"][i] ==-1:
                bought = False
                df["positions"][i]=0
            elif bought:
                df["positions"][i]=1
                
        df.dropna(inplace=True)
        return df

    def __calc_profit(self,df):
        # daily profit
        df["daily_profit"] = (np.log(df['close'] / df['close'].shift(1))).round(3)    # daily profit (log return ln(Pt+1/Pt))

            # calculate strategy profit
            # each positions =1 is our strategy buy or hold if we bought previously
        df['strategy_profit'] = (df['positions'].shift(1) * df['daily_profit']).round(3)  # profity for the strategy depends on positions and daily profit
            # We need to get rid of the NaN generated in the first row:
        df.dropna(inplace=True)
        return df

