import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BaseStrategy:

    def __init__(self, df, mv_type):
        self.df = df
        self.mvType = mv_type

    # calculates profit for the specific algorithm
    def calculate_profit(self):
        # daily profit
        self.df["daily_profit"] = np.log(
            self.df['close'] / self.df['close'].shift(1)).round(3)

        # calculate strategy profit
        # each signal =1 is our strategy buy or hold if we bought previously
        self.df['strategy_profit'] = self.df['signal'].shift(
            1) * self.df['daily_profit']
        # We need to get rid of the NaN generated in the first row:
        self.df.dropna(inplace=True)

        return self.df

    # function to generate buy and sell signals
    def _generate_signal_position(self, long_period, short_period):

        mva_short = f'{self.mvType}_{short_period}'.lower()
        mva_long = f'{self.mvType}_{long_period}'.lower()

        df = pd.DataFrame(index=self.df.index)
        df["signal"] = 0.0
        # generate the signal for buy (1) and sell (0)
        df['signal'][short_period:] = np.where(
            self.df[mva_short][short_period:] > self.df[mva_long][short_period:], 1.0, 0.0)

        # trading orders points based on signal
        df['positions'] = df['signal'].diff()

        # copy the columns from df to dateframe
        self.df["signal"] = df["signal"]
        self.df["positions"] = df["positions"]

    def plot_mva(self, ticker_stock, df, mva_short, mva_long, short_period_label, long_period_label):
        plt.figure(figsize=(20, 10))
        # plot close price
        df['close'].plot(color='k', label='Close Price')

        # plot short term mv
        df[mva_short].plot(color='b', label=short_period_label)

        # plot long term mv
        df[mva_long].plot(color='g', label=long_period_label)

        # plot sell
        plt.plot(df[df['positions'] == -1.0].index, df[mva_short]
                 [df["positions"] == -1.0], 'v', markersize=15, color='r', label="Sell")

        # plot buy
        plt.plot(df[df['positions'] == 1.0].index, df[mva_short]
                 [df['positions'] == 1.0],  '^', markersize=15, color='g', label="Buy")

        plt.ylabel('Price($)', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.title(ticker_stock, fontsize=20)
        plt.legend()
        plt.grid()
        plt.show()

    # function to plot mv, actual data, buy and sell signal points
    def _plot(self, ticker_stock, short_period, long_period):

        mva_short = f'{self.mvType}_{short_period}'.lower()
        mva_long = f'{self.mvType}_{long_period}'.lower()

        # plot close price
        self.plot_mva(ticker_stock, self.df, mva_short, mva_long,
                      f'{short_period}-day {self.mvType}', f'{long_period}-day {self.mvType}')

    def plot(self, ticker_stock, short_period, long_period,  X_train, X_test, y_pred):
        plt.figure(figsize=(20, 10))
        # merge with the original dataframe to get back the lost columns during feature engineering
        positions = X_train["signal"].diff()
        X_train["positions"] = positions

        X_test["signal"] = y_pred  # create the target from the predicted

        # create the position column from the predicated target column 'signal
        positions = X_test["signal"].diff()
        X_test["positions"] = positions

        # moving average column names
        mva_short = f'{self.mvType}_{short_period}'.lower()
        mva_long = f'{self.mvType}_{long_period}'.lower()

        # labels for moving average
        short_period_label = f'{short_period}-day {self.mvType}'
        long_period_label = f'{long_period}-day {self.mvType}'

        # plotting the training data

        # plot close price
        X_train['close'].plot(color='k', label='Close Price')

        # plot short term mv
        X_train[mva_short].plot(color='b', label=short_period_label + ' train')

        # plot long term mv
        X_train[mva_long].plot(color='g', label=long_period_label + ' train')

        # plot sell
        plt.plot(X_train[X_train['positions'] == -1.0].index, X_train[mva_short]
                 [X_train["positions"] == -1.0], 'v', markersize=15, color='r', label="Sell")

        # plot buy
        plt.plot(X_train[X_train['positions'] == 1.0].index, X_train[mva_short]
                 [X_train['positions'] == 1.0],  '^', markersize=15, color='g', label="Buy")
 
        # plot the test data and the prediction
        
        # plot close price
        X_test['close'].plot(color='c', label='Close Price')

        # plot short term mv
        X_test[mva_short].plot(color='m', label=short_period_label + ' test')

        # plot long term mv
        X_test[mva_long].plot(color='y', label=long_period_label + ' test')

        # plot sell
        plt.plot(X_test[X_test['positions'] == -1.0].index, X_test[mva_short]
                 [X_test["positions"] == -1.0], 'v', markersize=15, color='r', label="Sell")

        # plot buy
        plt.plot(X_test[X_test['positions'] == 1.0].index, X_test[mva_short]
                 [X_test['positions'] == 1.0],  '^', markersize=15, color='g', label="Buy")

        plt.ylabel('Price($)', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.title(ticker_stock, fontsize=20)
        plt.show()
       