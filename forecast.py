import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

# contains stock data
df = pd.read_csv('prices.csv')

def time_series(symbol: str):
    """
    Given stock symbol plots time series
    """
    subset = df[df.symbol==symbol]
    plt.plot(subset.high.values, color='y', label='high', alpha=.7)
    plt.plot(subset.low.values, color='g', label='low', alpha=.7)
    plt.plot(subset.open.values, color='b', label='open', alpha=.7)
    plt.plot(subset.close.values, color='r', label='close', alpha=.5)
    plt.xlabel('days')
    plt.ylabel('stock price')
    plt.title(f'{symbol} stock price')
    plt.legend()
    plt.show()


def scale(frame):
    """
    Selects and scales data from data frame
    """
    scaler = MinMaxScaler()
    arr = frame[['open', 'low', 'high', 'close']].to_numpy()
    arr_scaled = scaler.fit_transform(arr)
    df = pd.DataFrame(arr_scaled, columns=['open', 'low', 'high', 'close'])
    return df


def prepare_data(frame, window):
    """
    Prepares/splits time series data for input into model
    """
    windows = []
    data = frame.to_numpy()
    for i in range(data.shape[0] - window):
        windows.append(data[i : i+window, :])
    arr = np.array(windows)
    index = int(len(windows) * .7)
    train = arr[:index, :, :]
    test = arr[index:, :, :]
    return train[:, :-1, :], train[:, -1:, -1:], test[:, :-1, :], test[:, -1:, -1:]


def plot_results(ticker, include_train=True):
    """
    Plots the model output vs. actual values. Defaults plots both train and
    test results, set include train to false to plot only ttest set
    """
    loss = tf.keras.losses.mean_squared_error(actual_close, predicted).numpy()
    plt.plot(range(actual_train.shape[0], actual_train.shape[0]+actual_close.shape[0]),
            actual_close, color='r', alpha=.8, label='Closing Price - test set')
    plt.plot(range(actual_train.shape[0], actual_train.shape[0]+actual_close.shape[0]),
            predicted, color='b', alpha=.8, label='Predicted Closing Price')
    if include_train:
        plt.plot(actual_train, color='m', alpha=.8, label='Closing Price - train set')
        plt.plot(predicted_train, color='c', alpha=.8, label='Predicted Closing Price - train set')
    plt.legend()
    plt.xlabel('days')
    plt.ylabel('stock price (scaled between 0 and 1)')
    plt.title(f'{ticker} stock price real and predicted. mse on test set = {round((sum(loss)/len(loss)), 3)}')
    plt.show()
    return round((sum(loss)/len(loss)), 3)


def train_model(stock):
    """
    Given ticker symbol, trains lstm model and returns predictions and actuals
    """
    stock = scale(df[df.symbol==stock])
    X_train, y_train, X_test, y_test = prepare_data(stock, 5)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='linear'))

    model.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10)

    predicted = model.predict(X_test)
    actual_close = np.squeeze(y_test)
    predicted_train = model.predict(X_train)
    actual_train = np.squeeze(y_train)
    return predicted, actual_close, predicted_train, actual_train, history.history


# Training and evaluating the model
sample = random.sample(list(df.symbol.unique()), 10)
mse = []
for stock in sample:
    predicted, actual_close, predicted_train, actual_train, history = train_model(stock)
    loss = plot_results(stock)
    mse.append(loss)
print(f'Average mean squared error: {round((sum(mse)/len(mse)), 2)}')
