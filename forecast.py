import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

# contains stock data
df = pd.read_csv('prices.csv')
df.date = pd.to_datetime(df.date, format='%Y-%m-%d %H:%M:%S')
df['year'] = pd.DatetimeIndex(df.date).year

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
    scaler = scaler.fit(arr)
    arr_scaled = scaler.transform(arr)
    df = pd.DataFrame(arr, columns=['open', 'low', 'high', 'close'])
    df['symbol'] = frame['symbol']
    return df, scaler


def prepare_data(frame, window):
    """
    Prepares/splits time series data for input into model
    """
    frame.drop(columns=['symbol'], inplace=True)
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


def plot_predictions(symbol):
    subset = df[(df.symbol==symbol) & (df.year >= 2015)]
    scaled, scaler = scale(subset)

    a, b, c, d = prepare_data(scaled, 5)
    x_t = np.concatenate([a, c])
    y_t = np.concatenate([b, d])

    results = model.evaluate(x_t, y_t)
    print(results)

    plt.plot(model.predict(x_t), label='predicted closing price')
    plt.plot(np.squeeze(y_t), alpha=.7, label = 'actual closing price')

    vals = ['January 1, 2015', 'July 1, 2015', 'January 1, 2016', 'July 1, 2016']
    ticks = [0, 182, 366, 547]
    plt.ylabel('Stock Price in USD')
    plt.xlabel('Day')
    plt.xticks(ticks, vals)
    plt.title(f'{symbol} MSE: {round(results[0], 2)}')
    plt.legend()
    plt.show()


def train_model():
    """
    Given ticker symbol, trains lstm model and returns predictions and actuals
    """
    subset = df[df.year < 2015].reset_index(drop=True)
    frame, scaler = scale(subset)

    print(frame.head())

    x_train_sets = []
    y_train_sets = []
    x_test_sets = []
    y_test_sets = []

    for symbol in frame.symbol.unique():
        print(symbol)
        X_train, y_train, X_test, y_test = prepare_data(frame[frame.symbol==symbol]
                                            , 5)
        print(X_train.shape)
        x_train_sets.append(X_train)
        y_train_sets.append(y_train)
        x_test_sets.append((symbol, X_test))
        y_test_sets.append(y_test)

    X_train = np.concatenate(x_train_sets)
    y_train = np.concatenate(y_train_sets)

    print(X_train.shape)
    print(y_train.shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='linear'))

    model.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=2, validation_split=.1)
    return model, history


model, history = train_model()
model.save("stock_model")
for i in random.sample(list(df.symbol.unique()), 20):
    plot_predictions(i)
