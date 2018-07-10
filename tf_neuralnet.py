import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_time_series_from_csv(filename: str, date_col: str, date_format: str = '%Y-%m-%d') -> pd.DataFrame:
    """
    Read time-series (ts) data and parse date column
    :param filename: file name / path
    :param date_col: the column that contains the dates (and will be used as index)
    :param date_format: the date format
    :return: the loaded data frame.
    """

    # Parse dates to create time-series
    def date_parser(date_str: str):
        return pd.datetime.strptime(date_str, date_format)

    data = pd.read_csv(filename, sep=',',
                       parse_dates=[date_col], index_col=date_col, date_parser=date_parser)

    # Sorted descending, to allow the returns to be calculated correctly
    data.sort_index(ascending=False, inplace=True)

    return data


def differencing(raw_ts: pd.Series, look_back: int = 1, log: bool = False) -> pd.Series:
    """
    :param raw_ts: A series that is chronologically ordered, with most recent date as the first row
    :param look_back: Amount of observations over which the difference is calculated
    :param log: Take the logarithmic differences
    :return: Returns a series with differenced values
    """

    # Second order differences are determined if series is already differenced ('d1')
    if '_d1' in raw_ts.name:
        raw_ts_adj = raw_ts + 1
        ts_diff = raw_ts_adj * raw_ts_adj.shift(-look_back)
        ts_diff.name = re.sub('_d1', '_d2', raw_ts.name)

    else:
        ts_diff = raw_ts / raw_ts.shift(-look_back)
        ts_diff.name = "{}_d1".format(raw_ts.name)

    # Logarithmic returns could be helpful to reduce non-stationarity
    if log:
        ts_diff = np.log(ts_diff)
        ts_diff.name = "log_{}".format(ts_diff.name)
    else:
        ts_diff -= 1

    return ts_diff


def build_ar(y, start_lag, n_lags) -> pd.DataFrame:
    """Create lags of a time-series (core of AR-model) and return
    :param start_lag:
    :param lags:
    :return:
    """

    # Placeholder for auto-regressive (AR) series
    ar_terms = pd.DataFrame()
    col_name = 'lag'

    # Create desired AR series
    for l in range(start_lag, n_lags + 1):
        temp_name = '{name}_{lag}'.format(name=col_name, lag=l)
        ar_terms[temp_name] = y.shift(-l)

    return ar_terms


def denormalize(df,norm_data):
    df = df['returns'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new


raw = read_time_series_from_csv(filename='foliekaas_only.csv', date_col='index')
prices = raw['marktprijs_foliekaas']

returns = differencing(prices)
returns.name = 'returns'

pred_context = 3
ar_terms = build_ar(y=returns, start_lag=pred_context, n_lags=10+pred_context)
df = pd.concat([returns, ar_terms], axis=1, join_axes=[returns.index])
df.sort_index(ascending=False, inplace=True)

df.dropna(inplace=True)
# df.shape -> (121,12)

split_date = pd.datetime(2017, 1, 1)
df_test = df[df.index > split_date]
df_train = df[df.index <= split_date]

y_train = np.array(df_train['returns'])
X_train = np.array(df_train.drop('returns', axis=1))

y_test = np.array(df_test['returns'])
X_test = np.array(df_test.drop('returns', axis=1))

# scaler = MinMaxScaler()
#
# X_train = MinMaxScaler().fit_transform(X_train)
# y_train = MinMaxScaler().fit_transform(y_train.values.reshape(-1, 1))
#
# # y is output and x is features.
# X_test = MinMaxScaler().fit_transform(X_test)
# y_test = MinMaxScaler().fit_transform(y_test.values.reshape(-1, 1))


def neural_net_model(X_data, input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim, 10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.tanh(layer_1)

    W_2 = tf.Variable(tf.random_uniform([10, 15]))
    b_2 = tf.Variable(tf.zeros([15]))
    layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    layer_2 = tf.nn.tanh(layer_2)

    W_O = tf.Variable(tf.random_uniform([15, 1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2, W_O), b_O)

    return output, W_O


# Input placeholder
xs = tf.placeholder("float")
ys = tf.placeholder("float")

# Model placeholder
output, W_O = neural_net_model(xs, 11)

# Cost function
cost = tf.reduce_mean(tf.square(output - ys))

# Optimizer
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

c_t = []
c_test = []

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for i in range(100):
        for j in range(X_train.shape[0]):
            sess.run([cost, train], feed_dict={xs: X_train[j, :].reshape(1, 11), ys: y_train[j]})

        pred = sess.run(output, feed_dict={xs: X_train})

        c_t.append(sess.run(cost, feed_dict={xs: X_train, ys: y_train}))
        c_test.append(sess.run(cost, feed_dict={xs: X_test, ys: y_test}))
        print('Epoch :', i, 'Cost :', c_t[i])

    pred_test = sess.run(output, feed_dict={xs: X_test})
    pred_train = sess.run(output, feed_dict={xs: X_train})

    for i in range(y_test.shape[0]):
        print('Original :', y_test[i], 'Predicted :', pred[i])

    print('Cost :', sess.run(cost, feed_dict={xs: X_test, ys: y_test}))

