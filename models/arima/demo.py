# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

使用ARIMA模型进行预测

"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.Series.rolling(timeseries, window=20).mean()
    rolstd = pd.Series.rolling(timeseries, window=20).std()


    # Plot rolling statistics:
    plt.ioff()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)

filepath = "../../data_processed/1027_5/data_2.csv"

with open(filepath) as f:
    data = pd.read_csv(f)

# print(data)

"""
    以0.6MA的数据为例

"""
data_ =  data[data["MA"]==0.6]
data_ = data_[["攻角", "实际转速"]]
# 将data_的索引值从0开始
data_.reset_index(inplace=True, drop=True)
times = pd.DataFrame([i for i in np.arange(0, len(data_["攻角"]), 1)])
data_.insert(0, "时间", times)
# test_stationarity(data_["实际转速"])

# 滑动平均
# rolmean = np.array(pd.Series.rolling(data_["实际转速"], window=10).mean())
# data_origin = np.array(data_["实际转速"])
# data_diff = data_origin - rolmean
# data_diff = data_diff[~np.isnan(data_diff)]
# test_stationarity(pd.Series(data_diff))

# 差分
# data_shift = data_["实际转速"].shift()
# diff_1 = data_["实际转速"] - data_shift # 一阶差分
# diff_2 = data_shift - data_shift.shift()
# # print(data_shift)
# # print(data_shift.shift())
# # exit()
# test_stationarity(diff_1[1:])
# test_stationarity(diff_2[2:])

# 滑动平均后，在做差分
rolmean = np.array(pd.Series.rolling(data_["实际转速"], window=10).mean())
data_origin = np.array(data_["实际转速"])
data_diff = data_origin - rolmean
data_diff = pd.Series(data_diff[~np.isnan(data_diff)])
diff_1 = data_diff - data_diff.shift()
test_stationarity(diff_1[1:])
