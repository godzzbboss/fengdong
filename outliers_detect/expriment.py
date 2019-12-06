# -*- coding: utf-8 -*-

"""
__author__ == "BigBrother"

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from fbprophet import Prophet

def fit_predict_model(dataframe,
                      interval_width = 0.95,
                      changepoint_range = 0.8,
                      changepoint_prior_scale=0.05,
                      n_changepoints = 50,
                      daily_seasonality=False,
                      yearly_seasonality=False,
                      weekly_seasonality=False
                     ):
    m = Prophet(daily_seasonality = daily_seasonality, yearly_seasonality = yearly_seasonality, weekly_seasonality = weekly_seasonality,
                seasonality_mode = 'additive',
                interval_width = interval_width,
                changepoint_range = changepoint_range,
                changepoint_prior_scale= changepoint_prior_scale,
                n_changepoints=n_changepoints)
    # 非线性增长趋势
    dataframe["cap"] = round(dataframe["y"][0], 2) + 0.004
    dataframe["floor"] = round(dataframe["y"][0], 2) - 0.004

    m = m.fit(dataframe)
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop = True)
    return forecast, m

def plt_scatter(data, y_true, y_pred=None, feature="实际MA", fname=None, save=False):
    colors = ["blue", "#e6550d", "green"]
    # color = [colors[0] if i==1 else colors[1] for i in y_true]
    fig = plt.figure(figsize=(6,8), tight_layout=True)
    # set_size(fig, w_mul=0.6)
    # 获取正常样本的索引
    normal_index = [i for i, x in enumerate(y_true) if x == 1]
    abnormal_index = [i for i in range(len(y_true)) if i not in normal_index]

    f1 = plt.scatter(data.iloc[normal_index]["攻角"], data.iloc[normal_index][feature], c=colors[0], s=1)
    f2 = plt.scatter(data.iloc[abnormal_index]["攻角"], data.iloc[abnormal_index][feature], c=colors[1], s=1)
    plt.grid(linestyle="-.", linewidth=0.4)
    if y_pred is not None:
        pred_abnormal_index = [i for i, x in enumerate(list(y_pred)) if x == -1] #预测的异常样本的索引
        print(data)
        print(pred_abnormal_index)

        f3 = plt.scatter(data.ix[pred_abnormal_index, "攻角"], data.ix[pred_abnormal_index, feature], c=colors[2], marker="^")
        # exit()
        plt.legend([f1, f2, f3], ["normal", "abnormal", "pred_abnormal"], fontsize=4)
    else:
        plt.legend([f1, f2], ["Normal", "Abnormal"], fontsize=4)
    plt.xticks(np.arange(-4, 12, 2), fontsize=4)
    # plt.xticklabels(np.arange(-4,12,2), fontdict={"fontsize": 4})
    plt.xlabel("Angle of attack", fontdict={"fontsize":4})
    if feature == "实际MA":
        name = "Actual Mach number"
    else:
        name = "Actual roate speed"
    y_min = np.min(data["实际MA"]) - 0.001
    y_max = np.max(data["实际MA"]) + 0.001
    plt.ylim(y_min, y_max)
    # plt.yticks(np.arange(0.897, 0.922+0.006, 0.006), fontsize=4)
    plt.ylabel(name, fontdict={"fontsize":4})
    plt.show()

def t_test(data, feature, confidence_interval):
    """
    对data的feature进行t值检验，将不在confidence_interval之内的
    判为异常值
    :param data:
    :param feature:
    :param confidence_interval:
    :return:
    """
    mu = np.mean(data_[feature])
    sigma = np.std(data_[feature])
    # data_["t_score"] = (data["实际MA"]-mu) / sigma
    from scipy import stats
    interval = stats.t.interval(confidence_interval, len(data_) - 1, mu, sigma)  # t分布95%置信区间
    index = np.logical_and((data_["实际MA"] >= interval[0]).values, (data_["实际MA"] <= interval[1]))
    data["pred"] = -1
    data.ix[index, "pred"] = 1 # 正常样本
    not_index = [not i for i in index]
    # print(data_.ix[not_index, "label"])
    # exit()
    return data

def isolate_forest_test(data, feature, topk=5):
    """
    使用孤立森林对data中的feature进行采样
    :param data:
    :param feature:
    :return:
    """
    from sklearn.ensemble import IsolationForest
    data_feature = data[[feature]]
    rng = np.random.RandomState(666888)
    clf = IsolationForest(n_estimators=400, max_samples="auto", random_state=rng)
    clf.fit(data_feature)
    anomaly_score = clf.decision_function(data_feature) # 其实返回的是异常得分的相反数

    topk_idx = np.argsort(anomaly_score)[:topk] # 得到topk异常点索引
    print(topk_idx)
    data["pred"] = 1
    data["score"] = 0
    data.ix[topk_idx, "pred"] = -1
    data.ix[topk_idx, "score"] = 1
    return data

def get_anomaly_score_prophet(forecast, threshold=0.001):
    """获取prophet预测的异常得分"""
    forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    # anomaly score
    forecasted['score'] = 0
    forecasted.loc[forecasted['anomaly'] == 1, 'score'] = \
        (forecasted['fact'] - forecasted['yhat_upper']) / np.abs(forecast['fact']-forecast['yhat'])
    forecasted.loc[forecasted['anomaly'] == -1, 'score'] = \
        (forecasted['yhat_lower'] - forecasted['fact']) / np.abs(forecast['fact']-forecast['yhat'])

    forecasted.ix[forecasted['anomaly'] != 0, 'anomaly'] = -1 # 异常
    forecasted.ix[forecasted['anomaly']==0, 'anomaly'] = 1 # 正常

    return forecasted

mpl.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 200 #分辨率

abnormal_filenames = ["5_0.1.csv", "6_0.2.csv", "7_0.2.csv", "8_0.2.csv", "9_0.1.csv", "10_0.2.csv"]
data = pd.read_csv("./data/origin/" + abnormal_filenames[2], encoding="gbk")
cols = data.columns
# print(cols)
# exit()
data_ = data[["攻角","实际MA","label"]]

# t值检验
# data_ = t_test(data_, "实际MA", 0.96)

# 孤立森林
print(data_[data_["label"]==-1])
data_ = isolate_forest_test(data_, "实际MA", topk=8)

data1 = data_.ix[data_["pred"]==1, ["实际MA"]]

index_dict = {k:v for k, v in zip(range(data1.shape[0]), data1.index)}

data1["ds"] = pd.date_range(start="2019-10-1", periods=data1.values.shape[0], freq="H")
data1.rename(columns={"实际MA":"y"}, inplace=True)
np.random.seed(2020)
forecast, model = fit_predict_model(data1,
                                    interval_width = 0.95,
                                    changepoint_range = 1,
                                    changepoint_prior_scale=0.05,
                                    daily_seasonality=True,
                                    n_changepoints=45)

cols = forecast.columns
cols = [i for i in cols if i != "cap"]
forecast = forecast[cols]
model.plot(forecast, xlabel='Angle of attack', ylabel='Actual Mach number')
plt.xticks([])
# plt.yticks([round(i,3) for i in np.arange(0.998, 1.005+0.001, 0.001)], [round(i,3) for i in np.arange(0.998, 1.005+0.001, 0.001)])
# plt.yticks(np.arange(0.898, 0.903, 0.001), np.arange(0.898, 0.904, 0.001))
plt.legend(["True", "Pred", "95% confidence interval"])
plt.show()

# 得到prophet的预测结果
forecasted = get_anomaly_score_prophet(forecast)

abnormal_index = forecasted[forecasted["anomaly"]==-1].index
origin_abnormal_index = [index_dict[i] for i in abnormal_index]
data_.ix[origin_abnormal_index, "pred"] = -1

# 对anomaly_score进行归一化处理
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
# mm = MinMaxScaler()
# anomaly_score = mm.fit_transform(forecasted["score"])



# data_.ix[abnormal_index, "score"] = forecasted.ix[abnormal_index, 'score']
# print(forecasted.ix[abnormal_index, 'score'])
# print(data_.ix[data_["score"]!=0, "score"])
# exit()
# anomaly_score = forecasted["score"]
# anomaly_score = anomaly_score.reshape(-1,1)


fpr, tpr, thresholds = roc_curve(data_["label"], data_["pred"])
roc_auc = auc(fpr, tpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, lw=1)
plt.show()





from sklearn.metrics import precision_score, recall_score, roc_auc_score
print(precision_score(data_['label'], data_['pred'], pos_label=-1))
print(recall_score(data_['label'], data_['pred'], pos_label=-1))
# print(roc_auc_score(data_['label'], roc_auc_score(data_['label'], data_["pred"]))) # decision_function值越小越异常,所以取负号)) # decision_function值越小越异常,所以取负号