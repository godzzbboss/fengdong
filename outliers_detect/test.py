# -*- coding: utf-8 -*-

"""
__author__ == "BigBrother"

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
    给数据打标签，-1表示异常，1表示正常
"""
# origin_data_path = "../data_processed/1027_16/origin_data/"
# outlier_data_path = "../data_processed/1027_16/outlier_data/"
#
# origin_filenames = os.listdir(origin_data_path)
# outlier_filenames = os.listdir(outlier_data_path)
#
# # print(len(origin_filenames) == len(origin_filenames))
# for filename in origin_filenames:
#     origin_data = pd.read_csv(origin_data_path + filename, encoding="gbk")
#     outlier_data = pd.read_csv(outlier_data_path + filename, encoding="gbk")
#
#     origin_data_ = origin_data[["攻角", "实际MA", "实际转速"]]
#     outlier_data_ = outlier_data[["攻角", "实际MA", "实际转速"]]
#     origin_data_["label"] = 1
#
#     # 添加标签
#     for i in range(origin_data_.shape[0]):
#         # print(origin_data_.iloc[i]["攻角"])
#         # print(outlier_data_["攻角"])
#         # print(origin_data_.iloc[i]["攻角"] not in list(outlier_data_["攻角"]))
#         # exit()
#         if origin_data_.loc[i]["攻角"] in list(outlier_data_["攻角"]):
#             origin_data_.loc[i:i+1]["label"] = -1
#     # plt.scatter(origin_data_["攻角"], origin_data_["实际转速"])
#     # plt.show()
#     origin_data_.to_csv("./data/" + filename, index=False)


"""使用孤立森林进行异常检测"""
from sklearn.metrics import precision_score, recall_score, roc_auc_score
# def plt_scatter():
#     pass
#

# def get_P_R_AUC(true_y, pred_y):
#     """计算P,R,AUC"""





rootpath = "./data/"
filenames = os.listdir(rootpath)
from sklearn.ensemble import IsolationForest
rng = np.random.RandomState(172124)
clf = IsolationForest(n_estimators=200, max_samples=200, random_state=rng)

data = pd.read_csv(rootpath + filenames[5])
cols = data.columns
data_X = data[cols[:3]]
data_y = data[cols[-1]].values.reshape(-1, 1)
clf.fit(data_X)
pred_y = clf.predict(data_X) # -1表示异常，1表示正常，返回的是一个样本是否为正常样本
pred_y = pred_y.reshape(-1, 1)
# print(pred_y)
# print(data_y)
print(precision_score(data_y.ravel(), pred_y.ravel(), pos_label=-1))
print(recall_score(data_y.ravel(), pred_y.ravel(), pos_label=-1))
data_y_ = []
for i in data_y:
    if i[0] == -1:
        data_y_.append(1) # 异常样本
    else:
        data_y_.append(0) # 正常样本
print(roc_auc_score(data_y_, -clf.decision_function(data_X))) # decision_function值越小越异常,所以取负号



