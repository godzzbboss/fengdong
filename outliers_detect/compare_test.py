# -*- coding: utf-8 -*-

"""
__author__ == "BigBrother"

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import iforest


# ============与粗细粒度结合的异常检测算法进行对比===============

abnormal_filenames = ["5_0.1.csv", "6_0.2.csv", "7_0.2.csv", "8_0.2.csv", "9_0.1.csv", "10_0.2.csv"]
data = pd.read_csv("./data/origin/" + abnormal_filenames[5], encoding="gbk")
cols = data.columns
data_ = data[["攻角","实际MA","label"]]


# ================== 孤立森林预测 ==============================
# from sklearn.ensemble import IsolationForest
# feature = "实际MA"
# data_feature = data_[[feature]]
# rng = np.random.RandomState(666888)
# clf = IsolationForest(n_estimators=1000, max_samples="auto")
# clf.fit(data_feature)
# anomaly_score = clf.decision_function(data_feature) # 其实返回的是异常得分的相反数
# topk = 18
# topk_idx = np.argsort(anomaly_score)[:topk] # 得到topk异常点索引
# data_["pred"] = 1
# data_.ix[topk_idx, "pred"] = -1

# ====================== LOF ==================================
# from sklearn.neighbors import LocalOutlierFactor as LOF
# feature = "实际MA"
# data_feature = data_[[feature]]
# clf = LOF(n_neighbors=20, contamination=0.05)
# clf.fit_predict(data_feature)
# neg_lof = clf.negative_outlier_factor_
# topk = 8
# topk_idx = np.argsort(neg_lof)[:topk]
# data_["pred"] = 1
# data_.ix[topk_idx, "pred"] = -1

# ==================== MAD =====================================
""" m1 = M({x_i| i=1,...n})
    m2 = M({|x_i-m1|, i=1,...n})
    MAD = b * m2, for normal distribution, b=1.4826
    
    For normal sample x_i, median - 3 * MAD <= x_i <= median + 3 * MAD  

"""
feature = "实际MA"
data_feature = data_[[feature]]
m1 = np.median(data_feature)
m2 = np.median([np.abs(i - m1) for i in data_feature.values])
mad = 1.4826 * m2
data_["pred"] = 1
print(max(data_feature.values), min(data_feature.values))

idx = np.logical_or(data_["实际MA"] < (m1 - 2 * mad), data_["实际MA"] > (m1 + 2 * mad))
data_.ix[idx, "pred"] = -1

# 评估
from sklearn.metrics import roc_auc_score, precision_score, recall_score

print(precision_score(data_['label'], data_["pred"], pos_label=-1))
print(recall_score(data_['label'], data_["pred"], pos_label=-1))
print(roc_auc_score(data_['label'], data_["pred"]))
