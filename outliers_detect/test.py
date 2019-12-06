# -*- coding: utf-8 -*-

"""
__author__ == "BigBrother"

"""
import os
import pandas as pd
import numpy as np
from tkinter import Tk
import matplotlib.pyplot as plt
from matplotlib import cm, colors

import matplotlib as mpl

def set_size(fig=None, w=None, h=None, w_mul=1.0, h_mul=1.0):
    """设置show的图片的尺寸"""
    if fig is None:
        fig = plt.gcf()
    # figManager = fig.canvas.manager.window
    figManager = plt.get_current_fig_manager().window
    if w is None:
        w = Tk().winfo_screenwidth()
    if h is None:
        h = Tk().winfo_screenheight()
        # print( '\n{}\n{}\n'.format(w, h))
    w_ = w * w_mul
    h_ = h * h_mul
    figManager.setGeometry(w/2,0,w_,h_) # (0,0)表示图片左上角起始位置

"""
    给数据打标签，-1表示异常，1表示正常
"""
# origin_data_path = "../data_processed/1027_16/origin_data/"
# outlier_data_path = "../data_processed/1027_16/outlier_data/"
#
# origin_filenames = os.listdir(origin_data_path)
# outlier_filenames = os.listdir(outlier_data_path)
#
# # print(origin_filenames)
# # exit()
# # print(len(origin_filenames) == len(origin_filenames))
# cmap = {1:"blue", -1:"red"}
# for filename in origin_filenames:
#     try:
#         with open(origin_data_path + filename) as f1:
#             origin_data = pd.read_csv(f1)
#         with open(outlier_data_path + filename) as f2:
#             outlier_data = pd.read_csv(f2)
#         origin_data_ = origin_data[["攻角", "MA", "实际MA", "实际转速", "角速度"]]
#         outlier_data_ = outlier_data[["攻角", "MA", "实际MA", "实际转速", "角速度"]]
#         origin_data_["label"] = 1
#         # print(origin_data_)
#         # exit()
#         # 添加标签
#         for i in range(origin_data_.shape[0]):
#             # print(origin_data_.iloc[i]["攻角"])
#             # print(outlier_data_["攻角"])
#             # print(origin_data_.iloc[i]["攻角"] not in list(outlier_data_["攻角"]))
#             if origin_data_.loc[i]["攻角"] in list(outlier_data_["攻角"]):
#                 origin_data_.loc[i:i]["label"] = -1
#         # cs = [cmap[i] if i == 1 else cmap[i] for i in origin_data_["label"]]
#         # plt.scatter(origin_data_["攻角"], origin_data_["实际MA"], c=cs)
#         # plt.show()
#         origin_data_.to_csv("./data/origin/" + filename, index=False, encoding="gbk")
#     except Exception as e:
#         print(e)
# exit()


"""
    正常样本与异常样本散点图

"""
mpl.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 550 #分辨率
def plt_scatter(data_X, y_true, y_pred=None, name="实际MA", fname=None, save=False):
    """使用孤立森林进行异常检测"""
    colors = ["blue", "#e6550d"]
    # color = [colors[0] if i==1 else colors[1] for i in y_true]
    fig = plt.figure(figsize=(6,8), tight_layout=True)
    set_size(fig, w_mul=0.6)
    # 获取正常样本的索引
    normal_index = [i for i, x in enumerate(y_true) if x == 1]
    abnormal_index = [i for i in range(len(y_true)) if i not in normal_index]

    f1 = plt.scatter(data_X.iloc[normal_index]["攻角"], data_X.iloc[normal_index][name], c=colors[0], s=1)
    f2 = plt.scatter(data_X.iloc[abnormal_index]["攻角"], data_X.iloc[abnormal_index][name], c=colors[1], s=1)
    plt.grid(linestyle="-.", linewidth=0.4)
    if y_pred is not None:
        pred_abnormal_index = [i for i, x in enumerate(list(y_pred.ravel())) if x == -1] #预测的异常样本的索引
        f3 = plt.scatter(data_X.iloc[pred_abnormal_index]["攻角"], data_X.iloc[pred_abnormal_index][name], c=colors[2], marker="^")
        plt.legend([f1, f2, f3], ["normal", "abnormal", "pred_abnormal"], fontsize=4)
    else:
        plt.legend([f1, f2], ["Normal", "Abnormal"], fontsize=4)
    plt.xticks(np.arange(-4, 12, 2), fontsize=4)
    # plt.xticklabels(np.arange(-4,12,2), fontdict={"fontsize": 4})
    plt.xlabel("Angle of attack", fontdict={"fontsize":4})
    if name == "实际MA":
        name = "Actual Mach number"
    else:
        name = "Actual roate speed"
    y_min = np.min(data["实际MA"]) - 0.001
    y_max = np.max(data["实际MA"]) + 0.001
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(0.696, 0.716, 0.005), fontsize=4)
    plt.ylabel(name, fontdict={"fontsize":4})
    save_fig_name = fname[:-4] + "_" + name + ".png"
    if save:
        plt.savefig("./figures/" + save_fig_name)
    plt.show()

rootpath = "./data/origin/"
filenames = os.listdir(rootpath)
abnormal_filenames = ["5_0.1.csv", "6_0.2.csv", "7_0.2.csv", "8_0.2.csv", "9_0.1.csv", "10_0.2.csv"]
data = pd.read_csv(rootpath + abnormal_filenames[0], encoding="gbk")
cols = data.columns
data_X = data[cols[:4]]
data_y = data[cols[-1]].values.reshape(-1, 1)
plt_scatter(data_X, data_y.ravel(), name="实际MA", fname=abnormal_filenames[0])

"""
包含异常数据文件的实际MA直方图

"""
# mpl.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['ytick.direction'] = 'in'
# # plt.rcParams['figure.dpi'] = 550 #分辨率
#
# rootpath = "./data/origin/"
# filenames = os.listdir(rootpath)
# abnormal_filenames = ["5_0.1.csv", "6_0.2.csv", "7_0.2.csv", "8_0.2.csv", "9_0.1.csv", "10_0.2.csv"]
# data = pd.read_csv(rootpath + abnormal_filenames[0], encoding="gbk")
#
# data_ma = data["实际MA"]
# fig = plt.figure(figsize=(8,6))
# # set_size(fig, w_mul=0.6)
# plt.hist(data_ma, bins=40, histtype='bar', rwidth=0.8)
# plt.show()



"""
    使用孤立森林进行异常检测

"""
# from sklearn.metrics import precision_score, recall_score, roc_auc_score
# from sklearn.ensemble import IsolationForest
#
# def help1(item):
#     if item > 0:
#         return 1
#     else:
#         return -1
#
# def help2(item):
#     if -1 in item.ravel():
#         return -1
#     else:
#         return 1
#
# rootpath = "./data/origin/"
# filenames = os.listdir(rootpath)
# abnormal_filenames = ["5_0.1.csv", "6_0.2.csv", "7_0.2.csv", "8_0.2.csv", "9_0.1.csv", "10_0.2.csv"]
# data = pd.read_csv(rootpath + abnormal_filenames[5])
# cols = data.columns
# data_X = data[cols[:3]]
# data_y = data[cols[-1]].values.reshape(-1, 1)
#
# # rand_seeds = [172124, 66 , 88, 2019, 17, 21, 24]
# rand_seeds = [172124]
# final_res = None
# for seed in rand_seeds:
#     rng = np.random.RandomState(seed)
#     clf = IsolationForest(n_estimators=400, max_samples=400, random_state=rng)
#     clf.fit(data_X)
#     pred_y = clf.predict(data_X) # -1表示异常，1表示正常，返回的是一个样本是否为正常样本
#     pred_y = pd.Series(pred_y.reshape(-1, 1).ravel())
#     final_res = pd.concat([final_res, pred_y], axis=1)
# print(final_res)
# exit()
# #
# # 投票集成
# res = final_res.sum(axis=1)
# res = res.map(lambda x: help1(x))
# #
# # # 只要有一个预测为异常则为异常
# # res = final_res.apply(lambda x: help2(x), axis=1)
# #
# # plt_scatter(data_X, data_y, res, "实际转速", abnormal_filenames[5])
# # # print(pred_y)
# # # print(data_y)
# print(precision_score(data_y.ravel(), res.ravel(), pos_label=-1))
# print(recall_score(data_y.ravel(), res.ravel(), pos_label=-1))
# data_y_ = []
# for i in data_y:
#     if i[0] == -1:
#         data_y_.append(1) # 异常样本
#     else:
#         data_y_.append(0) # 正常样本
# print(roc_auc_score(data_y_, -clf.decision_function(data_X))) # decision_function值越小越异常,所以取负号



