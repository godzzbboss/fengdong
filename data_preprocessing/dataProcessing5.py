# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from scipy.io import loadmat

import os
import pandas as pd

# data = pd.read_csv("../data_processed/1027_11/data.csv", encoding="gbk")
# speed = data["实际转速"]
# plt.hist(speed, 20, edgecolor="black")
# plt.show()
# exit()

# 正常显示中文
mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False

colors = cm.Set1.colors

filepath = "../results/10/"
storepath = "../plot_figures/results/"
filenames = os.listdir(filepath)


# f_list = []
# for i, filename in enumerate(filenames):
#     print(filename)
#     pred_y = np.array(loadmat(filepath+filename)["y_mean"])
#     plt.plot(np.linspace(-4, 10, 29), pred_y, label=i)
#     # f_list.append(f1)
# plt.legend(["0.1","0.2"])
# plt.savefig(storepath+"9.png")
# plt.show()
#
# exit()


fig = plt.figure(figsize=(10, 8), tight_layout=True)
count = 1
for i in np.linspace(0.7, 0.79, 10):
    y_name = str(round(i, 2)) + "_y.mat"
    s2_name = str(round(i, 2)) + "_s2.mat"
    ax = fig.add_subplot(2,5,count)
    s2 = np.array(loadmat(filepath+s2_name)["y_s21"])
    pred_y = np.array(loadmat(filepath+y_name)["y_mean1"])
    ax.plot(np.linspace(-4, 10, 29), pred_y, color=colors[0], alpha=0.6)
    ax.fill_between(np.linspace(-4, 10, 29), (pred_y - 2 * np.sqrt(s2)).ravel(), (pred_y + 2 * np.sqrt(s2)).ravel(),
                 color=colors[2], alpha=0.5)
    count += 1
    # ax.set_ylim([1350,1750])
    ax.set_xlim([-5, 11])
    ax.set_title(str(round(i, 2))+"MA_"+str(0.2))
plt.show()
exit()
# fig.savefig(storepath+"14.png")
# exit()





# s2 = np.array(loadmat(filepath+"0_0.2_s2")["y_s21"])
# pred_y = np.array(loadmat(filepath+"0_0.2_y")["y_mean1"])

s2 = np.array(loadmat(filepath+"pred_test_s2")["y_s21"])
pred_y = np.array(loadmat(filepath+"pred_test_y")["y_mean1"])

s21 = np.array(loadmat(filepath+"pred_test_s21")["y_s21"])
pred_y1 = np.array(loadmat(filepath+"pred_test_y1")["y_mean1"])

# true_y = np.array(loadmat("./test_y.mat")["test_y"])
# f1 = plt.plot(np.linspace(0.6,1.5,181), pred_y, color=colors[0], alpha=0.6)
f2 = plt.plot(np.linspace(0.6,1.5,181), pred_y1, color=colors[1], alpha=0.5)
# f2, = plt.plot(true_y,color="r", alpha=0.6)

# plt.fill_between(np.linspace(0.6,1.5,181), (pred_y - 2 * np.sqrt(s2)).ravel(), (pred_y + 2 * np.sqrt(s2)).ravel(),
#                  color=colors[2], alpha=0.5)

plt.fill_between(np.linspace(0.6,1.5,181), (pred_y1 - 2 * np.sqrt(s21)).ravel(), (pred_y1 + 2 * np.sqrt(s21)).ravel(),
                 color=colors[3], alpha=0.4)
# plt.fill(np.linspace(1,len(pred_y),len(pred_y)), (pred_y-2*np.sqrt(s2)).ravel(),  color="b", alpha=0.1)
# plt.legend([f1,f2],["pred_y","true_y"])
# plt.grid(linestyle="-.")
# plt.savefig("../results/no_under_sample_result.png")
plt.title("角速度:0.2 攻角：0")
plt.xticks(list(np.linspace(0.6,1.5,10)))
# plt.savefig(storepath+"16.png")
plt.show()

