# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
from scipy.io import loadmat
from matplotlib import cm

mas = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]

# """
#     从原始数据中找到需要的数据
# """
# filepath = "../data_processed/1027_0/"
# filenames = os.listdir(filepath)
# count = 0
# for filename in filenames:
#     with open(filepath+filename) as f:
#         data = pd.read_csv(f)
#     if data["MA"][0] in mas:
#         print(filename)
#         count+=1
# print(count)
# exit()

# """
#     可视化原始数据, 将16组数据画成一张图
# """
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# plt.rcParams['figure.dpi'] = 150 #分辨率
#
# filepath1 = "../data_processed/1027_16/cleaned_data/"
# storepath = "./figures/"
# filenames = os.listdir(filepath1)
# fig = plt.figure(figsize=(8, 6), tight_layout=True)
# i = 1
# for filename in filenames:
#     # aoa_speed = filename.split("_")[1][:3]
#     ax = fig.add_subplot(4, 4, i)
#     with open(filepath1 + filename) as f:
#         data = pd.read_csv(f)
#         # data["角速度"] = aoa_speed
#         # data = data.round({"MA": 5, "实际MA": 5, "攻角": 3, "实际转速": 3, "目标转速": 3, "截面积": 4})
#         # data.to_csv(filepath1+filename, index=False)
#     # print(data)
#     # exit()
#     print(filename)
#     ax.scatter(data["攻角"], data["实际转速"], s=0.7, c="#104E8B")
#     ax.set_title(str(data["MA"][0]) + " MA-" + str(data["角速度"][0]) + "°/sec", fontdict={"fontsize": 6})
#     ax.set_xlim(-6, 12)
#     # ax.axhline(y=data["MA"][0] + 0.001, linewidth=0.8, linestyle="--")
#     # ax.axhline(y=data["MA"][0] - 0.001, linewidth=0.8, linestyle="--")
#     # ax.set_xticks([int(i) for i in np.linspace(-6, 12, 6).tolist()])
#     # ax.set_yticks([np.round(i,3) for i in np.linspace(np.min(data["实际MA"])-0.001, np.max(data["实际MA"])+0.001, 8).tolist()])
#     ax.tick_params(labelsize=7, labelcolor="#000000")
#
#     if i % 4 == 1:
#         ax.set_xlabel("攻角", fontdict={"fontsize": 6})
#         ax.set_ylabel("实时转速", fontdict={"fontsize": 6})
#     i += 1
# # plt.savefig(storepath+"origin_aoa_ma.png")
# plt.show()

# """
#     只可视化存在异常点的数据
#
# """
# mpl.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# plt.rcParams['figure.dpi'] = 350 #分辨率
#
# filepath1 = "../data_processed/1027_16/cleaned_data/"
# storepath = "./figures/"
# filenames = ["5_0.1.csv","6_0.2.csv","7_0.2.csv","8_0.2.csv","9_0.1.csv","10_0.2.csv"]
# fig = plt.figure(figsize=(15, 10), tight_layout=True)
# i = 1
# for filename in filenames:
#     # aoa_speed = filename.split("_")[1][:3]
#     ax = fig.add_subplot(2, 3, i)
#     with open(filepath1 + filename) as f:
#         data = pd.read_csv(f)
#         # data["角速度"] = aoa_speed
#         # data = data.round({"MA": 5, "实际MA": 5, "攻角": 3, "实际转速": 3, "目标转速": 3, "截面积": 4})
#         # data.to_csv(filepath1+filename, index=False)
#     # print(data)
#     # exit()
#     # # 只保留攻角-4~10内的数据
#     # index = np.logical_and(data["攻角"]>=-4,data["攻角"]<=10)
#     # data = data[index]
#     # print(data)
#     # exit()
#     print(filename)
#     ax.scatter(data["攻角"], data["实际MA"], s=0.7, c="#104E8B")
#     ax.set_title(str(data["MA"][0]) + " MA-" + str(data["角速度"][0]) + "°/s", fontdict={"fontsize": 8})
#     ax.set_xlim(-4, 10)
#     ax.axhline(y=data["MA"][0] + 0.001, linewidth=0.8, linestyle="--")
#     ax.axhline(y=data["MA"][0] - 0.001, linewidth=0.8, linestyle="--")
#     ax.set_xticks([int(i) for i in np.linspace(-6, 12, 10).tolist()])
#     ax.set_yticks([np.round(i,3) for i in np.linspace(np.min(data["实际MA"])-0.001, np.max(data["实际MA"])+0.001, 8).tolist()])
#     ax.tick_params(labelsize=6, labelcolor="#000000")
#
#     if i % 3 == 1:
#         ax.set_xlabel("Angle of attack", fontdict={"fontsize": 8})
#         ax.set_ylabel("Actual Mach number", fontdict={"fontsize": 8})
#     i += 1
# # plt.savefig(storepath+"cleaned_aoa_ma_error.tif",dpi=1000)
# plt.show()


# """
#     测试结果的置信区间
#
# """
# mpl.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# plt.rcParams['figure.dpi'] = 350  # 分辨率
#
# filepath2 = "../data_processed/1027_16/test_results/"
# colors = cm.Set1.colors
# fig = plt.figure(figsize=(10, 8), tight_layout=True)
# count = 1
# for i in np.linspace(0.9, 0.99, 10):
#     y_name = str(round(i, 2)) + "_y.mat"
#     s2_name = str(round(i, 2)) + "_s2.mat"
#     ax = fig.add_subplot(2, 5, count)
#     s2 = np.array([i / 1000000 for i in list(np.array(loadmat(filepath2 + s2_name)["y_s21"]).ravel())]).reshape(-1, 1)
#     pred_y = np.array([i / 1000 for i in np.array(loadmat(filepath2 + y_name)["y_mean1"]).ravel()]).reshape(-1, 1)
#
#     ax.plot(np.linspace(-4, 10, 29), pred_y, color="#436EEE", alpha=0.9, LineWidth=1.5)
#     ax.fill_between(np.linspace(-4, 10, 29), (pred_y - 2 * np.sqrt(s2)).ravel(), (pred_y + 2 * np.sqrt(s2)).ravel(),
#                     color="#BCD2EE", alpha=0.5)
#     ax.set_ylim([1.920, 2.100])
#     ax.set_xlim([-5, 11])
#     ax.set_xticks([int(i) for i in np.linspace(-4, 10, 5)])
#     # ax.set_xticks([int(i) for i in np.linspace(-4, 10, 5)])
#     ax.set_title(str(round(i, 2)) + "MA", fontdict={"fontsize": 6})
#     if count % 5 == 1:
#         ax.set_xlabel("Angle of attack", fontdict={"fontsize": 8})
#         ax.set_ylabel("Rotate speed/" + "$10^{3}$", fontdict={"fontsize": 8})
#     count += 1
# plt.show()
# exit()

"""
    0.9马赫数条件下测试

"""
mpl.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 350  # 分辨率

filepath = "../data_processed/1027_16/0.9_results/"


fig = plt.figure(figsize=(10,8), tight_layout=True)

data_origin = np.array(loadmat(filepath + "data_origin.mat")["data_test_origin"])
data_5 = np.array(loadmat(filepath+"data_5.mat")["data_9_2"]) # 以5为间隔采样的的数据
data_10 = np.array(loadmat(filepath+"data_10.mat")["data_9_1"]) # 以10为间隔采样的的数据
aoa_diff = list(set(data_5[:,0]).difference(set(data_10[:,0])))
data_diff = data_5[[data_5[i,0] in aoa_diff for i in range(data_5.shape[0])],:]


y_mean_5 = np.array(loadmat(filepath+"y_mean_test_5.mat")["y_mean_test"])
y_mean_10 = np.array(loadmat(filepath+"y_mean_test_10.mat")["y_mean_test"])

y_s2_5 = np.array(loadmat(filepath+"y_s2_test_5.mat")["y_s2_test"])
y_s2_10 = np.array(loadmat(filepath+"y_s2_test_10.mat")["y_s2_test"])

ax1 =  fig.add_subplot(1,2,1)
ax1.plot(data_origin[:,0], y_mean_10/1000, c="#377eb8", alpha=1, zorder=1, label="Pred")
ax1.plot(data_origin[:,0], data_origin[:,3]/1000, c= "#f03b20", alpha=0.7, zorder=1, label="True")
ax1.fill_between(data_origin[:,0], (y_mean_10/1000 - 2 * np.sqrt(y_s2_10/1000000)).ravel(), (y_mean_10/1000 + 2 * np.sqrt(y_s2_10/1000000)).ravel(),
                    color="#a6cee3", alpha=0.5)
ax1.scatter(data_10[:,0], data_10[:,5]/1000, s=18, c="#1a9850", marker=".", zorder=2, label="Origin data")
ax1.set_xticks(np.arange(-5,12,2))
ax1.set_yticks([i/1000 for i in np.arange(1970,2050,10)])
ax1.set_xlabel("Angle of attack", fontdict={"fontsize":8})
ax1.set_ylabel("Rotate speed/" + "$10^{3}$", fontdict={"fontsize": 8})
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles,labels, fontsize="xx-small", loc="upper left")


ax2 =  fig.add_subplot(1,2,2)
ax2.plot(data_origin[:,0], y_mean_5/1000, c="#377eb8", alpha=1,zorder=1, label="Pred")
ax2.plot(data_origin[:,0], data_origin[:,3]/1000, c= "#f03b20", alpha=0.7, zorder=1, label="True")
ax2.fill_between(data_origin[:,0], (y_mean_5/1000 - 2 * np.sqrt(y_s2_5/1000000)).ravel(), (y_mean_5/1000 + 2 * np.sqrt(y_s2_5/1000000)).ravel(),
                    color="#a6cee3", alpha=0.5)
ax2.scatter(data_10[:,0], data_10[:,5]/1000, s=18, c="#1a9850", marker=".", zorder=2, label="Origin data")
ax2.scatter(data_diff[:,0], data_diff[:,5]/1000, s=18, c="#756bb1", marker=".", zorder=2, label="Joined data")

ax2.set_xticks(np.arange(-5,12,2))
ax2.set_yticks([i/1000 for i in np.arange(1970,2050,10)])

handles_, labels_ = ax2.get_legend_handles_labels()
ax2.legend(handles_, labels_, fontsize="xx-small", loc="upper left")
# ax2.set_xlabel("Angle of attack", fontdict={"fontsize":8})
# ax2.set_ylabel("Rotate speed/" + "$10^{3}$", fontdict={"fontsize": 8})
plt.show()


# """
#     处理后的完整数据集的特征分布图，共2578条数据
# """
# mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# plt.rcParams['figure.dpi'] = 350 #分辨率
# filepath = "../data_processed/1027_16/data11.csv"
# with open(filepath) as f:
#     data = np.loadtxt(f, delimiter=",")
#
# # 攻角
# aoa = data[:,0]
# actual_ma = data[:,2]
# area = data[:,4]
#
# fig  = plt.figure(figsize=(9,4), tight_layout=True)
#
# ax1 = fig.add_subplot(1,3,1)
# ax1.hist(aoa, color="#0504aa", bins=15, rwidth=0.8, alpha=0.9)
# ax1.set_xlabel("Angle of attack", fontdict={"fontsize":8})
# ax1.set_ylabel("Number of samples", fontdict={"fontsize":8})
# ax1.tick_params(labelsize=6, labelcolor="#000000")
# ax1.set_xticks(np.linspace(-5,11,5).tolist())
#
# ax2 = fig.add_subplot(1,3,2)
# ax2.hist(actual_ma, color="#0504aa", bins=15, rwidth=0.8, alpha=0.9)
# ax2.tick_params(labelsize=6, labelcolor="#000000")
# ax2.set_xticks([0.6,0.7,0.8,0.9,1,1.1])
# ax2.set_xlabel("Actual mach number", fontdict={"fontsize":8})
#
# ax3 = fig.add_subplot(1,3,3)
# ax3.hist(area, color="#0504aa", bins=15, rwidth=0.8, alpha=0.9)
# ax3.tick_params(labelsize=8, labelcolor="#000000")
# ax3.set_xticks([1,1.16])
# ax3.set_xlabel("Test section area", fontdict={"fontsize":8})
# plt.show()


# """
#     5折交叉验证的绝对误差
# """
#
# mpl.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# mpl.rcParams['figure.dpi'] = 350 #分辨率
#
# filepath = "../data_processed/1027_16/5_fold/abs_error.csv"
# with open(filepath) as f:
#     data = np.loadtxt(f, delimiter=",")
#
# error1 = data[:,4]
# error2 = data[:,3]
# error3 = data[:,2]
# error4 = data[:,1]
# error5 = data[:,0]
#
# fig = plt.figure(figsize=(15,10), tight_layout=True)
#
# ax1 = fig.add_subplot(2,3,1)
# ax1.hist(error1, color="#0504aa", bins=30, rwidth=0.8, alpha=0.9)
# ax1.set_xlabel("Absolute error", fontdict={"fontsize":8})
# ax1.set_ylabel("Number of samples", fontdict={"fontsize":8})
# ax1.tick_params(labelsize=6, labelcolor="#000000")
# ax1.set_title("First cross validation",fontdict={"fontsize":6})
# ax1.set_xticks(np.arange(-4,5,1))
#
# ax2 = fig.add_subplot(2,3,2)
# ax2.hist(error1, color="#0504aa", bins=30, rwidth=0.8, alpha=0.9)
# ax2.tick_params(labelsize=6, labelcolor="#000000")
# ax2.set_title("Second cross validation",fontdict={"fontsize":6})
# ax2.set_xticks(np.arange(-4,5,1))
#
# ax3 = fig.add_subplot(2,3,3)
# ax3.hist(error1, color="#0504aa", bins=30, rwidth=0.8, alpha=0.9)
# ax3.tick_params(labelsize=6, labelcolor="#000000")
# ax3.set_title("Third cross validation",fontdict={"fontsize":6})
# ax3.set_xticks(np.arange(-4,5,1))
#
# ax4 = fig.add_subplot(2,3,4)
# ax4.hist(error1, color="#0504aa", bins=30, rwidth=0.8, alpha=0.9)
# ax4.set_xlabel("Absolute error", fontdict={"fontsize":8})
# ax4.set_ylabel("Number of samples", fontdict={"fontsize":8})
# ax4.tick_params(labelsize=6, labelcolor="#000000")
# ax4.set_title("Fourth cross validation",fontdict={"fontsize":6})
# ax4.set_xticks(np.arange(-4,5,1))
#
# ax5 = fig.add_subplot(2,3,5)
# ax5.hist(error1, color="#0504aa", bins=30, rwidth=0.8, alpha=0.9)
# ax5.tick_params(labelsize=6, labelcolor="#000000")
# ax5.set_title("Fifth cross validation",fontdict={"fontsize":6})
# ax5.set_xticks(np.arange(-4,5,1))
#
# plt.show()

# """
#     分组预测结果
#
# """
# mpl.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# plt.rcParams['figure.dpi'] = 350 #分辨率
#
# filepath = "../data_processed/1027_16/group_test_results/"
# origin_data = np.array(loadmat(filepath+"1.0_origin_data.mat")["sort_data_test_origin"])
# aoa = origin_data[:,0]
# true_y = np.array([i/1000 for i in origin_data[:,3].ravel()]).reshape(-1,1)
#
# add_pred_y = np.array([i/1000 for i in np.array(loadmat(filepath+"1.0_pred_y.mat")["sort_y_mean_test"]).ravel()])
# ard_pred_y = np.array([i/1000 for i in np.array(loadmat(filepath+"ard_1.0_pred_y.mat")["sort_y_mean_test"]).ravel()])
#
# fig = plt.figure(figsize=(10,6), tight_layout=True)
# ax1 = fig.add_subplot(1,2,1)
# ax1.plot(aoa, true_y, color="r", LineWidth=1.5, label="True_rotate_speed")
# ax1.plot(aoa, add_pred_y, color="#32CD32", LineWidth=1.5, label="Pred_rotate_speed")
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, fontsize="xx-small")
# ax1.set_xlabel("Angle of attack", fontdict={"fontsize":8})
# ax1.set_ylabel("Rotate speed/"+"$10^{3}$", fontdict={"fontsize":8})
# ax1.set_title("1.0MA", fontdict={"fontsize":8})
#
# ax2 = fig.add_subplot(1,2,2)
# ax2.plot(aoa, true_y, color="r", LineWidth=1.5, label="True_rotate_speed")
# ax2.plot(aoa, ard_pred_y, color="#32CD32", LineWidth=1.5, label="Pred_rotate_speed")
# # ax1.set_xlim(-6,12)
# ax2.set_title("1.0MA", fontdict={"fontsize":8})
#
#
# plt.show()
