# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
'''
    对处理好的数据进行可视化
'''
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np
import os
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_all_data(filepath, storepath):
    '''
        按角速度跟截面积对数据进行可视化
    :param filepath:
    :param storepath:
    :return:
    '''
    with open(filepath + 'data_final.csv') as f:
        data = np.loadtxt(f, delimiter=',')
    # with open(filepath + 'data_2.csv') as f:
    #     data_2 = np.loadtxt(f, delimiter=',')
    # with open(filepath + 'data_3.csv') as f:
    #     data_3 = np.loadtxt(f, delimiter=',')
    # with open(filepath + 'data_4.csv') as f:
    #     data_4 = np.loadtxt(f, delimiter=',')

    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    # ax1 = ax.scatter(data_1[:,0], data_1[:,1], data_1[:, 4], s=2)
    # ax2 = ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 4], s=2)

    # ax3 = ax.scatter(data_3[:, 0], data_3[:, 1], data_3[:, 4], s=2)

    ax4 = ax.scatter(data[:, 0], data[:, 1], data[:, 4], s=2)
    # plt.title("角速度0.2，截面积7/6")
    plt.show()

    # plt.legend([ax1,ax2,ax3,ax4],["0.1, 1","0.1, 7/6","0.2, 1","0.2, 7/6"])
    # plt.show()


from scipy.io import loadmat
from matplotlib import colors

def plot_pred_data(filepath1, filepath2):
    """
        可视化预测结果，三维
    :param filepath:
    :return:
    """
    filenames = os.listdir(filepath1)
    print(len(filenames))
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    jet = cm.get_cmap("jet")
    cnorm = colors.Normalize(vmin=0.6, vmax=1.5)
    scalarmap = cm.ScalarMappable(norm=cnorm, cmap=jet)
    for filename in filenames:
        ma_ = float(filename.split("_")[0])
        aoa = np.linspace(-4,10,29)
        ma = np.array([ma_] * 29).reshape(-1,1)
        speed = np.array(loadmat(filepath + filename)["y_mean1"])
        c = scalarmap.to_rgba(ma_)
        ax.scatter(aoa, ma, speed, color=c, label=ma_)
    # 获取角速度为0.1的真实数据
    with open(filepath2 + "data1_final.csv") as f:
        data = np.loadtxt(f, delimiter=",")
    data_ = data[data[:,3]==0.1,:]
    ma1 = data_[:,1]
    aoa1 = data_[:,0]
    speed1 = data_[:,5]
    ax.scatter(aoa1,ma1,speed1, color='b', marker="^", label="true", alpha=0.2, s=15)
    handles, labels = ax.get_legend_handles_labels()
    font1 = {'size': 7}
    plt.legend(handles, labels, ncol=3, prop=font1)
    plt.title("角速度0.2")
    plt.show()


def plot_speed(filepath, storepath):
    """
        绘制不同MA下，连续变攻角下攻角和转速的对应关系
    :return:
    """
    file_names = os.listdir(filepath)
    file_names1 = [name for name in file_names if "0.1" in name]
    file_names2 = [name for name in file_names if "0.2" in name]
    num_f1 = len(file_names1)
    num_f2 = len(file_names2)

    # 0.1角速度的一个图，0.2角速度的一个图
    fig1 = plt.figure(figsize=(10, 8), tight_layout=True, num=1)
    fig2 = plt.figure(figsize=(10, 8), tight_layout=True, num=2)
    gs1 = gridspec.GridSpec(4, 4)
    gs2 = gridspec.GridSpec(3, 4)
    row1, row2 = 0, 0
    col1, col2 = 0, 0
    for file_name in file_names1:
        with open(filepath + file_name) as f:
            data = pd.read_csv(f)
        if row1 == 3:
            ax = fig1.add_subplot(gs1[row1, 1:3])
        else:
            ax = fig1.add_subplot(gs1[row1, col1])
        ax.plot(data["攻角"], data["实际转速"])
        ax.set_title(file_name + "-" + str(data["MA"][0]))
        ax.grid()
        col1 += 1
        if col1 % 3 == 1 and col1 != 1:
            row1 += 1
            col1 = 0

    for file_name in file_names2:
        with open(filepath + file_name) as f:
            data = pd.read_csv(f)

        ax = fig2.add_subplot(gs2[row2, col2])
        ax.plot(data["攻角"], data["实际转速"])
        ax.set_title(file_name + "-" + str(data["MA"][0]))
        ax.grid()
        col2 += 1
        if col2 % 3 == 1 and col2 != 1:
            row2 += 1
            col2 = 0
    return fig1, fig2


def plot_speed2(filedir, speed="实际转速", alpha=1):
    """
        将相同MA下，不同角速度下的攻角-转速对应关系
    :return:
    """
    with open(filedir) as f:
        data = pd.read_csv(f)

    ma1 = sorted(data.ix[data["角速度"] == 0.1]["MA"].unique())
    ma2 = sorted(data.ix[data["角速度"] == 0.2]["MA"].unique())
    ma = [i for i in ma1 if i in ma2]  # ma1与ma2都包含的MA

    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    for i, m in enumerate(ma):
        index1 = np.logical_and(data["MA"] == m, data["角速度"] == 0.1)
        index2 = np.logical_and(data["MA"] == m, data["角速度"] == 0.2)

        data1 = data.ix[index1]
        data2 = data.ix[index2]
        ax = fig.add_subplot(2, 5, i + 1)
        ax.plot(data1["攻角"], data1[speed], "g-", label="0.1", alpha=alpha)
        ax.plot(data2["攻角"], data2[speed], "r-", label="0.2", alpha=alpha)
        ax.set_title(str(m))
        ax.legend(loc=2, fontsize="xx-small")
        ax.grid(linestyle="-.")
    return fig


def get_color_dict(MA, colors):
    """

        产生MA于颜色之间的映射，返回字典，键为MA，值为RGBA颜色

    :param MA: Series
    :return:
    """
    mas = set(list(MA))
    color_dict = {}
    for i, ma in enumerate(mas):
        color_dict[str(ma)] = colors[i]

    return color_dict


def data_plot(filepath):
    """
        对合并的数据进行可视化
    :return:
    """
    file_names = os.listdir(filepath)
    data = []
    for file_name in file_names:
        with open(filepath + file_name) as f:
            data.append(pd.read_csv(f))

    data1 = data[0]
    data2 = data[1]
    data1.sort_values(axis=0, by=["MA", "攻角"], inplace=True)
    data2.sort_values(axis=0, by=["MA", "攻角"], inplace=True)

    fig1 = plt.figure(figsize=(8, 5), tight_layout=True)
    fig2 = plt.figure(figsize=(8, 5), tight_layout=True)

    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    # print(len(data1["实际转速"]))
    # print(len(data2["实际转速"]))
    # print(np.linspace(0,len(data1["实际转速"]),3066))
    # exit()
    # cmap = plt.cm.get_cmap("Set1")
    # l1 = list(data1["MA"])
    # l2 = list(data2["MA"])
    # c1 = cmap(l1)
    # c2 = cmap(l2)
    color_dict1 = get_color_dict(data1["MA"], plt.cm.Paired.colors)
    color_dict2 = get_color_dict(data2["MA"], plt.cm.Paired.colors)
    plt.get_cmap()

    ma1 = sorted(list(set(data1["MA"])))
    ma2 = sorted(list(set(data2["MA"])))
    ma1_counts = data1["MA"].value_counts()
    ma2_counts = data2["MA"].value_counts()
    start1 = 1
    stop1 = 0
    start2 = 1
    stop2 = 0

    for ma in ma1:
        counts = ma1_counts[ma]
        stop1 += counts
        ax1.plot(np.linspace(start1, stop1, counts), list(data1["实际转速"][data1["MA"] == ma]), linewidth=2,
                 linestyle="--", color=color_dict1[str(ma)], label=str(ma))
        start1 += counts
    for ma in ma2:
        counts = ma2_counts[ma]
        stop2 += counts
        ax2.plot(np.linspace(start2, stop2, counts), list(data2["实际转速"][data2["MA"] == ma]), linewidth=2,
                 linestyle="--", color=color_dict2[str(ma)], label=str(ma))
        start2 += counts

    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax1.set_title("角速度0.1")
    ax2.set_title("角速度0.2")
    ax1.grid()
    ax2.grid()

    return fig1, fig2


def plot_aoa_ma(filepath):
    """
        对1027_8中的文件，画出攻角跟实时马赫数的关系图

    :return:
    """
    filenames = os.listdir(filepath)
    data = []
    for filename in filenames:
        with open(filepath + filename) as f:
            data_ = pd.read_csv(f)
        data.append(data_)
    fig1 = plt.figure(figsize=(10, 6), tight_layout=True)
    fig2 = plt.figure(figsize=(10, 6), tight_layout=True)
    fig3 = plt.figure(figsize=(10, 6), tight_layout=True)
    fig4 = plt.figure(figsize=(10, 6), tight_layout=True)

    for i, d in enumerate(data):
        if i < 6:
            ax = fig1.add_subplot(2, 3, i + 1)
            ax.axhline(d["MA"][0] + 0.001, color="r", linestyle="--")
            ax.axhline(d["MA"][0] - 0.001, color="r", linestyle="--")
            ax.set_title(str(d["MA"][0]) + "---" + str(d["角速度"][0]))
        elif (i >= 6) & (i < 12):
            ax = fig2.add_subplot(2, 3, i - 6 + 1)
            ax.axhline(d["MA"][0] + 0.001, color="r", linestyle="--")
            ax.axhline(d["MA"][0] - 0.001, color="r", linestyle="--")
            ax.set_title(str(d["MA"][0]) + "---" + str(d["角速度"][0]))
        elif (i >= 12) & (i < 18):
            ax = fig3.add_subplot(2, 3, i - 12 + 1)
            ax.axhline(d["MA"][0] + 0.001, color="r", linestyle="--")
            ax.axhline(d["MA"][0] - 0.001, color="r", linestyle="--")
            ax.set_title(str(d["MA"][0]) + "---" + str(d["角速度"][0]))
        else:
            ax = fig4.add_subplot(3, 3, i - 18 + 1)
            ax.axhline(d["MA"][0] + 0.001, color="r", linestyle="--")
            ax.axhline(d["MA"][0] - 0.001, color="r", linestyle="--")
            ax.set_title(str(d["MA"][0]) + "---" + str(d["角速度"][0]))
        ax.scatter(d["攻角"], d["实际MA"], s=3, c="g")

    return fig1, fig2, fig3, fig4


if __name__ == "__main__":

    # filepath = "../data_processed/1027_10/"
    filepath = "../results/14/"
    storepath = "../data_processed/1027_11/"
    savefigure = "../plot_figures/1027_10/"
    filedir = "../data_processed/1027_11/data1.csv"
    # fig = plot_speed2(filedir, "实际转速", 0.8)
    # fig.savefig("../plot_figures/1027_10/aoa_speed.png")
    # plt.show()
    # exit()

    plot_pred_data(filepath, storepath)
    exit()
    fig1, fig2, fig3, fig4 = plot_aoa_ma(filepath)
    fig1.savefig(savefigure + "fig111.png")
    fig2.savefig(savefigure + "fig222.png")
    fig3.savefig(savefigure + "fig333.png")
    fig4.savefig(savefigure + "fig444.png")
    plt.show()


    # fig1, fig2 = data_plot(filepath)
    # fig1.savefig(storepath + "0.1.png")
    # fig2.savefig(storepath + "0.2.png")
    # exit()
    # fig1, fig2 = plot_speed(filepath, storepath)

    # fig1.savefig(savefigure + "aoa_speed_0.1.jpg")
    # fig2.savefig(savefigure + "aoa_speed_0.2.jpg")
    # plt.show()
