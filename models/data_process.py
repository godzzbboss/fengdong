# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
"""
    本文件为最终的可视化与数据处理代码
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def load_data(filepath):
    """
        返回攻角、实时马赫数、截面积、转速
    :param filepath:
    :return:
    """
    with open(filepath) as f:
        data = np.loadtxt(f, delimiter=",")
    data = data[:, [0, 2, 4, 5]]
    return data


def plot_data(data):
    """
        画出完整数据的特征分布
    :param data:
    :return:
    """
    plt.hist(data[:, 3], bins="auto", color="#0504aa", rwidth=0.9, alpha=0.75)
    # plt.grid(axis="y",alpha=0.7)
    plt.xlabel("转速")
    plt.xticks(np.linspace(1400, 2400, 11))
    plt.savefig("./figures/speed.png")
    plt.show()
    # plt.hist(data[:, 1], 20)
    # plt.hist(data[:, 2], 20)
    # plt.show()


def plot_data1(data):
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], color="#0504aa", alpha=0.75, s=5)
    ax.set_xlabel("攻角")
    ax.set_ylabel("实时马赫数")
    ax.set_zlabel("截面积")
    # ax.set_zticks([])
    # plt.savefig("./figures/feature.png")
    plt.show()


def split_train_test1(data):
    """
        随机划分训练集跟测试集
    :param data:
    :return:
    """
    np.random.seed(3)
    np.random.shuffle(data)
    kf = KFold(n_splits=5, shuffle=False, random_state=10)
    i = 1
    for tr_inds, ts_inds in kf.split(data):
        trainstorepath = "../data_processed/1027_16/5_fold/norm_train" + str(i) + "_.csv"
        teststorepath = "../data_processed/1027_16/5_fold/norm_test" + str(i) + "_.csv"
        train_data = data[tr_inds, :][:, [0, 1, 2, 3]]
        test_data = data[ts_inds, :][:, [0, 1, 2, 3]]
        train_x = train_data[:, [0, 1, 2]]
        train_y = train_data[:, 3].reshape(-1, 1)
        test_x = test_data[:, [0, 1, 2]]
        test_y = test_data[:, 3].reshape(-1, 1)
        print(train_x)
        # 标准化
        ss = StandardScaler()
        norm_train_x = ss.fit_transform(train_x)
        norm_test_x = ss.transform(test_x)
        # print(ss.mean_, ss.scale_)
        # print(norm_train_x.shape)
        # print(train_y.shape)
        # exit()
        norm_train_data = np.concatenate((norm_train_x, train_y), axis=1)
        norm_test_data = np.concatenate((norm_test_x, test_y), axis=1)

        np.savetxt(trainstorepath, norm_train_data, delimiter=",")
        np.savetxt(teststorepath, norm_test_data, delimiter=",")
        i += 1


def split_train_test2(data):
    """
        分组划分数据集
    :param data:
    :return:
    """
    # 将数据打乱
    np.random.seed(3)
    np.random.shuffle(data)

    index1 = data[:, 1] == 0.7
    n_index1 = [not i for i in index1]
    index2 = data[:, 1] == 0.8
    n_index2 = [not i for i in index2]
    index3 = data[:, 1] == 0.9
    n_index3 = [not i for i in index3]
    index4 = data[:, 1] == 1
    n_index4 = [not i for i in index4]
    index5 = data[:, 1] == 1.1
    n_index5 = [not i for i in index5]

    train_data_1 = data[n_index1, :][:, [0, 2, 4, 5]]
    test_data_1 = data[index1, :][:, [0, 2, 4, 5]]

    train_data_2 = data[n_index2, :][:, [0, 2, 4, 5]]
    test_data_2 = data[index2, :][:, [0, 2, 4, 5]]

    train_data_3 = data[n_index3, :][:, [0, 2, 4, 5]]
    test_data_3 = data[index3, :][:, [0, 2, 4, 5]]

    train_data_4 = data[n_index4, :][:, [0, 2, 4, 5]]
    test_data_4 = data[index4, :][:, [0, 2, 4, 5]]

    train_data_5 = data[n_index5, :][:, [0, 2, 4, 5]]
    test_data_5 = data[index5, :][:, [0, 2, 4, 5]]

    # 标准化并且保存
    ss = StandardScaler()
    for j, i in enumerate(np.arange(0.7, 1.2, 0.1)):
        i = round(i,2)
        j = j + 1
        train_storepath = "../data_processed/1027_16/group_data/train_"+str(i)+".csv"
        test_storepath = "../data_processed/1027_16/group_data/test_" + str(i) + ".csv"

        train_data = eval("train_data_" + str(j))
        test_data = eval("test_data_" + str(j))

        train_x = train_data[:, [0, 1, 2]]
        train_y = train_data[:, 3].reshape(-1,1)
        test_x = test_data[:, [0, 1, 2]]
        test_y = test_data[:, 3].reshape(-1,1)

        train_data = np.concatenate([train_x,train_y], axis=1)
        test_data = np.concatenate([test_x,test_y], axis=1)

        norm_train_x = ss.fit_transform(train_x)
        norm_test_x = ss.transform(test_x)

        norm_train_data = np.concatenate([norm_train_x, train_y], axis=1)
        norm_test_data = np.concatenate([norm_test_x, test_y], axis=1)

        np.savetxt(train_storepath,train_data, delimiter=",")
        np.savetxt(test_storepath, test_data, delimiter=",")

def get_outlier(origin_data_path, cleaned_data_path):
    """
        找的异常数据并返回
    :param origin_data_path: 原始数据文件路径
    :param cleaned_data_path: 清洗后数据的路径
    :return:
    """
    from models.myutils import open_many
    with open_many([origin_data_path, cleaned_data_path]) as fs:
        origin_data = pd.read_csv(fs[0])
        cleaned_data = pd.read_csv(fs[1])
    return origin_data, cleaned_data




if __name__ == "__main__":
    filepath = "../data_processed/1027_16/data11.csv"

    # 找到异常数据
    origin_filepath = "../data_processed/1027_16/origin_data/"
    cleaned_filepath = "../data_processed/1027_16/cleaned_data/"
    origin_filenames = os.listdir(origin_filepath)
    cleaned_filenames = os.listdir(cleaned_filepath)

    for i, j in zip(origin_filenames, cleaned_filenames):
        # print(origin_filepath+i)
        origin_data, cleaned_data = get_outlier(origin_filepath+i, cleaned_filepath+j)
        origin_data["攻角"] = origin_data["攻角"].apply(lambda x: round(x, 4))
        cleaned_data["攻角"] = cleaned_data["攻角"].apply(lambda x: round(x,4))
        # print(origin_data)
        # print(cleaned_data)
        # exit()


        inds = origin_data["攻角"].isin(cleaned_data["攻角"]).apply(lambda x: not x) # 找到异常数据的索引
        outlier_data = origin_data[inds]
        plt.scatter(outlier_data["攻角"],outlier_data["实际MA"])
        plt.title(i)
        plt.show()
        # 将异常数据保存
        storepath = "../data_processed/1027_16/outlier_data/"
        outlier_data.to_csv(storepath + i, index=False)

