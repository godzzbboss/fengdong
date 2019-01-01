# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_samples_num(filepath):
    """
        获取filepath中每个文件的样本个数
    :param filepath:
    :return:
    """
    import os
    from collections import OrderedDict
    d = OrderedDict()
    filenames = os.listdir(filepath)
    for filename in filenames:
        with open(filepath + filename) as f:
            data = pd.read_csv(f)
        key = str(data["MA"][0]) + "-" + filename.split("_")[1][0:3]
        value = data.shape[0]
        d[key] = value
    # print(sorted(zip(d.keys(), d.values())))
    return d

if __name__ == "__main__":

    # filepath = "../data_processed/1027_11/data1.csv"
    filepath = "../data_processed/1027_11/"

    # -----------------------------------测试-------------------------------------
    with open(filepath + "data1_final.csv") as f:
        data = np.loadtxt(f, delimiter=",")
    data1 = data[np.logical_and(data[:,1]==0.7,data[:,3]==0.2),:] # 马赫数为0.7， 角速度为0.2的数据
    data2 = data[np.logical_and(data[:, 1] == 0.8, data[:, 3] == 0.2), :]  # 马赫数为0.8， 角速度为0.2的数据
    # print(np.mean(data[:,5]))
    # print(np.mean(data1[:,5]))
    # exit()
    # print(data1.shape)
    fig = plt.figure(figsize=(8,6))
    x = data1[1:,0]-data1[0:-1,0]
    # print(x.shape)
    # exit()
    # plt.plot(data1[1:,0]-data1[0:-1,0], data1[1:,5]-data1[0:-1,5])
    plt.plot(np.log(data2[:,0]), data2[:,5])
    plt.show()



