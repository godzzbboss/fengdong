# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import random

# datapath = "../data_processed/1027_11/data_final.csv"
# # storepath = "../data_processed/1027_12/standardnoise/"
# storepath = "../data_processed/1027_13_1/"
# with open(datapath) as f:
#     data = np.loadtxt(f, delimiter=",")
# print(data.shape)
# columns = data.columns
# train_x, test_x, train_y, test_y = train_test_split(data[columns[0:4]], data[columns[4]], test_size=0.5, random_state=1)

# 标准化全部数据
# data_x = data[data.columns[0:4]]
# data_y = data[data.columns[4]]
# ss = StandardScaler()
# norm_train_x = ss.fit_transform(train_x)
# norm_test_x = ss.transform(test_x)
# print(ss.mean_, ss.scale_)
# exit()




# print(min(data[:,0]), max(data[:,0]))
# print(min(data[:,1]), max(data[:,1]))
# print(min(data[:,2]), max(data[:,2]))
# print(min(data[:,3]), max(data[:,3]))
# print(min(data[:,4]), max(data[:,4]))
# exit()

# 训练集， 测试集
# columns = data.columns
# train_x, test_x, train_y, test_y = train_test_split(data[:,0:4], data[:,4], test_size=0.5, random_state=1)
#
# train_y = np.array(train_y)[:,np.newaxis]
# test_y = np.array(test_y)[:,np.newaxis]

# 未归一化的训练集跟测试集
# data_train_55 = np.concatenate((train_x, train_y), axis=1)
# data_test_55 = np.concatenate((test_x, test_y), axis=1)

# np.savetxt(storepath + "data_train_55_origin.csv", data_train_55, delimiter=",")
# np.savetxt(storepath + "data_test_55_origin.csv", data_test_55, delimiter=",")
# exit()


# 对训练集的角速度跟截面积加噪声
# data_num = train_x.shape[0]
#
# aoa_speed_noise = []
# for i in range(data_num):
#     aoa_speed_noise.append(random.gauss(0,0.01))
#
# area_noise  = []
# for i in range(data_num):
#     area_noise.append(random.gauss(0,0.01))
# train_x[:,2] = train_x[:,2] + aoa_speed_noise
# train_x[:,3] = train_x[:,3] + area_noise
# 标准化
# ma = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.05, 0.75, 0.85, 0.95]
trainpath = "../data_processed/1027_15/aoa_train.csv"
for i in range(10):
    testpath = "../data_processed/1027_15/aoa_test" + str(i) + ".csv"
    data_train = np.loadtxt(trainpath, delimiter=",")
    data_test = np.loadtxt(testpath, delimiter=",")
    train_x = data_train[:, [0, 1, 2]]
    train_y = data_train[:, 3]
    test_x = data_test[:, [0, 1, 2]]
    test_y = data_test[:, 3]
    ss = StandardScaler()
    norm_train_x = ss.fit_transform(train_x)
    np.set_printoptions(precision=9)
    print(i)
    print(ss.mean_, ss.scale_)
    norm_test_x = ss.transform(test_x)

    # mm = MinMaxScaler()
    # norm_train_x = mm.fit_transform(train_x)
    # norm_test_x = mm.transform(test_x)
    #
    # print(norm_test_x)
    # exit()
    #
    data_train = np.concatenate((norm_train_x, train_y.reshape(-1, 1)), axis=1)
    data_test = np.concatenate((norm_test_x, test_y.reshape(-1, 1)), axis=1)

    np.savetxt("../data_processed/1027_15/norm_aoa_train.csv", data_train, delimiter=",")
    np.savetxt("../data_processed/1027_15/norm_aoa_test"+str(i)+".csv", data_test, delimiter=",")
