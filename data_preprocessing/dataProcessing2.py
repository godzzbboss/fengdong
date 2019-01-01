# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict, Counter

"""
    处理原始数据

"""

columns = ["攻角", "MA", "目标转速", "实际转速", "实际MA"]

'''
    将原始的txt文件处理成csv文件

'''


def process_data1(loadFileAddr, storeFileAddr):
    with open(loadFileAddr, "r") as f:
        with open(storeFileAddr, "w") as f1:
            for line in f.readlines():
                # 正则表达式，将空白符换成逗号
                line1 = re.sub("\s+", ",", line)
                # 删除问号
                line1 = re.sub("\?", "", line1)
                line1 = line1[0:-1] + "\n"
                # 读一行写一行
                print(line1, file=f1, sep="", end="")
                # f1.write(line1)


'''
    对csv文件进行处理， 只保留["攻角", "MA", "目标转速", "实际转速", "实际MA"]，这几列

'''


def process_data2(filedir, file_names_csv):
    global columns
    data_list = []
    for f_name in file_names_csv:
        with open(filedir + f_name) as f:
            data = pd.read_csv(f)
        data.rename(columns={"俯仰角": "攻角", "目标M数": "MA", "输出转速（主转速）": "目标转速", "主转速": "实际转速", "Mcp1": "实际MA"},
                    inplace=True)
        data_ = data[columns]
        data_list.append(data_)
    return data_list


'''
    获得每个文件对应的角速度
'''


def get_rotate_speed(filespath):
    '''
        获取每个文件对应的角速度

    :param files: 文件名列表
    :return: 字典，键为文件名，值为其对应的角速度
    '''
    # 定义一个有序字典，保存文件名及其对应的角速度
    order_dict = OrderedDict()

    files = os.listdir(filespath)
    for f_name in files:
        with open(filespath + f_name) as f:
            data = pd.read_csv(f)
        # 每两个数据的时间间隔是500毫秒
        avg_rotate_speed = np.average(np.array(data["攻角"][1:]) - np.array(data["攻角"][0:-1])) * 2
        order_dict[f_name] = avg_rotate_speed

    return order_dict


'''
    画图确定是否加前馈

'''


def plot_data(filepath, savepath):
    file_names = os.listdir(filepath)
    ind = 1
    for file_name in file_names:
        # 每6个图一个画板
        if ind % 6 == 1:
            ind = 1
            fig = plt.figure(figsize=(10, 6))
            # plt.subplots_adjust(bottom=0.1)
        with open(filepath + file_name) as f:
            data = pd.read_csv(f)
        ax = fig.add_subplot(3, 3, ind)
        ax.plot(data["攻角"], data["实际MA"])
        ax.axhline(y=(data["MA"][0] + 0.001))
        ax.axhline(y=(data["MA"][0] - 0.001))
        ax.set_title(file_name)
        plt.tight_layout(pad=2, h_pad=3)
        plt.subplots_adjust(top=0.7)
        if ind % 6 == 0:
            plt.savefig(savepath + file_name.split("_")[0] + ".png")
        ind += 1
    plt.show()


'''
    对1027_0中的数据进行进一步处理，删除每组数据的前两个跟后两个数据，这四个数据
    角速度不稳定, 增加角速度跟截面积两列，0.85以上的截面积是0.85以下截面积的7/6.

'''


def process_data3(filepath, storepath):
    file_names = os.listdir(filepath)
    for file_name in file_names:
        with open(filepath + file_name) as f:
            data = pd.read_csv(f)
        # cond = np.array(data["实际MA"] > data["MA"][0] + 0.001) | np.array(data["实际MA"] < data["MA"][0] - 0.001)
        # data_over_error_band = data[cond]
        # print(data_over_error_band)
        # exit()
        data = data[2: -2]
        data.reset_index(inplace=True, drop=True)
        data["角速度"] = file_name.split("_")[1][0:3]
        if data["MA"][0] < 0.85:
            data["截面积"] = 1
        else:
            data["截面积"] = 7/6
        data.to_csv(storepath + file_name, index=False)

    return None


if __name__ == "__main__":
    '''
        对原始的csv文件进行处理，只保存需要的列，并进行保存

    '''

    filepath = "../data_processed/1027_1/"
    storepath = "../data_processed/1027_10/"
    process_data3(filepath, storepath)



    # # 获取当前工作目录
    # file_dir = "../new_data/1027_1/"
    # file_names = os.listdir(file_dir)
    # # 找出所有csv文件
    # file_names_csv = [file_name for file_name in file_names if "csv" in file_name]
    #
    # data_list = process_data2(file_dir, file_names_csv)
    #
    # # 将处理好的数据进行保存, 对攻角进行分组取平均，并计算每个文件对应的角速度
    # storeaddr = "../data_processed/1027_0/"
    # name = 1
    # for data in data_list:
    #     data = data.groupby(["攻角"], as_index = False).mean()
    #     data.to_csv(storeaddr + str(name) + ".csv", index=False)
    #     name += 1
    #
    #
    # '''
    #     获得每个文件的角速度
    #
    # '''
    # file_dir = "../data_processed/1027_0/"
    # file_names = os.listdir(file_dir)
    # d = get_rotate_speed(file_dir)
    # for f_name in file_names:
    #     if d[f_name] < 0.15 and d[f_name] > 0.05:
    #         d[f_name] = 0.1
    #     elif d[f_name] > 0.15:
    #         d[f_name] = 0.2
    #     else:
    #         pass
    #
    #     os.rename(file_dir+f_name, file_dir+f_name.split(".")[0]+"_"+str(d[f_name])+ ".csv")
    # exit()




    # # 定义一个字典，用于保存相同MA的文件
    # file_dict = defaultdict(list)
    #
    # figure = plt.figure(figsize=(10,6))
    # files_ma = set()
    #
    # for f in file_names:
    #     with open(file_dir + f) as file:
    #         data = pd.read_csv(file)
    #     # 用元组作为字典的键
    #     file_dict[(data["MA"][0], f)].append(data)
    # print(file_dict.keys())
    # exit()
    #     # files_ma.add((f,data_["MA"][0]))
    # # print(sorted(files_ma, key = lambda x : x[1]))
    # print(list(file_dict.keys())[1][1])
    # exit()


    '''

        对数据可视化，得出哪些文件是加前馈的，哪些文件是不加前馈的


    '''

    # plot_data("../data_processed/1027_2/", "../plot_figures/1027_2/")
    # process_data3("../data_processed/1027_1/", "../data_processed/1027_2/")
