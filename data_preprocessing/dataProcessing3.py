# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""

import pandas as pd
import os

import numpy as np


def data_process1(filepath, storepath):
    """
    保存MA，攻角，实际转速
    :param filepath:
    :param storepath:
    :return:
    """
    file_names = os.listdir(filepath)
    columns = ["MA", "攻角", "实际转速"]
    for file_name in file_names:
        with open(filepath + file_name) as f:
            data = pd.read_csv(f)

        data_ = data[columns]
        data_.to_csv(storepath + file_name, index=False)


def data_process2(static_data, dynamic_data):
    """
        使用静态数据中的转速替换动态数据中的转速，攻角一一对应
    :return:
    """
    num1 = static_data.shape[0]
    d = {}  # 键为静态数据中的（MA，攻角），值为静态数据中的实际转速
    for i in range(num1):
        d[(static_data.ix[i]["MA"], static_data.ix[i]["攻角"])] = static_data.ix[i]["实际转速"]

    # 遍历动态数据，若其（MA，攻角）在d中，将其对应的实际转速替换为d中的实际转速
    num2 = dynamic_data.shape[0]
    dynamic_data["替换后的转速"] = dynamic_data["实际转速"]
    for i in range(num2):
        k = (dynamic_data.ix[i]["MA"], dynamic_data.ix[i]["攻角"])
        if k in d.keys():
            dynamic_data.ix[i]["替换后的转速"] = d[k]
        else:
            dynamic_data.ix[i]["替换后的转速"] = dynamic_data.ix[i]["实际转速"]
    return dynamic_data


def data_process3(filedirs):
    """

        对所有文件夹中的数据保留合适的位数：data = data.round({"MA": 5, "攻角": 3, "实际转速": 3})
        MA保留5位小数，攻角保留3位小数，实际转速保留3位小数

    :return:
    """
    for filedir in filedirs:
        file_names = os.listdir(filedir)
        for file_name in file_names:
            with open(filedir + file_name) as f:
                data = pd.read_csv(f)
            data = data.round({"MA": 5, "实际MA": 5, "攻角": 3, "实际转速": 3, "目标转速": 3, "截面积": 4})
            data.to_csv(filedir + file_name, index=False)
    return None


def data_process4(filepath, storepath):
    """
        将 4 个特征的数据保留合适的位数
    :param filepath:
    :param storepath:
    :return:
    """
    filenames = os.listdir(filepath)
    ma = []
    for filename in filenames:
        with open(filepath + filename) as f:
            data = pd.read_csv(f)
        ma.append(data["MA"][0])
        data = data.round({"MA": 5, "实际MA": 5, "攻角": 3, "实际转速": 3, "目标转速": 3, "截面积": 4})
        data.to_csv(storepath + filename, index=False)

    return None


def data_process5(filepath, storepath):
    """
        将1027_11中的数据按照角速度跟截面积进行划分
    :param filepath:
    :param storepath:
    :return:
    """
    with open(filepath + "data.csv") as f:
        data = pd.read_csv(f)
    data = np.array(data)
    # np.set_printoptions(precision=10)
    # print(data)
    # exit()
    # print(data[:,3])
    index1 = np.logical_and(data[:, 2] == 0.1, data[:, 3] == 1)  # 角速度0.1， 截面积1
    index2 = np.logical_and(data[:, 2] == 0.1, data[:, 3] == 7 / 6)  # 角速度0.1， 截面积7/6
    index3 = np.logical_and(data[:, 2] == 0.2, data[:, 3] == 1)  # 角速度0.2， 截面积1
    index4 = np.logical_and(data[:, 2] == 0.2, data[:, 3] == 7 / 6)  # 角速度0.2， 截面积7/6

    data1 = data[index1, :]
    data2 = data[index2, :]
    data3 = data[index3, :]
    data4 = data[index4, :]

    np.savetxt(storepath + "data_1.csv", data1, delimiter=",")
    np.savetxt(storepath + "data_2.csv", data2, delimiter=",")
    np.savetxt(storepath + "data_3.csv", data3, delimiter=",")
    np.savetxt(storepath + "data_4.csv", data4, delimiter=",")


def get_file(filepath, rotate_speed):
    """
        根据角速度获取文件名
    :param filepath:
    :param rotate_speed:
    :return:
    """
    file_names = os.listdir(filepath)
    file_names_ = [file_name for file_name in file_names if str(rotate_speed) in file_name]
    return file_names_


def data_merge(filepath, angspeed):
    """
     对文件夹中相应角速度的数据进行合并
    :param filenames:
    :return:
    """
    columns = ["MA", "攻角", "实际转速"]
    file_names = os.listdir(filepath)
    file_names_ = [file_name for file_name in file_names if str(angspeed) in file_name]
    data = pd.DataFrame(columns=columns)
    for file_name in file_names_:
        with open(filepath + file_name) as f:
            data_ = pd.read_csv(f)
        data_ = data_[columns]
        data = pd.concat([data, data_], axis=0)

    return data


def data_merge2(filepath, storepath, columns):
    """
        对1027_8中的数据进行合并，保留需要的特征
    :param filepath:
    :param storepath:
    :return: data
    """
    filenames = os.listdir(filepath)
    data = pd.DataFrame(columns=columns)
    for filename in filenames:
        with open(filepath + filename) as f:
            data_ = pd.read_csv(f)[columns]
        data = pd.concat([data, data_], axis=0)

    data.to_csv(storepath + "data.csv", index=False)


def down_sample(filepath, storepath):
    """
        对训练集进行欠采样，每隔三个样本采一个。
    :param filepath:
    :param storepath:
    :return:
    """
    filename = "data_train.csv"
    with open(filepath + filename) as f:
        data = pd.read_csv(f)
    index = [i for i in range(0, data.shape[0], 3)]
    down_data = data.ix[index]

    down_data.to_csv(storepath + "down_data.csv", index=False)


def generate_train_test_data(filepath, storepath, train_rate, method=1):
    """
        对filepath中数据进行采样，生成训练集跟测试集，
        如果method=1, 10%的测试集(两侧)，90%的训练集（中间）
        如果method=2, 10%的测试集(随机)，90%的训练集（随机）

    :param filepath:
    :param storepath:
    :param method:
    :return:
    """
    columns = ["攻角", "实际MA", "角速度", "截面积", "实际转速"]
    filenames = os.listdir(filepath)
    data_train = None
    data_test = None

    for filename in filenames:
        with open(filepath + filename) as f:
            data = pd.read_csv(f)
        data = data[columns]
        l = data.shape[0]
        data_np = np.array(data)
        if method == 1:
            top_test_data = data_np[0:int(np.ceil(l * 0.05)), :]
            mid_train_data = data_np[int(np.ceil(l * 0.05)):int(np.ceil(l * 0.05)) + int(np.ceil(l * 0.85)), :]
            bottom_test_data = data_np[int(np.ceil(l * 0.05)) + int(np.ceil(l * 0.85)):, :]
            test_data_ = np.concatenate((top_test_data, bottom_test_data), axis=0)
            if data_train is None:
                data_train = mid_train_data
            else:
                data_train = np.concatenate((data_train, mid_train_data), axis=0)
            if data_test is None:
                data_test = test_data_
            else:
                data_test = np.concatenate((data_test, test_data_), axis=0)
        else:
            import random
            random.seed(1)  # 设置随机种子
            index = [i for i in range(l)]
            index1 = random.sample(index, int(l * train_rate))
            index2 = [i for i in index if i not in index1]
            train_data_ = data_np[index1, :]
            test_data_ = data_np[index2, :]
            if data_train is None:
                data_train = train_data_
            else:
                data_train = np.concatenate((data_train, train_data_), axis=0)
            if data_test is None:
                data_test = test_data_
            else:
                data_test = np.concatenate((data_test, test_data_), axis=0)

    data_train = pd.DataFrame(data_train, columns=columns)
    data_test = pd.DataFrame(data_test, columns=columns)
    data_train.to_csv(storepath + "data_train1.csv", index=False)
    data_test.to_csv(storepath + "data_test1.csv", index=False)


def get_data_num(filepath):
    """
        获取filepath路径下，每个文件数据的个数
    :param filepath:
    :return:
    """
    from collections import OrderedDict
    filenames = os.listdir(filepath)
    data_count = OrderedDict()
    for filename in filenames:
        with open(filepath + filename) as f:
            data = pd.read_csv(f)
        data_count[str(data["MA"][0]) + "-" + str(data["角速度"][0])] = data.shape[0]

    return data_count


def under_sample(data, sample_num=1400):
    """
        对训练集data中的数据进行分区间进行随机采样
    :param data:
    :param start:
    :param end:
    :return:
    """
    import random
    index1 = data["实际转速"] < 2000
    index2 = np.logical_and(data["实际转速"] >= 2000, data["实际转速"] < 2600)
    index3 = np.logical_and(data["实际转速"] >= 2600, data["实际转速"] < 3200)

    data_1 = data.ix[index1]
    data_2 = data.ix[index2]
    data_3 = data.ix[index3]
    data_1.reset_index(inplace=True, drop=True)
    data_2.reset_index(inplace=True, drop=True)
    data_3.reset_index(inplace=True, drop=True)

    # sample_num = min(data_1.shape[0],data_2.shape[0],data_3.shape[0],data_4.shape[0])

    sample_index1 = sorted(random.sample(list(data_1.index), sample_num))
    sample_index2 = sorted(random.sample(list(data_2.index), sample_num))
    sample_index3 = sorted(random.sample(list(data_3.index), sample_num - 800))

    sample_data_1 = data_1.ix[sample_index1]
    sample_data_2 = data_2.ix[sample_index2]
    sample_data_3 = data_3.ix[sample_index3]

    data_ = pd.concat([sample_data_1, sample_data_2, sample_data_3])
    # print(data_.shape[0])
    # exit()

    return data_


def under_sample1(data, start_speed, end_speed, sample_num):
    """
        根据实际转速，对训练数据集data进行采样
    :param data:
    :param start_speed:
    :param end_speed:
    :param sample_num:
    :return:
    """
    import random
    index = np.logical_and(data["实际转速"] >= start_speed, data["实际转速"] < end_speed)
    data_ = data.ix[index]
    data_.reset_index(inplace=True, drop=True)
    if data_.shape[0] > sample_num:
        sample_index = sorted(random.sample(list(data_.index), sample_num))
    else:
        sample_index = list(data_.index)
    data_ = data_.ix[sample_index]

    return data_


def get_density(data, start_speed, end_speed):
    """
        获取转速区间的密度
    :param data: ndarray
    :param start_speed:
    :param end_speed:
    :return:
    """
    index = np.logical_and(data[:, 4] >= start_speed, data[:, 4] < end_speed)
    density = len(index[index == True]) / (end_speed - start_speed)

    return density


def under_sample2(data, step, density):
    """
        对data根据转速区间的密度进行欠采样，使其密度最大为density
    :param data:
    :param step: 每个区间的间隔
    :param density:
    :return:
    """
    min_speed = 1400
    max_speed = 2900
    import random
    final_data = None  # 保存最终的数据
    for i in range(min_speed, max_speed, step):
        index = np.logical_and(data[:, 4] >= i, data[:, 4] < i + step)
        data_ = data[index, :]
        while get_density(data_, i, i + step) > density:  # 欠采样
            inds = [j for j in range(data_.shape[0])]
            sampled_inds = random.sample(inds, len(inds) - 1)
            data_ = data_[sampled_inds, :]

        if final_data is None:
            final_data = data_
        else:
            final_data = np.concatenate((final_data, data_), axis=0)

    return final_data


def get_sample_num(data, start_speed, end_speed):
    """
        获得转速区间内的样本个数
    :param data:
    :param start_speed:
    :param end_speed:
    :return:
    """
    index = np.logical_and(data["实际转速"] >= start_speed, data["实际转速"] < end_speed)
    data_ = data.ix[index]

    return data_.shape[0]


if __name__ == "__main__":

    filepath = "../data_processed/1027_12/standardscale/"
    storepath = "../data_processed/1027_13/"

    with open(filepath + "data_train_55.csv") as f:
        data = np.loadtxt(f, delimiter=",")

    final_data = under_sample2(data, 100, 1.7)
    # 获取每个区间的密度
    densities = {}
    for i in range(1400, 2900, 100):
        density = get_density(final_data, i, i + 100)
        densities[str(i) + "-" + str(i + 100)] = density
    # print(densities)
    np.savetxt(storepath + "final_data_train_1.7.csv", final_data, delimiter=",")



    # generate_train_test_data(filepath, storepath, 0.5, 2)
    # exit()

    # with open(storepath + "data1.csv") as f:
    #     data = pd.read_csv(f)
    # index = np.logical_and(data["实际转速"] >= 2550, data["实际转速"] < 2600)
    # print(data.ix[index]["MA"].unique())
    # exit()

    # with open(storepath + "under_data_train.csv") as f:
    #     data = pd.read_csv(f)
    #
    #
    # data = under_sample(data)
    # data.to_csv(storepath + "under_data_train1.csv", index=False)

    # under_sample_data = pd.DataFrame(columns=data.columns)
    # # 以50转速为区间，随机采样一半
    # for i in range(1400, 2850, 50):
    #     index = np.logical_and(data["实际转速"] >= i, data["实际转速"] < i+50)
    #     sample_num = int(data.ix[index].shape[0] * 2 / 3) # 采2/3
    #     data_ = under_sample1(data, i, i+50, sample_num)
    #     under_sample_data = pd.concat([under_sample_data, data_], axis=0)
    #
    # under_sample_data.to_csv(storepath + "under_data_train2.csv", index=False)
    #
    # print(under_sample_data)
    # exit()
    #
    #
    # from collections import OrderedDict
    #
    # d = OrderedDict() # 保存每个区间的样本个数
    # for i in range(1400, 2850, 50):
    #     sample_num = get_sample_num(under_sample_data, i, i+50)
    #     d[str(i)+"-"+str(i+50)] = sample_num
    # print(d)
    # exit()

    # 区间样本密度
    # mean_density = data.shape[0] / (max(data["实际转速"])-min(data["实际转速"]))
    # print(mean_density)
    # exit()
    # d = OrderedDict()
    # for i in range(1400, 2700, 300):
    #     sample_num = get_sample_num(data, i, i+300)
    #     d[str(i)+"-"+str(i+300)] = sample_num
    # print(d)
    # print(np.array(list(d.values()))/3)
    # exit()

    # 基于滑动窗口的密度采样法

    # data_list = []
    # 对第一个区间采样
    # data_ = under_sample2(data, 1400, 1400 + 150, mean_density)
    # index1 = np.logical_and(data_["实际转速"] >= 1400, data_["实际转速"] < 1400 + 50)
    # index2 = np.logical_and(data_["实际转速"] >= 1400 + 50, data_["实际转速"] < 1400 + 150)
    # data1 = data_.ix[index1]
    # data2 = data_.ix[index2]
    # data_list.append(data1)
    # for i in range(1550, 2850, 50):
    #     index_temp = np.logical_and(data["实际转速"] >= i, data["实际转速"] < i + 50)
    #     data_temp = pd.DataFrame(columns=data.columns)
    #     data_temp = pd.concat([data2, data.ix[index_temp]], axis=0) # 获取新的滑动区间
    #     data_temp.reset_index(inplace=True, drop=True)
    #     data_ = under_sample2(data_temp, i-100, i+50, mean_density) # 对当前滑动区间数据进行欠采样
    #     index1 = np.logical_and(data_["实际转速"] >= i - 100, data_["实际转速"] < i - 50)
    #     index2 = np.logical_and(data_["实际转速"] >= i - 50, data_["实际转速"] < i + 50)
    #     data1 = data_.ix[index1]
    #     data2 = data_.ix[index2]
    #     data_list.append(data1)
    #
    # under_sample_data = pd.concat(data_list, axis=0)
    # under_sample_data.to_csv(storepath + "under_data_train.csv", index=False)
    # print(under_sample_data)
    # exit()



    # for i in range(1400, 2900, 50):
    #     data_ = under_sample2(data, i, i + 50, mean_density)
    #     under_sample_data = pd.concat([under_sample_data, data_], axis=0)
    # under_sample_data.to_csv(storepath + "under_data_train.csv", index=False)
    # print(under_sample_data)
    #
    # exit()



    # for i in range(1400, 2900, 100):
    #     if i in [1600, 2100, 2300, 2500, 2600]:
    #         data_ = under_sample1(data,i,i+100,200)
    #     else: # 如果不在这些区间，则不进行欠采样
    #         index = np.logical_and(data["实际转速"] >= i, data["实际转速"] < i+100)
    #         sample_num = data.ix[index].shape[0]
    #         data_ = under_sample1(data,i,i+100,sample_num)
    #     under_sample_data = pd.concat([under_sample_data, data_], axis=0)
    # under_sample_data.to_csv(storepath + "under_data_train1.csv", index=False)
    # exit()


    # columns = ["攻角", "MA", "实际MA", "角速度", "截面积", "实际转速"]
    # data_merge2(filepath, storepath, columns)


    #
    # down_sample(storepath, storepath)
    # exit()


    # data_1 = data_merge(filepath, 0.1)
    # data_2 = data_merge(filepath, 0.2)
    # data_1.to_csv(storepath+"data_1.csv", index=False)
    # data_2.to_csv(storepath + "data_2.csv", index=False)
    # exit()
