# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

data = pd.read_csv("../data_processed/J7.csv")
# 去重
data = data.drop_duplicates()
print(data)
#
# # file_directory = "F:/风洞/实验/2018-1-17(使用高斯过程)/gpml-matlab-v4.1-2017-10-19/\
# #                      gaohe/data_processed/data_processed.csv"
# # file_directory = re.sub("\s","",file_directory)
# # data_processed.to_csv(file_directory, index=False)
# # data_processed.to_csv("../data_processed/data_processed.csv", index=False)
#
# 将数据分区间
# l = [i/100 for i in range(58,114,2)]
# new_data = pd.DataFrame(columns=["AOA","MA","Rotate Speed"])
# for i in range(len(l)-1):
#     s = "data_" + str(i+1)
#     s = data_processed[(data_processed.MA>=l[i])&(data_processed.MA<l[i+1])]
#
#     if s.index.size>20:
#         s = s.sample(n=20, random_state=0, replace=True)
#     new_data = pd.concat([new_data,s])


# new_data.to_csv("F:/风洞/实验/2018-1-17(使用高斯过程)/gpml-matlab-v4.1-2017-10-19/gaohe/data_processed/sample_data_20.csv", index=False)
# new_data.to_csv("../data_processed/new_data.csv", index=False)


'''
    欠采样之后的样本分布

'''
# new_data = pd.read_csv("../data_processed/new_data.csv")
#
# fig = plt.figure(figsize=(6,6))
# # ax1 = fig.add_subplot(111, projection="3d")
# # ax1.scatter(new_data["AOA"].tolist(),new_data["MA"].tolist(),new_data["Rotate Speed"].tolist())
# # ax1.view_init(10,70)
# # #
# # #
# ax2 = Axes3D(fig)
# ax2.plot_trisurf(new_data["AOA"].tolist(),new_data["MA"].tolist(),new_data["Rotate Speed"].tolist())
# plt.show()


'''
    分析0.6-0.8跟1-1.1两个区间的数据

'''
# data_06_08 = pd.concat([data_2,data_3,data_4,data_5])
# data_06_08.to_csv("../data_processed/data_06_08.csv", index=False)
# data_10_11 = pd.concat([data_10,data_11])
# data_10_11.to_csv("../data_processed/data_10_11.csv", index=False)
