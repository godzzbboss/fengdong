# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""

"""

    读入数据

"""
from contextlib import contextmanager
import numpy as np
@contextmanager
def open_many(files=None, mode="r"):
    if files is None:
        files = []
    try:
        fds = []
        for f in files:
            fds.append(open(f,mode))
        yield fds
    except ValueError as e:
        print(e)
    finally:
        for fd in fds:
            fd.close()



def load_data(train_file_name, test_file_name):
    filepath = "../data_processed/1027_16/5_fold/"
    trainpath = filepath + train_file_name
    testpath = filepath + test_file_name
    data = [] # 定义一个空列表
    with open_many([trainpath, testpath], "r") as files:
        for f in files:
            data.append(np.loadtxt(f, delimiter=","))

    return data

# 读取5折交叉验证的数据
all_data = []
for i in range(1,6):
    all_data.append(load_data("norm_train"+str(i)+".csv","norm_test"+str(i)+".csv"))


# """
#     SVR
#
# """
# from sklearn.svm import SVR
# import numpy as np
# from sklearn.metrics import mean_squared_error
#
# train_x = all_data[4][0][:,(0,1,2)]
# train_y = all_data[4][0][:,3]
#
# test_x = all_data[4][1][:,(0,1,2)]
# test_y = all_data[4][1][:,3]
#
# reg = SVR(kernel="rbf", C=1000, gamma=5, tol=1e-3,epsilon=1)
# reg.fit(train_x,train_y)
# y_pred = reg.predict(test_x)
# print("RMSE:",np.sqrt(mean_squared_error(y_pred,test_y)))



# """
#     多项式回归
# """
#
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error
#
#
# import matplotlib.pyplot as plt
#
# model = Pipeline([('poly', PolynomialFeatures(degree=6)), ('linear', LinearRegression(fit_intercept=False))])
# # fit to an order-3 polynomial data
# train_x = all_data[4][0][:,(0,1,2)]
# train_y = all_data[4][0][:,3]
#
# test_x = all_data[4][1][:,(0,1,2)]
# test_y = all_data[4][1][:,3]
#
# model = model.fit(train_x, train_y)
# y_pred = model.predict(test_x)
# print("RMSE:",np.sqrt(mean_squared_error(y_pred,test_y)))


