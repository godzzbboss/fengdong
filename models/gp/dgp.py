# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import numpy as np
import GPy
from pylab import *
from sys import path
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
path.append("../")

import deepgp

# Utility to load sample data. It can be installed with pip. Otherwise just load some other data.
import pods

# --------------------  DATA PREPARATION ---------------#
# Load some mocap data.
# data = pods.datasets.cmu_mocap_35_walk_jog()
#
# Ntr = 100  # 训练集
# Nts = 500  # 测试集
#
# # All data represented in Y_all, which is the angles of the movement of the subject
# Y_all = data['Y']  # (2644,62)
# perm = np.random.permutation(Ntr + Nts)
# index_training = np.sort(perm[0:Ntr])
# index_test = np.sort(perm[Ntr:Ntr + Nts])
#
# Y_all_tr = Y_all[index_training, :]
#
# Y_all_ts = Y_all[index_test, :]
#
# # Some of the features (body joints) to be used as inputs, and some as outputs
# X_tr = Y_all_tr[:, 0:55].copy() # 输入55维
# Y_tr = Y_all_tr[:, 55:].copy() # 输出7维
#
# X_ts = Y_all_ts[:, 0:55].copy()
# Y_ts = Y_all_ts[:, 55:].copy()

# RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions.flatten() - targets.flatten()) ** 2).mean())

# my data
filepath = "../../data_processed/1027_16/5_fold/"
with open(filepath + "norm_train1.csv") as f:
    data_train = np.loadtxt(f, delimiter=",")
with open(filepath + "norm_test1.csv") as f:
    data_test = np.loadtxt(f, delimiter=",")


train_x = data_train[:,[0,1,2]]
train_y = data_train[:,3].reshape(-1,1)
test_x = data_test[:,[0,1,2]]
test_y = data_test[:,3].reshape(-1,1)
# -------------对目标值标准化---------------#
ss = StandardScaler()
norm_train_y = ss.fit_transform(train_y)
norm_test_y = ss.transform(test_y)
with open("./results/y_pred4.csv") as f:
    y_pred = np.loadtxt(f, delimiter=",").reshape(-1,1)
print(rmse(y_pred,norm_test_y))

exit()
# print(data_train.shape)
# print(data_test.shape)
# exit()
train_x = data_train[:,[0,1,2]]
train_y = data_train[:,3].reshape(-1,1)
test_x = data_test[:,[0,1,2]]
test_y = data_test[:,3].reshape(-1,1)

# print(np.var(train_y))
# exit()

# -------------对目标值标准化---------------#
ss = StandardScaler()
norm_train_y = ss.fit_transform(train_y)
norm_test_y = ss.transform(test_y)
# print(np.var(norm_train_y))
# exit()


# --------- Model Construction ----------#

# Number of latent dimensions (single hidden layer, since the top layer is observed)
Q = 1
# Define what kernels to use per layer
# 初始超参数选择

kern1 = GPy.kern.RBF(Q, ARD=True) + GPy.kern.Bias(Q) # Q表示隐结点个数
kern2 = GPy.kern.RBF(train_x.shape[1], ARD=True) + GPy.kern.Bias(train_x.shape[1])
# Number of inducing points to use
num_inducing = 100
# Whether to use back-constraint for variational posterior
back_constraint = False
# Dimensions of the MLP back-constraint if set to true
# encoder_dims = [[300], [150]]

m = deepgp.DeepGP([norm_train_y.shape[1], Q, train_x.shape[1]], norm_train_y, X_tr=train_x, kernels=[kern1, kern2], num_inducing=num_inducing,
                  back_constraint=back_constraint)

# print(np.mean(m.layers[0].Y))
# exit()
# --------- Optimization ----------#
# Make sure initial noise variance gives a reasonable signal to noise ratio.
# Fix to that value for a few iterations to avoid early local minima
for i in range(len(m.layers)):
    output_var = m.layers[i].Y.var() if i == 0 else m.layers[i].Y.mean.var()
    m.layers[i].Gaussian_noise.variance = output_var * 0.01
    m.layers[i].Gaussian_noise.variance.fix()

m.optimize(max_iters=1000, messages=True)

# # Unfix noise variance now that we have initialized the model
# for i in range(len(m.layers)):
#     m.layers[i].Gaussian_noise.variance.unfix()
#
# m.optimize(max_iters=1500, messages=True)

# # --------- Inspection ----------#
# # Compare with GP
# m_GP = GPy.models.SparseGPRegression(X=X_tr, Y=Y_tr, kernel=GPy.kern.RBF(X_tr.shape[1]) + GPy.kern.Bias(X_tr.shape[1]),
#                                      num_inducing=num_inducing)
# m_GP.Gaussian_noise.variance = m_GP.Y.var() * 0.01
# m_GP.Gaussian_noise.variance.fix()
# m_GP.optimize(max_iters=100, messages=True)
# m_GP.Gaussian_noise.variance.unfix()
# m_GP.optimize(max_iters=400, messages=True)




y_pred = m.predict(test_x)[0]
print(y_pred)
# 保存预测结果
np.savetxt("./results/y_pred4.csv", y_pred, delimiter=",")
# Y_pred_s = m.predict_withSamples(X_ts, nSamples=500)[0]
# Y_pred_GP = m_GP.predict(X_ts)[0]

print('# RMSE DGP               : ' + str(rmse(y_pred, norm_test_y)))
# print('# RMSE DGP (with samples): ' + str(rmse(Y_pred_s, Y_ts)))
# print('# RMSE GP                : ' + str(rmse(Y_pred_GP, Y_ts)))
