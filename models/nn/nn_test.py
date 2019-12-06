# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, MinMaxScaler

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class fengdong_NN(object):
    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data("norm_train1.csv",
                                                                                       "norm_test1.csv")
        self.input_node = 3  # 输入层节点个数
        self.output_node = 1  # 输出层节点个数
        self.hidden1_node = 40 # 隐藏层1节点个数
        self.hidden2_node = 30 # 隐藏层2节点个数
        self.hidden3_node = 30  # 隐藏层2节点个数
        self.batch_size = 280  # batch
        self.learning_rate_dacay = 0.995 # 学习率衰减系数
        self.global_step = tf.Variable(0, trainable=False)
        # self.ss = StandardScaler()

        self.w1 = tf.Variable(
            tf.truncated_normal([self.input_node, self.hidden1_node], dtype=tf.float32, stddev=0.1, seed=1))
        self.bias1 = tf.Variable(tf.constant(1, shape=[self.hidden1_node], dtype=tf.float32))
        self.w2 = tf.Variable(
            tf.truncated_normal([self.hidden1_node, self.hidden2_node], dtype=tf.float32, stddev=0.1, seed=2))
        self.bias2 = tf.Variable(tf.constant(1, shape=[self.hidden2_node], dtype=tf.float32))
        self.w3 = tf.Variable(
            tf.truncated_normal([self.hidden2_node, self.hidden3_node], dtype=tf.float32, stddev=0.1, seed=3))
        self.bias3 = tf.Variable(tf.constant(1, shape=[self.hidden3_node], dtype=tf.float32))
        self.w4 = tf.Variable(
            tf.truncated_normal([self.hidden3_node, self.output_node], dtype=tf.float32, stddev=0.1, seed=4))
        self.bias4 = tf.Variable(tf.constant(1, shape=[self.output_node], dtype=tf.float32))

        self.train_step = 1000  # epoch次数

    def load_data(self, train_file_name, test_file_name):
        train_data_path = "../../data_processed/1027_16/5_fold/" + train_file_name
        test_data_path = "../../data_processed/1027_16/5_fold/" + test_file_name

        with open(train_data_path) as f:
            train_data = np.loadtxt(f, delimiter=",")

        with open(test_data_path) as f:
            test_data = np.loadtxt(f, delimiter=",")

        train_data_x = train_data[:, [0, 1, 2]]
        train_data_y = train_data[:, 3].reshape(-1, 1)

        test_data_x = test_data[:, [0, 1, 2]]
        test_data_y = test_data[:, 3].reshape(-1, 1)

        # data_x = np.concatenate((train_x, test_x), axis=0)
        # data_y = np.concatenate((train_y, test_y), axis=0)

        # 对y进行归一化[-1,1]
        ss = StandardScaler()
        norm_train_data_y = ss.fit_transform(train_data_y)
        # print(ss.scale_)
        # exit()
        # norm_test_data_y = ss.transform(test_data_y)

        # mm = MinMaxScaler()
        # norm_train_data_y = mm.fit_transform(train_data_y)
        # norm_test_data_y = mm.transform(test_data_y)
        # print(norm_train_data_y)
        # exit()

        # 对y中心化
        norm_train_data_y = train_data_y - np.mean(train_data_y)
        norm_test_data_y = test_data_y - np.mean(test_data_y)

        # 目的是为了打乱数据集
        # 这里随意固定一个seed，只要seed的值一样，那么打乱矩阵的规律就是一眼的
        seed = 666
        np.random.seed(seed)
        np.random.shuffle(train_data_x)
        np.random.seed(seed)
        np.random.shuffle(norm_train_data_y)
        # print(train_data_y)
        # 将数据分为训练集、测试集
        print("训练集维度：", train_data_x.shape)
        print("测试集维度：", test_data_x.shape)
        # exit()
        return train_data_x, norm_train_data_y, test_data_x, norm_test_data_y

    # 前向传播
    def feedforward(self, input_tensor, w1, bias1, w2, bias2, w3, bias3, w4,bias4, avg_class=None):
        """
            前向传播
        :param input_tensor:
        :param w1:
        :param bias1:
        :param w2:
        :param bias2:
        :param avg_class: 是否对参数进行滑动平均
        :return:
        """
        if avg_class == None:
            hidden1_output = tf.nn.tanh(tf.matmul(input_tensor, w1) + bias1)
            # hidden1_output = tf.matmul(input_tensor, w1) + bias1
            hidden2_output = tf.nn.tanh(tf.matmul(hidden1_output, w2) + bias2)
            # hidden2_output = tf.matmul(hidden1_output, w2) + bias2
            hidden3_output = tf.nn.tanh(tf.matmul(hidden2_output, w3) + bias3)
            y = tf.matmul(hidden3_output, w4) + bias4 # 最后一层没有激活函数

            # hidden1_output = tf.nn.tanh(tf.matmul(input_tensor, w1) + bias1)
            # hidden2_output = tf.matmul(hidden1_output, w2) + bias2
            # # hidden3_output = tf.nn.tanh(tf.matmul(hidden2_output, w3) + bias3)
            # y = tf.matmul(hidden2_output, w3) + bias3  # 最后一层没有激活函数
        else:
            hidden_output = tf.nn.tanh(tf.matmul(input_tensor, avg_class.average(w1)) + avg_class.average(bias1))
            y = tf.matmul(hidden_output, avg_class.average(w2)) + avg_class.average(bias2)

        return y

    # 定义损失函数
    def get_loss(self, y, y_):
        """

        :param y: 预测值
        :param y_: 真实值
        :return:
        """
        # 均方差损失
        rmse_loss = tf.sqrt(tf.reduce_mean(tf.square(y - y_)))

        regularizer = tf.contrib.layers.l2_regularizer(0.02)  # L2正则

        regularization = regularizer(self.w1) + regularizer(self.w2)

        return rmse_loss + regularization

    # 训练
    def train(self):

        x = tf.placeholder(tf.float32, [None, self.input_node], name="input_x")
        y_ = tf.placeholder(tf.float32, [None, self.output_node], name="input_y")

        # variable_avg = tf.train.ExponentialMovingAverage(self.moving_avg_decay, self.global_step)  # 滑动平均的类
        # variable_avg_op = variable_avg.apply(tf.trainable_variables())  # 对所有的可以训练的变量执行滑动平均操作

        learning_rate = tf.train.exponential_decay(0.053, self.global_step, self.train_x.shape[0] / self.batch_size,
                                                   self.learning_rate_dacay)  # 指数衰减学习率
        y_pred = self.feedforward(x, self.w1, self.bias1, self.w2, self.bias2, self.w3, self.bias3, self.w4, self.bias4,None)  # 不对参数进行滑动平均
        # y_pred = self.ss.inverse_transform(y_pred)

        # 对参数进行滑动平均
        # y_avg = self.feedforward(x, self.w1, self.bias1, self.w2, self.bias2, variable_avg)



        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.get_loss(y_pred, y_),
                                                                     global_step=self.global_step)
        # train_op = tf.group(train_step, variable_avg_op) #

        # 测试集的RMSE
        # y_pred = self.ss.inverse_transform(y_pred)
        # y_ = self.ss.inverse_transform(y_)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_)))

        with tf.Session() as sess:
            # print(sess.run(y_pred))
            sess.run(tf.global_variables_initializer())  # 初始化所有变量
            train_rmses = []
            test_rmses = []
            for i in range(self.train_step):
                # 打乱训练数据
                np.random.seed(i)
                np.random.shuffle(self.train_x)
                np.random.seed(i)
                np.random.shuffle(self.train_y)

                # 将数据划分为小批次, 包含当前epoch的所有mini_batchs
                # 每个mini_batch是一个二维数组
                mini_batches = []
                for k in range(0, self.train_x.shape[0], self.batch_size):
                    start = k
                    end = min(k + self.batch_size, self.train_x.shape[0])
                    mini_batch_x = self.train_x[start:end, :]
                    mini_batch_y = self.train_y[start:end, :]
                    mini_batch = np.concatenate((mini_batch_x,mini_batch_y), axis=1)
                    mini_batches.append(mini_batch)

                for mb in mini_batches:
                    train_x_batch = mb[:,[0,1,2]]
                    train_y_batch = mb[:,3].reshape(-1,1)
                    sess.run(train_op, feed_dict={x: train_x_batch, y_: train_y_batch})

                # print(self.mm.inverse_transform(self.train_y))
                # exit()
                train_rmse = sess.run(rmse, feed_dict={x: self.train_x, y_: self.train_y})
                test_rmse = sess.run(rmse, feed_dict={x: self.test_x, y_: self.test_y})
                train_rmses.append(train_rmse)
                test_rmses.append(test_rmse)
                # print(sess.run(self.w1))
                print("第%d次epoch后，训练集RMSE为%g" % (i, train_rmse))
                print("第%d次epoch后，测试集RMSE为%g" % (i, test_rmse))
                print("+++++++++++++++++++++++++++++++++++++++++++++")

            plt.rcParams['figure.dpi'] = 250  # 分辨率
            mpl.rcParams['font.sans-serif'] = ['Times new roman']  # 指定默认字体
            mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

            x = np.linspace(1, self.train_step, 1000)
            l1, = plt.plot(x[50:], train_rmses[50:], c="r", alpha=0.8)
            l2, = plt.plot(x[50:], test_rmses[50:], c="g", alpha=0.8)
            plt.xlabel("Number of epoch")
            plt.ylabel("RMSE")
            plt.title("Fifth cross validation")
            plt.tick_params(labelsize=8, labelcolor="#000000")
            plt.legend([l1, l2], ["Train rmse", "Test rmse"])
            # plt.title("第5折")
            # plt.yticks([10,20,30])
            plt.show()
            # 产生每一轮迭代的训练数据
            # print("训练样本个数：%d" % self.train_x.shape[0] )
            # exit()

            # sess.run(variable_avg_op)

            # # 训练结束后，输出测试精度
            # test_accurate = sess.run(accurate, feed_dict=test_feed)
            # print("%d次迭代后，测试精度为%g" % (self.train_step, test_accurate))
            # print(sess.run(variable_avg.average(self.w1)))


if __name__ == "__main__":
    fengdong = fengdong_NN()
    # print(fengdong.mm.min_)
    fengdong.train()
    # print(fengdong.train_y)
