# -*- coding: utf-8 -*-

"""
__author__ == "BigBrother"

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from skimage.transform import resize

import matplotlib.pyplot as plt

D = 1 # 输入的特征维数
EPOCH = 100000
BATCH_SIZE = 200
MEAN = 1870.2110819990296

"""
    创建自己的数据集
"""
class MyDataset(Dataset):

    # 重写Dataset的__init__, __len__, __getitem__方法

    def __init__(self, rootpath, filename, transform=None):
        """
        在init方法中读取数据集和初始化tranform操作
        :param rootpath:
        :param filename:
        :param transform:
        """
        # 读取数据集
        with open(rootpath + filename) as f:
            self.data = pd.read_csv(f)
            self.data = self.data.values
        self.transform = transform # 数据转换操作

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[[idx],:].reshape(1, -1)

        if self.transform:
            sample = self.transform(sample)

        return sample

class CenterY():
    def __call__(self, sample):
        sample[:,3] -= MEAN

        return sample


"""
    将样本转为tensor
"""
class ToTensor():
    def __call__(self, sample):
        return torch.from_numpy(sample)


class NN(nn.Module):
    def __init__(self):
        """
            定义各种网络层
        """
        super(NN, self).__init__()
        self.fc1 = nn.Linear(D, 2000)
        # self.fc2 = nn.Linear(2000, 2000)
        self.fc2 = nn.Linear(2000, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)

        return x


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-2, 2)
        m.bias.data.fill_(1)


if __name__ == "__main__":
    # mydataset = MyDataset(rootpath="E:/风洞/data_processed/1027_16/5_fold/",
    #                       filename="norm_train1.csv",
    #                       transform=transforms.Compose(
    #                           [ToTensor()]
    #                         ))
    # dataloader = DataLoader(mydataset, batch_size=len(mydataset), shuffle=True, drop_last=False)

    X = torch.unsqueeze(torch.linspace(-5, 5, 100), dim=1)
    y = torch.sin(X) + 0.2 * torch.rand(X.size())

    # 定义网络
    net = NN()
    # net.apply(weights_init_uniform)
    # 定义损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    for epoch in range(EPOCH):
        out = net(X) # 前向传播
        loss = criterion(out, y) # 计算损失
        optimizer.zero_grad() # 在计算梯度时，将上一次的梯度清零
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 梯度下降

        if epoch % 5 == 0:
            plt.cla()
            plt.scatter(X, y, c="blue")
            plt.scatter(X.detach().numpy(), out.detach().numpy(), c="red")
            plt.text(0.5, 0, "Loss=" + str(loss.item()))
            plt.pause(0.01)
#

