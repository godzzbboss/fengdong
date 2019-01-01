# -*- coding: utf-8 -*-

# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
# distances, indices = nbrs.kneighbors(np.array([-1,-1]).reshape(1,-1))
# print(distances,indices)


import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class SmoteR():
    def __init__(self):
        pass

    def generate_synthetic_data(self, data_positive, k=6, num=2):
        """
            对于每一个正类样本，产生num个新样本
        :param data_positive:
        :param k:
        :param num:
        :return:
        """
        random.seed(5) # 设置随机种子
        data_positive_x = data_positive[:,0:4]
        data_positive_y = data_positive[:,4][:,np.newaxis]

        # print(data_positive_y.shape)
        # exit()
        final_new_cases = []
        for i, data_x in enumerate(data_positive_x):
            data_y = data_positive_y[i][0]
            index = [j for j in range(data_positive_x.shape[0]) if j != i]
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(data_positive_x[index,:])
            distances, indices = nbrs.kneighbors(data_x.reshape(1,-1))
            # print(distances, indices.ravel())
            # exit()
            k_neighbors_x = data_positive_x[indices.ravel(),:]
            k_neighbors_y = data_positive_y[indices.ravel(),:]
            k_index = [ii for ii in range(k_neighbors_x.shape[0])]
            new_cases = []
            for j in range(num): # 产生num个新样本
                ind = random.sample(k_index,1)
                case_x = k_neighbors_x[ind,:].ravel() # 从data的k近邻中随机选择一个样本
                case_y = k_neighbors_y[ind,:].ravel()[0]
                diff_x = data_x - case_x
                new_x = np.array(data_x + random.random() * diff_x).reshape(1,-1) # 新样本的特征
                # print("case_x:",case_x)
                # print("new_x:", new_x)
                # print("data:", data)
                # exit()

                d1 = np.linalg.norm(new_x-data_x, 2)
                d2 = np.linalg.norm(new_x-case_x, 2)


                # old_settings = np.seterr(all='warn', over='raise')
                # from decimal import Decimal
                # d1 = Decimal(d1)
                # d2 = Decimal(d2)
                # data_y = Decimal(data_y)
                # case_y = Decimal(case_y)
                # +1防止除0
                new_y = np.array([(d2+0.5)/(d1+d2+1) * data_y + (d1+0.5)/(d1+d2+1) * case_y]).reshape(1,-1)
                # print(new_x.shape)
                # exit()
                new_case = np.concatenate((new_x, new_y), axis=1)
                new_cases.append(new_case)
            final_new_cases.append(new_cases)

        # print(final_new_cases)
        # 产生新样本
        new_data = None
        flag = False
        for new_cases in final_new_cases:
            for new_case in new_cases:
                if flag == False:
                    new_data = new_case
                    flag = True
                else:
                    new_data = np.concatenate((new_data, new_case),axis=0)

        return new_data







    # def smoter_over_sample(self):
    #     pass

    def smoter_under_sample(self, data_negative, new_data_num, sample_num):
        """
            每产生一个新样本就从从负类样本中随机取sample_num个样本，组成新的负类样本

        :param data:
        :param new_data_num:
        :param sample_num:
        :return:
        """
        import random
        index = [i for i in range(data_negative.shape[0])]
        for i in range(new_data_num):
            random.seed(i) # 每次的随机种子不一样
            index_sampled = random.sample(index, sample_num)
            data_sampled_ = data_negative[index_sampled,:]
            if i == 0:
                data_sampled = data_sampled_
            else:
                data_sampled = np.concatenate((data_sampled,data_sampled_),axis=0)
        return data_sampled

    def my_under_sample(self, data_positive, data_negative, k, sample_num):

        data = np.concatenate((data_positive, data_negative), axis=0)
        # print(data.shape)

        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(data[:,0:4])

        # 对positve中每个样本求其k近邻
        k_index = []
        for i in range(data_positive.shape[0]):
            # print(np.array(data_positive[i,:]).reshape(1,-1))
            # exit()
            distances, indices = nbrs.kneighbors(np.array(data_positive[:,0:4][i,:]).reshape(1,-1))
            k_index.extend(indices.ravel())
        # print(k_index)

        # 去重
        k_index_ = set(k_index)
        index = {i for i in range(data.shape[0])}
        diff_index = index - k_index_ # 对差集中的样本进行欠采样

        data_1 = data[list(k_index_),:] # 不用欠采样的样本
        # print(data_1.shape)
        # exit()
        data_2 = data[list(diff_index),:] # 欠采样的样本
        # print(data_2.shape)
        # exit()

        index_ = [i for i in range(data_2.shape[0])]
        sampled_index = random.sample(index_, sample_num)
        data_under_sampled = data_2[sampled_index,:]

        final_data_train = np.concatenate((data_1, data_under_sampled), axis=0)
        return  final_data_train







if __name__ == "__main__":

    filepath = "../data_processed/1027_14/"
    smoter = SmoteR()
    import io
    with open(filepath + "data_positive.csv") as f:
        data_positive = np.loadtxt(f, delimiter=",")
    with open(filepath + "data_negative.csv") as f:
        data_negative = np.loadtxt(f, delimiter=",")

    data_positive_n, data_positive_d = data_positive.shape
    data_negative_n, data_negative_d = data_negative.shape
    # print(data_positive_n)
    # print(data_negative_n)
    # exit()

    final_data_train = smoter.my_under_sample(data_positive, data_negative, 8, 1000)
    # exit()



    # new_data = smoter.generate_synthetic_data(data_positive)
    # new_data_num = new_data.shape[0]
    # data_under_sampled = smoter.smoter_under_sample(data_negative, new_data_num, 10)
    # 最终的训练集
    # final_data_train = np.concatenate((data_positive, new_data, data_under_sampled), axis=0)

    # 保存
    np.savetxt(filepath + "final_data_train_1000.csv", final_data_train, delimiter=",")




