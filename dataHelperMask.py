# -*- coding:utf8 -*-
from nltk.tokenize import word_tokenize
import numpy as np
import pickle as pkl

with open('./data/dwords.p', 'rb') as f:
    dtr = pkl.load(f)


def load_set(embed, datapath, embed_dim, alpha):
    with open(datapath, 'rb') as f:
        mylist = pkl.load(f)

    s0 = []
    s1 = []
    labels = []
    m0 = []
    m1 = []
    for each in mylist:
        s0x = each[0]
        s1x = each[1]
        label = each[2]
        score = float(label)
        labels.append(score)
        if len(each) == 5:
            sum0 = np.sum(np.array(each[3], dtype=float))
            sum1 = np.sum(np.array(each[4], dtype=float))
            if sum0 != 0:
                each[3] = np.array(each[3], dtype=float) / sum0 * alpha
            else:
                each[3] = np.array(each[3], dtype=float)
            each[3][-1] += 1.0
            m0.append(each[3])

            if sum1 != 0:
                each[4] = np.array(each[4], dtype=float) / sum1 * alpha
            else:
                each[4] = np.array(each[4], dtype=float)
            each[4][-1] += 1.0
            m1.append(each[4])

        else:
            sum0 = np.sum(np.array(each[5], dtype=float))
            sum1 = np.sum(np.array(each[6], dtype=float))
            if sum0 != 0:
                each[5] = np.array(each[5], dtype=float) / sum0 * alpha
            else:
                each[5] = np.array(each[5], dtype=float)
            each[5][-1] += 1.0
            m0.append(each[5])

            if sum1 != 0:
                each[6] = np.array(each[6], dtype=float) / sum1 * alpha
            else:
                each[6] = np.array(each[6], dtype=float)
            each[6][-1] += 1.0
            m1.append(each[6])

        for i, ss in enumerate([s0x, s1x]):
            words = word_tokenize(ss)
            index = []
            for word in words:
                if word in dtr:
                    index.append(embed[dtr[word]])
                elif word in embed.vocab:
                    index.append(embed[word])
                else:
                    index.append(np.zeros(embed_dim, dtype=float))

            if i == 0:
                s0.append(np.array(index, dtype=float))
            else:
                s1.append(np.array(index, dtype=float))

    return [s0, s1, labels, m0, m1]


# 从文件中读取数据集，并分为训练、验证和测试集合
def load_data(max_len, embed, datapath, embed_dim, alpha):
    print 'alpha: ',alpha
    train_set = load_set(embed=embed, datapath=datapath, embed_dim=embed_dim, alpha=alpha)

    train_set_x1, train_set_x2, train_set_y, train_set_m1, train_set_m2 = train_set

    # train_set length
    n_samples = len(train_set_x1)

    # 打散数据集
    sidx = np.random.permutation(n_samples)

    train_set_x1 = [train_set_x1[s] for s in sidx]
    train_set_x2 = [train_set_x2[s] for s in sidx]
    train_set_y = [train_set_y[s] for s in sidx]
    train_set_m1 = [train_set_m1[s] for s in sidx]
    train_set_m2 = [train_set_m2[s] for s in sidx]

    # 打散划分好的训练和测试集
    train_set = [train_set_x1, train_set_x2, train_set_y, train_set_m1, train_set_m2]

    # 创建用于存放训练、测试和验证集的矩阵
    new_train_set_x1 = np.zeros([len(train_set[0]), max_len, embed_dim], dtype=float)
    new_train_set_x2 = np.zeros([len(train_set[0]), max_len, embed_dim], dtype=float)
    new_train_set_y = np.zeros(len(train_set[0]), dtype=float)
    mask_train_x1 = np.zeros([len(train_set[0]), max_len], dtype=float)
    mask_train_x2 = np.zeros([len(train_set[0]), max_len], dtype=float)

    def padding_and_generate_mask(x1, x2, y, m1, m2, new_x1, new_x2, new_y, new_mask_x1, new_mask_x2):
        for i, (x1, x2, y, m1, m2) in enumerate(zip(x1, x2, y, m1, m2)):
            new_x1[i, 0:len(x1)] = x1
            new_mask_x1[i, 0:len(x1)] = m1
            new_y[i] = y
            new_x2[i, 0:len(x2)] = x2
            new_mask_x2[i, 0:len(x2)] = m2

        new_set = [new_x1, new_x2, new_y, new_mask_x1, new_mask_x2]
        del new_x1, new_x2, new_y
        return new_set

    train_set = padding_and_generate_mask(train_set[0], train_set[1], train_set[2], train_set[3], train_set[4],
                                          new_train_set_x1, new_train_set_x2, new_train_set_y, mask_train_x1,
                                          mask_train_x2)

    return train_set


def load_data_raw(max_len, embed, datapath, embed_dim, alpha):
    train_set = load_set(embed=embed, datapath=datapath, embed_dim=embed_dim, alpha = alpha)

    # 创建用于存放训练、测试和验证集的矩阵
    new_train_set_x1 = np.zeros([len(train_set[0]), max_len, embed_dim], dtype=float)
    new_train_set_x2 = np.zeros([len(train_set[0]), max_len, embed_dim], dtype=float)
    new_train_set_y = np.zeros(len(train_set[0]), dtype=float)
    mask_train_x1 = np.zeros([len(train_set[0]), max_len], dtype=float)
    mask_train_x2 = np.zeros([len(train_set[0]), max_len], dtype=float)

    def padding_and_generate_mask(x1, x2, y, m1, m2, new_x1, new_x2, new_y, new_mask_x1, new_mask_x2):
        for i, (x1, x2, y, m1, m2) in enumerate(zip(x1, x2, y, m1, m2)):
            new_x1[i, 0:len(x1)] = x1
            new_mask_x1[i, 0:len(x1)] = m1
            new_y[i] = y
            new_x2[i, 0:len(x2)] = x2
            new_mask_x2[i, 0:len(x2)] = m2

        new_set = [new_x1, new_x2, new_y, new_mask_x1, new_mask_x2]
        del new_x1, new_x2, new_y
        return new_set

    train_set = padding_and_generate_mask(train_set[0], train_set[1], train_set[2], train_set[3], train_set[4],
                                          new_train_set_x1, new_train_set_x2, new_train_set_y, mask_train_x1,
                                          mask_train_x2)

    return train_set


# return batch dataset
def batch_iter(data, batch_size):
    # get dataset and label
    x1, x2, y, mask_x1, mask_x2 = data
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)
    data_size = len(x1)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)
        return_x1 = x1[start_index:end_index]
        return_x2 = x2[start_index:end_index]
        return_y = y[start_index:end_index]
        return_mask_x1 = mask_x1[start_index:end_index]
        return_mask_x2 = mask_x2[start_index:end_index]

        yield [return_x1, return_x2, return_y, return_mask_x1, return_mask_x2]
