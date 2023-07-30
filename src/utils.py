import os.path
import random
import logging

import numpy as np
import torch
import pickle
import datetime

from sklearn.model_selection import train_test_split
from torch import nn
from src.AEDataset import AEDataset, PreDataset
from tqdm import tqdm


def get_date_time_second_string():
    r = random.randint(1, 100)
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M")
    second_string = str(now.second)
    result_string = f"{date_string}, {time_string}, {second_string}" + str(r)
    return result_string


def gene_AE_data():
    if os.path.exists('../data/AE_data.pkl'):
        return pickle.load(open('../data/AE_data.pkl', 'rb'))
    test_list = []
    train_list = pickle.load(open('../dataset/pkl_data/MIT_train.pkl', 'rb'))[0]
    test_list += pickle.load(open('../dataset/pkl_data/MIT_pr.pkl', 'rb'))[0]
    test_list += pickle.load(open('../dataset/pkl_data/MIT_sec.pkl', 'rb'))[0]
    X_train = AEDataset(gene_tensor(train_list))
    X_test = AEDataset(gene_tensor(test_list))
    dataset = {0: X_train, 1: X_test}
    pickle.dump(dataset, open('../data/AE_data.pkl', 'wb'))
    return dataset


def gene_tensor(d):
    """

    :param d:  d为四维list，battery_num * feature_num * cycle_num * feature_size(1000)
    :return:   tensor for train auto_encoder, data_num * feature_num * feature_size(1000)
    """
    x = []
    for i in d:
        tensor = torch.FloatTensor(i).permute(1, 0, 2)
        x.append(tensor)
    return torch.cat(x, dim=0)


# def permute_data(d, window_size):
#     """
#     输入原始数据和窗口，构造数据集
#     :param window_size:
#     :param d:
#     :return:
#     """
#     x = []
#     re = []
#     for i in d:
#         tensor = torch.FloatTensor(i).permute(1, 0, 2)
#         padding = (window_size - (tensor.shape[0] % window_size)) % window_size
#         padding_tensor = torch.zeros_like(tensor[:padding])
#         tensor = torch.cat([tensor, padding_tensor], dim=0)
#         x.append(tensor)
#     for item in x:
#         re.append([item[i:i + window_size] for i in range(0, len(item), window_size)])
#     return re


def init_log(log_path):
    # 创建logger对象
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理程序并添加到logger
    handler = logging.FileHandler(log_path, mode='w+')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def sample(all_feature, final_life, window_size, ):
    all_feature = all_feature.permute(1, 0, 2)
    other_feature = []
    empty_window = 80  # second_window 从15开始
    first_window = all_feature[:5]
    final_life = final_life - empty_window - window_size - 5
    step = 5
    right = 5 + empty_window
    while final_life > window_size:
        second_window = all_feature[right: (right + window_size), :, :]
        input_ = [torch.cat((first_window, second_window), dim=0), final_life]
        right += step
        final_life -= step
        other_feature.append(input_)
    return other_feature


def gen_predict_data(feature, label, win_size, ):
    data = []
    for i in range(len(feature)):
        data.extend(sample(torch.tensor(np.array(feature[i])), label[i], win_size))
    return data


def gen_early_data(feature, label, win_size):
    data = []
    for i in range(len(label)):
        print(np.array(feature[i]))
        _feature = torch.FloatTensor(feature[i]).permute(1, 0, 2)
        input_ = _feature[:win_size]
        output_ = label[i]
        # input_ = torch.cat((_feature[:5], _feature[100:100 + win_size]), dim=0)
        # output_ = label[i] - 100 - win_size
        data.append([input_, output_])
    return data


def load_data(win_size=100):
    if os.path.exists('../data/predict_data.pkl'):
        return pickle.load(open('../data/predict_data.pkl', 'rb'))
    # data_list = ['MIT_sec', 'MIT_train', 'MIT_pr', ]
    data_list = ['MIT_dataset', ]
    feature, label = [], []
    for path in tqdm(data_list):
        file = pickle.load(open('../dataset/pkl_data/' + path + '.pkl', 'rb'))
        for i in file[0]:
            feature.append(i)
        label += file[1]
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.25, random_state=4)
    early_train_data, early_test_data = gen_early_data(x_train, y_train, win_size), gen_early_data(x_test, y_test,
                                                                                                   win_size)
    # train_data, test_data = gen_predict_data(x_train, y_train, win_size), \
    #                         gen_predict_data(x_test, y_test, win_size)

    dic = {0: early_train_data, 1: early_test_data, 2: early_train_data, 3: early_test_data}
    pickle.dump(dic, open('../data/predict_data.pkl', 'wb'))
    return dic


def elastic_net_loss(y_pred, y_true, model, l1_ratio=0.75, alpha=0.0007):
    mse = nn.MSELoss()(y_pred.float(), y_true.float())
    l1_norm = 0.0
    l2_norm = 0.0

    for name, param in model.named_parameters():
        # if 'auto_encoder' not in name:
        l1_norm += torch.norm(param, p=1)
        l2_norm += torch.norm(param, p=2)
    loss = mse + alpha * (l1_ratio * l1_norm + (1 - l1_ratio) * l2_norm)
    return loss
