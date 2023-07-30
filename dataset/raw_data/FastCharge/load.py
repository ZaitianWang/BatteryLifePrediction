import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.manifold import TSNE


def read():
    with open('battery_dataset.pkl', "rb") as f:
        dd = pickle.load(f)
    data = []
    y = []
    for i in dd:
        data += list(dd[i][0])
        y += list(dd[i][1])
    return convert_to_array(data), convert_to_array(y)


def convert_to_array(my_list):
    """
    将三维 list 转换为 NumPy 数组
    """
    if isinstance(my_list, list):  # 判断当前元素是否为 list
        return np.array([convert_to_array(x) for x in my_list])  # 递归调用 convert_to_array() 函数
    else:  # 如果当前元素不是 list，则直接将其转换为 NumPy 数组
        return np.array(my_list)


def get_embedding():
    dataset1 = pickle.load(open('data/train_dataset.pkl', 'rb'))
    dataset2 = pickle.load(open('data/test1_dataset.pkl', 'rb'))
    dataset3 = pickle.load(open('data/test2_dataset.pkl', 'rb'))
    data = dataset1.x + dataset2.x + dataset3.x
    y = dataset1.y + dataset2.y + dataset3.y
    model = torch.load('save/loss2_weight_0.25/pre_model.pt').cpu()
    with torch.no_grad():
        embed, _ = model(torch.FloatTensor(data))
    return convert_to_array(_), np.array(y),


def get_data():
    dataset1 = pickle.load(open('data/train_dataset.pkl', 'rb'))
    dataset2 = pickle.load(open('data/test1_dataset.pkl', 'rb'))
    dataset3 = pickle.load(open('data/test2_dataset.pkl', 'rb'))
    data = dataset1.x + dataset2.x + dataset3.x
    y = dataset1.y + dataset2.y + dataset3.y

    return convert_to_array(data), np.array(y),


def plot_embedding(data, label, ):
    # 基础颜色为红色
    base_color = (1.0, 0.0, 0.0)

    # 获取 label 的最大值和最小值
    min_label = np.min(label)
    max_label = np.max(label)

    # 遍历每个点，并根据 label 的大小为它们分配颜色
    for i in range(len(label)):
        # 计算 alpha 值，alpha 值越大，颜色越深
        alpha = (label[i] - min_label) / (max_label - min_label)
        # 使用基础颜色和 alpha 值来创建当前点的颜色
        color = tuple(alpha * np.array(base_color) + (1 - alpha) * np.array((1.0, 1.0, 1.0)))
        # 绘制当前点
        plt.scatter(data[i, 0], data[i, 1], color=color)

    # 显示散点图
    plt.show()


def main():
    data, label = read()
    data = data.reshape(data.shape[0], -1)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    plt.scatter(result[:, 0], result[:, 1], c=label)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
