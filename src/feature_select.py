import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pickle.load(open('../data/numpy_data.pkl', 'rb'))
y = np.array([i[-1] for i in dataset['Y']])
dataset['Y'] = y
feature_names = ['discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time',
                 'ir', 'delta_Q', "delta_V", 'delta_time']
# 假设你有一个特征矩阵X和一个目标向量y
X = dataset['X']
X = X.reshape(X.shape[0], X.shape[1], -1)
y = dataset['Y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 创建PCA对象并指定降维后的目标维度
# pca = PCA(n_components=2)
#
# # 用训练集数据拟合PCA模型，并将训练集和测试集数据都进行降维
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# 创建随机森林分类器对象
rf = RandomForestClassifier()

# 使用降维后的训练集数据进行模型训练
rf.fit(X_train, y_train)

feature_importances = rf.feature_importances_

# 打印每个特征的权重
for feature_name, importance in zip(feature_names, feature_importances):
    print(f"Feature: {feature_name}, Importance: {importance}")
