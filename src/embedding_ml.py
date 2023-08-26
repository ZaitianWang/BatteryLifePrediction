import sys

from sklearn.model_selection import cross_val_score
sys.path.append('/home/wangzaitian/work/2307/battery/BatteryLifePrediction')
from utils import *
import torch
import argparse
from torch.utils.data import DataLoader
from layer.Net import *
import xgboost as xgb
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

torch.set_num_threads(1)

with open('dataset/pkl_data/FastCharge/train_embed.pkl', 'rb') as f:
    train_embed = pickle.load(f)
with open('dataset/pkl_data/FastCharge/test_embed.pkl', 'rb') as f:
    test_embed = pickle.load(f)
with open('dataset/pkl_data/FastCharge/RUL_embed.pkl', 'rb') as f:
    RUL_embed = pickle.load(f)

# xgboost: 7.80, 16.41
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=10, learning_rate=0.1, eval_metric='rmse', early_stopping_rounds=10)
model.fit(train_embed['x'].detach().cpu().numpy(), train_embed['y'].detach().cpu().numpy(), 
            # eval_set=[(RUL_embed['x'].detach().cpu().numpy(), RUL_embed['y'][:,-1].detach().cpu().numpy())])
            eval_set=[(test_embed['x'].detach().cpu().numpy(), test_embed['y'].detach().cpu().numpy())])
# t_pred = model.predict(test_embed['x'].detach().cpu().numpy())

# svm: 8.05, 16.29
# model = svm.SVC(kernel='linear', C=1, tol=1e-3, verbose=True, gamma='scale')
# model.fit(train_embed['x'].detach().cpu().numpy(), train_embed['y'].detach().cpu().numpy())
# t_pred = model.predict(test_embed['x'].detach().cpu().numpy())

# knn: 7.95, 16.30
# k_error = []
# for i in range(20):
#     model = KNeighborsRegressor(n_neighbors=i+1)
#     scores = cross_val_score(model, train_embed['x'].detach().cpu().numpy(), train_embed['y'].detach().cpu().numpy(), cv=5, scoring='neg_mean_squared_error')
#     print('k: ', i+1, 'score: ', np.mean(scores))
#     k_error.append(np.mean(scores))
# k = np.argmax(k_error) + 1
# print('best k: ', k)
# model = KNeighborsRegressor(n_neighbors=k)
# model.fit(train_embed['x'].detach().cpu().numpy(), train_embed['y'].detach().cpu().numpy())

# rul test
print('RUL test')
print('MSE: ', mean_squared_error(RUL_embed['y'][:,-1].detach().cpu().numpy(), model.predict(RUL_embed['x'].detach().cpu().numpy())))
print('Percent error: ', np.mean(np.abs(RUL_embed['y'][:,-1].detach().cpu().numpy() - model.predict(RUL_embed['x'].detach().cpu().numpy())) / RUL_embed['y'][:,-1].detach().cpu().numpy()) * 100)

# early test
print('Early test')
print('MSE: ', mean_squared_error(test_embed['y'].detach().cpu().numpy(), model.predict(test_embed['x'].detach().cpu().numpy())))
print('Percent error: ', np.mean(np.abs(test_embed['y'].detach().cpu().numpy() - model.predict(test_embed['x'].detach().cpu().numpy())) / test_embed['y'].detach().cpu().numpy()) * 100)
