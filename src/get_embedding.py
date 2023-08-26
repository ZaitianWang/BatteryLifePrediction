import sys
sys.path.append('/home/wangzaitian/work/2307/battery/BatteryLifePrediction')
from utils import *
import torch
import argparse
from torch.utils.data import DataLoader
from layer.Net import *

# from torchcam.methods import SmoothGradCAMpp
torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = pickle.load(open('dataset/pkl_data/FastCharge/d&c_no_split_data.pkl', 'rb'))
early_train_dataset, early_dataset, test_dataset = dataset['train_dataset'], dataset['early_dataset'], dataset['test_dataset']

loader = DataLoader(early_train_dataset,
                    shuffle=True,
                    batch_size=64,
                    drop_last=False)

early_dataset = DataLoader(early_dataset,
                           shuffle=False,
                           batch_size=64,
                           drop_last=False)

test_dataset = DataLoader(test_dataset,
                            shuffle=False,
                            batch_size=64,
                            drop_last=False)

# model = CNN(early_train_dataset.cycle_len)
# model = Trans(early_train_dataset.cycle_len)
model = JoinModel(early_train_dataset.cycle_len)
# view model structure
print(model)
# lsd = torch.load('../save/cnn_model.pt')
lsd = torch.load('models/0817_5_1024/7.779462635517122023-08-17, 18:55, 5091.pth')
new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in lsd.items()}

model.load_state_dict(new_state_dict)
model = torch.compile(model).to(device)
train_embed, train_label = [], []
for step, batch in enumerate(loader):
    with torch.no_grad():
        x1, x2, y = batch
        train_label.append(y[:, -1])
        emb = model.encode(x1.to(device))
        # emb = model.features(x1.to(device))
        # emb = emb.view(emb.shape[0], -1)
        # emb = model.fc1(emb)
        train_embed.append(emb)
train_label = torch.cat(train_label, dim=0)
train_embed = torch.cat(train_embed, dim=0)

test_embed, test_label = [], []
for step, batch in enumerate(early_dataset):
    with torch.no_grad():
        x1, x2, y = batch
        test_label.append(y)
        emb = model.encode(x1.to(device))
        # emb = model.features(x1.to(device))
        # emb = emb.view(emb.shape[0], -1)
        # emb = model.fc1(emb)
        test_embed.append(emb)
test_label = torch.cat(test_label, dim=0)
test_embed = torch.cat(test_embed, dim=0)

RUL_embed, RUL_label = [], []
for step, batch in enumerate(test_dataset):
    with torch.no_grad():
        x1, x2, y = batch
        RUL_label.append(y)
        emb = model.encode(x1.to(device))
        # emb = model.features(x1.to(device))
        # emb = emb.view(emb.shape[0], -1)
        # emb = model.fc1(emb)
        RUL_embed.append(emb)
RUL_label = torch.cat(RUL_label, dim=0)
RUL_embed = torch.cat(RUL_embed, dim=0)
with open("dataset/pkl_data/FastCharge/train_embed.pkl", 'wb') as f:
    pickle.dump({'x': train_embed, 'y': train_label}, f)

with open('dataset/pkl_data/FastCharge/test_embed.pkl', 'wb') as f:
    pickle.dump({'x': test_embed, 'y': test_label}, f)

with open("dataset/pkl_data/FastCharge/RUL_embed.pkl", 'wb') as f:
    pickle.dump({'x': RUL_embed, 'y': RUL_label}, f)