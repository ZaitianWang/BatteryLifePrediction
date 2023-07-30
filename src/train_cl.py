import argparse
import random

from torch.utils.data import DataLoader
from layer.Net import *
from utils import *
import time

torch.set_num_threads(2)


# export PYTHONPATH=/home/guoshipeng/BLP

def mape(pred, actual):
    ape = torch.abs(actual - pred) / actual
    mape = torch.mean(ape)
    return mape


def generate_valid_subset(length, min_subset_length):
    begin = random.randint(0, length - 1)
    end = random.randint(begin, 200 - 1)

    if end - begin < min_subset_length:
        return generate_valid_subset(length, min_subset_length)  # 递归调用函数直到满足条件

    return begin, end


def generate_train_data(batch, label):
    sample_num = 10
    re, target = [], []
    for j in range(batch.shape[0]):
        tensor = batch[j]
        zero = torch.zeros_like(tensor)
        zero[:, -100:, :] = tensor[:, 0:100, :]
        re.append(zero)
        target.append(label[j][100])
        for _ in range(sample_num - 1):
            zero = torch.zeros_like(tensor)
            begin, end = generate_valid_subset(150, 50)
            zero[:, -end + begin:, :] = tensor[:, begin:end, :]
            target.append(label[j][end])
            re.append(zero)

    return torch.stack(re), torch.stack(target)


def train(nni_params, dir_path):
    device = 'cuda:' + str(nni_params['cuda_index']) if torch.cuda.is_available() else 'cpu'
    dataset = pickle.load(open(nni_params['dataset_path'], 'rb'))
    train_dataset, early_dataset, test_dataset = dataset['train_dataset'], dataset['early_dataset'], dataset[
        'test_dataset']
    loader = DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=nni_params['batch_size'],
                        drop_last=True)

    early_dataset = DataLoader(early_dataset,
                               shuffle=True,
                               batch_size=nni_params['batch_size'],
                               drop_last=False)

    test_dataset = DataLoader(test_dataset,
                              shuffle=True,
                              batch_size=nni_params['batch_size'],
                              drop_last=False)
    # model = torch.compile(JoinModel(None, train_dataset.cycle_len).to(device))
    model = JoinModel(train_dataset.cycle_len).to(device)
    # state_dict = torch.load("../save/model.pt")
    # new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    # model.load_state_dict(new_state_dict)
    # model = torch.compile(model)

    print("loading data form ", nni_params['dataset_path'])
    print("starting training.... 模型参数量(单位： W) ：")
    print(sum(p.numel() for p in model.parameters()) / 10000)
    optimizer = torch.optim.Adam(model.parameters(), lr=nni_params['lr'])
    min_loss = 10
    for e in range(nni_params['epoch']):
        total_loss = []
        begin_time = time.time()
        for step, batch in enumerate(loader):
            x, _, y = batch
            x, y = x.to(device), y[:, -1].to(device)
            loss = model.contrastive_loss(x, y)
            total_loss.append(loss.item()/nni_params['batch_size'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for step, batch in enumerate(test_dataset):
            x, _, y = batch
            x, y = x.to(device), y[:, -1].to(device)
            loss = model.contrastive_loss(x, y)
            total_loss.append(loss.item()/nni_params['batch_size'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = np.mean(total_loss)
        print(e, '| 耗时(s) ：', time.time() - begin_time, "train loss : ", t_loss)
        if t_loss < min_loss:
            min_loss = t_loss
            torch.save(model.state_dict(), dir_path + '/cl_model' + '.pt')
            print('model saved! ')


def get_parameters():
    parser = argparse.ArgumentParser(description='Battery Life Predict')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cuda_index', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=90)
    parser.add_argument('--dataset_path', type=str, default='../data/dataset_1.pkl')
    parser.add_argument('--path', type=str, default='../save/')
    parser.add_argument('--d_model', type=int, default=32)
    # parser.add_argument('--pretrain_epoch', type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    params = vars(get_parameters())
    path = params['path']
    train(params, path)
