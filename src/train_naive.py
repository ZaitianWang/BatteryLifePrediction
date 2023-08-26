from os import device_encoding
import sys
sys.path.append('/home/wangzaitian/work/2307/battery/BatteryLifePrediction')

from utils import *
import argparse
import random

from torch.utils.data import DataLoader
from layer.Net import *
import time

torch.set_num_threads(1)


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
    # device = 'cpu'
    dir_path = dir_path + get_date_time_second_string()
    os.mkdir(dir_path)
    logger = init_log(dir_path + '/predict.log')
    dataset = pickle.load(open(nni_params['dataset_path'], 'rb'))
    train_dataset, early_dataset, test_dataset = dataset['train_dataset'], dataset['early_dataset'], dataset[
        'test_dataset']
    loader = DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=nni_params['batch_size'],
                        drop_last=True)

    early_dataset = DataLoader(early_dataset,
                               shuffle=False,
                               batch_size=nni_params['batch_size'],
                               drop_last=False)

    test_dataset = DataLoader(test_dataset,
                              shuffle=False,
                              batch_size=nni_params['batch_size'],
                              drop_last=False)
    # model = torch.compile(JoinModel(train_dataset.summary_len, train_dataset.cycle_len).to(device))
    model = JoinModel(train_dataset.cycle_len).to(device) # my main transformer model
    # model = CNN(train_dataset.cycle_len).to(device) # my CNN model
    # state_dict = torch.load("../save/cl_model.pt")
    # new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    # model.load_state_dict(new_state_dict)
    model = torch.compile(model)

    print("loading data form ", nni_params['dataset_path'])
    print("starting training.... 模型参数量(单位： W) ：")
    print(sum(p.numel() for p in model.parameters()) / 10000)
    optimizer = torch.optim.Adam(model.parameters(), lr=nni_params['lr'])
    min_err = 25
    logger.info("loading data from " + nni_params['dataset_path'])
    for e in range(nni_params['epoch']):
        total_loss = []
        begin_time = time.time()
        for step, batch in enumerate(loader):
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y[:, -1].to(device)
            # print(x1.shape, x2.shape, y.shape)
            y_pred = model(x1)
            # y_pred = model(x1, y)
            loss = mape(actual=y, pred=y_pred)
            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 1:
                logger.info("{} |  loss : {:3f} ".format(e, loss))
        err1 = evaluate_early(nni_params, early_dataset, model)
        print(e, '| 耗时(s) ：', time.time() - begin_time, "train loss : ", np.mean(total_loss) * 100,
              '  | early prediction error : ', err1, )
        logger.info(
            'epoch:{}, train_loss: {}, early_prediction error: {} | '.format(e, np.mean(total_loss) * 100, err1))
        if min_err > err1:
            flag = True
            err2 = evaluate_RUL(nni_params, test_dataset, model)
            min_err = err1
            logger.info(
                'The best model was saved! Epoch: {}, min_error : {:3f}, RUL error : {:3f}'.format(e, min_err, err2))
            torch.save(model.state_dict(), dir_path + '/' + str(err1) + get_date_time_second_string() + '.pth')
            print('model saved! RUL error: ', err2)


def evaluate_early(args, loader, model):
    device = 'cuda:' + str(args['cuda_index']) if torch.cuda.is_available() else 'cpu'
    predictions = []
    label_ = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            # y_pred = model(x1, x2).cpu().tolist()
            y_pred = model(x1).cpu().tolist()
            label_.extend(y.tolist())
            predictions.extend(y_pred)

    er = mape(pred=torch.tensor(predictions), actual=torch.tensor(label_)).item() * 100
    return er


# def evaluate_early(args, loader, model):
#     device = 'cuda:' + str(args['cuda_index']) if torch.cuda.is_available() else 'cpu'
#     predictions = []
#     label_ = []
#     with torch.no_grad():
#         for step, batch in enumerate(loader):
#             x, y = batch
#             y += 100
#             first_100 = x[:, :, :100, :]
#             x = torch.cat([torch.zeros_like(first_100), first_100], dim=2)
#             pre = model(x.to(device)).cpu().tolist()
#             if type(pre) == list:
#                 label_.extend(y.tolist())
#                 predictions.extend(pre)
#             else:
#                 label_.append(int(y))
#                 predictions.append(pre)
#
#     er = mape(pred=torch.tensor(predictions), actual=torch.tensor(label_)).item() * 100
#     return er


def evaluate_RUL(args, loader, model):
    device = 'cuda:' + str(args['cuda_index']) if torch.cuda.is_available() else 'cpu'
    predictions = []
    label_ = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y[:, -1].to(device)
            # y_pred = model(x1, x2).cpu().tolist()
            y_pred = model(x1).cpu().tolist()
            label_.extend(y.tolist())
            predictions.extend(y_pred)
    er = mape(pred=torch.tensor(predictions), actual=torch.tensor(label_)).item() * 100
    return er


def get_parameters():
    parser = argparse.ArgumentParser(description='Battery Life Predict')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--cuda_index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--dataset_path', type=str, default='dataset/pkl_data/FastCharge/d&c_no_split_data.pkl')
    # parser.add_argument('--dataset_path', type=str, default='dataset/pkl_data/FastCharge/dc_split_d_data.pkl')    
    parser.add_argument('--dataset_path', type=str, default='dataset/pkl_data/FastCharge/dataset_08240730.pkl')
    parser.add_argument('--path', type=str, default='save/')
    parser.add_argument('--d_model', type=int, default=32)
    # parser.add_argument('--pretrain_epoch', type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    params = vars(get_parameters())
    path = params['path']
    train(params, path)
