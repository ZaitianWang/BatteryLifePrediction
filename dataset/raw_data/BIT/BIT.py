import glob
import pickle
from random import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
from src.AEDataset import myDataset

MaxTIME = 1500
MaxPoint = 500


def interpolate_array(arg_list):
    time = arg_list[-1]
    # 均匀插值
    for x in range(len(arg_list)):
        f = interp1d(time, arg_list[x])
        new_x = np.linspace(np.min(time), MaxTIME, MaxPoint)
        arg_list[x] = f(new_x)
    return arg_list


def pad_array(arr_list, charge=True, max_=MaxTIME):
    time = arr_list[-1]
    if not charge:
        time = time - np.min(time)
    mask = time >= 0
    padding_len = MaxPoint - len(time)
    if padding_len <= 0:
        mask = mask[:MaxPoint]
    else:
        mask = np.concatenate((mask, np.repeat(False, padding_len)))
    # 计算需要填充的长度
    pad_seg = np.linspace(np.max(time), max_, 2)
    for i in range(len(arr_list) - 1):
        seg = np.repeat(arr_list[i][-2], len(pad_seg))
        arr_list[i] = np.concatenate((arr_list[i], seg))
    arr_list[-1] = np.concatenate((time, pad_seg))
    arr_list.append(mask)

    return arr_list


def split_cycle(dataframe, ):
    """
    :param dataframe: raw pkl_data
    :return: pkl_data spilt in cycle
    The goal is to split the pkl_data in each cycle,
    includes  current, voltage, discharge_capacity, charge_capacity and temperature
    """
    cycle_number = np.array(dataframe['Cycle_Index'])
    index = np.where(np.diff(cycle_number) == 1)[0] + 1

    charge_voltage, discharge_capacity, charge_current = [], [], []
    delta_Q, delta_V, delta_time = [], [], []
    discharge_voltage, discharge_current, discharge_time, charge_temperature = [], [], [], []
    all_capacity = np.array(dataframe['Capacity(Ah)'])
    all_current = np.array(dataframe['Current(A)'])
    all_voltage = np.array(dataframe['Voltage(V)'])
    all_temperature = np.array(dataframe['Temperature(℃)'])
    # 跳过第一次循环
    for j in range(len(index) - 1):
        c = all_current[index[j]:index[j + 1]]
        v = all_voltage[index[j]:index[j + 1]]
        capa = all_capacity[index[j]:index[j + 1]]
        t = all_temperature[index[j]:index[j + 1]]
        # capa = [0,..., 0,...,] charge and discharge. Get the index of the second zero
        indices = np.where(capa == 0)[0][1]
        # charge_voltage.append(v[:indices])
        # charge_current.append(c[:indices])
        # charge_temperature.append(t[:indices])

        ds_v, ds_c, ds_cap = v[indices:], c[indices:], capa[indices:]
        ds_time = range(len(ds_cap))
        ds_v, ds_c, ds_cap, ds_time, mask = pad_array([ds_v, ds_c, ds_cap, ds_time], False, MaxTIME)  # 将时间pad，其余变量用最后一个值填充
        ds_v, ds_c, ds_cap, ds_time = interpolate_array([ds_v, ds_c, ds_cap, ds_time])  # 按照时间将其他变量对齐
        if j == 0:  # 记录第一次循环的特征
            ds_cap_init = ds_cap
            v_init = ds_v
            time_init = ds_time
        delta_Q.append(ds_cap - ds_cap_init)
        delta_V.append(ds_v - v_init)
        delta_time.append(ds_time - time_init)
        discharge_voltage.append(ds_v)
        discharge_current.append(ds_c)
        discharge_capacity.append(ds_cap)
        discharge_time.append(ds_time)

    battery = {
        'ir': np.divide(np.array(discharge_voltage), np.array(discharge_current),
                        out=np.zeros_like(discharge_voltage), where=np.array(discharge_current) != 0),
        'delta_V': delta_V,
        'delta_Q': delta_Q,
        'delta_time': delta_time,
        'discharge_time': discharge_time,
        'discharge_Q': discharge_capacity,
        'discharge_voltage': discharge_voltage,
        'discharge_current': discharge_current,
        'life': len(delta_V),
        'charge_protocol': 'CCCV',
        'material': 'lithium-ion battery'}
    return battery


def gen_dataset(data, name):
    print('waiting..')
    window, shift = 200, 10  # 窗口大小和移动步长
    feature1 = ['discharge_voltage', 'discharge_current', 'discharge_Q', 'discharge_time', 'ir', 'delta_V',
                'delta_time']
    print("｜{}｜".format(feature1))
    input_data, input_data_summary, label = [], [], []  # 预测的输入，前100个循环预测总寿命
    train_data, train_data_summary, train_target = [], [], []  # 训练的输入，按照窗口间隔生成训练数据
    test_data, test_data_summary, test_target = [], [], []  # 训练的输入，按照窗口间隔生成训练数据

    keys = list(data.keys())
    random.seed(0)
    random.shuffle(keys)
    # train_idx = [keys[i] for i in range(1, 84, 2)]
    # val_cells = [keys[i] for i in range(0, 84, 2)]
    # secondary_test_idx = [keys[i] for i in range(84, 124)]
    for battery_name in keys[:int(len(keys) * 0.75)]:
        battery = data[battery_name]
        life = battery['cycle_index']
        temp = []
        for i in feature1:
            temp.append(battery[i])
        temp = torch.FloatTensor(np.array(temp))
        is_end = False
        for x in range(0, battery['life'], shift):
            dt = temp[:, x:x + window, :]
            dt_life = life[x:x + window]
            if dt.shape[1] < window:
                # 不足一个window的数据进行填充，注意在有效数据前填充0
                pad_num = (window - dt.shape[1])
                zeros = torch.zeros(dt.shape[0], pad_num, dt.shape[2])
                dt = torch.cat([zeros, dt], dim=1)
                dt_life = [-1] * pad_num + dt_life
                is_end = True  # 发生填充时，提前退出循环
            train_target.append(dt_life)
            train_data.append(dt)
            if is_end:
                break

    for battery_name in keys[:int(len(keys) * 0.75):]:
        battery = data[battery_name]
        life = battery['cycle_index']
        label.append(battery['life'] - window - 5)
        temp = []
        for x in feature1:
            temp.append(battery[x])
        temp = torch.FloatTensor(np.array(temp))
        is_end = False
        for x in range(0, battery['life'], shift):
            dt = temp[:, x:x + window, :]
            dt_life = life[x:x + window]
            if dt.shape[1] < window:
                pad_num = (window - dt.shape[1])
                zeros = torch.zeros(dt.shape[0], pad_num, dt.shape[2])
                dt = torch.cat([zeros, dt], dim=1)
                dt_life = [-1] * pad_num + dt_life
                is_end = True  # 发生填充时，提前退出循环
            test_data.append(dt)
            test_target.append(dt_life)
            if is_end:
                break
        input_data.append(temp[:, :window, :])

    train_dataset = myDataset(torch.stack(train_data), torch.tensor(train_target))
    early_dataset = myDataset(torch.stack(input_data), torch.tensor(label))
    test_dataset = myDataset(torch.stack(test_data), torch.tensor(test_target))

    re = {'train_dataset': train_dataset, 'early_dataset': early_dataset, 'test_dataset': test_dataset}
    with open(name, 'wb') as f:
        pickle.dump(re, f)
    return re


if __name__ == '__main__':
    path = '../../pkl_data/BIT_battery.pkl'
    BIT_battery = {}
    count = 0
    for filename in tqdm(glob.glob('*.xlsx')):
        count += 1
        df = pd.read_excel(filename)
        battery = split_cycle(df, )
        BIT_battery['BIT_battery_' + str(count)] = battery
        print(battery['life'])
    gen_dataset(BIT_battery, path)

