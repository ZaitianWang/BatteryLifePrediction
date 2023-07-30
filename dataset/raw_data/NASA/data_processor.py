import pickle

import numpy as np
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt

# convert str to datatime
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# load .mat pkl_data
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split("/")[-1].split(".")[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['pkl_data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


# get capacity pkl_data
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['pkl_data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]


# get the charge pkl_data of a battery
def getBatteryValues(Battery, Type='charge'):
    data=[]
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['pkl_data'])
    return data


# 提取锂电池容量
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['pkl_data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]


# 获取锂电池充电或放电时的测试数据
def getBatteryValues(Battery, Type='charge'):
    data = []
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['pkl_data'])
    return data


def get_feature(b_data):
    count = 0
    battery = {}
    charge_voltage, discharge_capacity, charge_current = [], [], []
    discharge_voltage = []
    discharge_current = []
    discharge_temperature = []
    charge_temperature = []
    for i in b_data:
        if i['type'] == 'charge':
            charge_voltage.append(np.array(i['pkl_data']['Voltage_measured']))
            charge_current.append(np.array(i['pkl_data']['Current_measured']))
            charge_temperature.append(np.array(i['pkl_data']['Temperature_measured']))
        elif i['type'] == 'discharge':
            discharge_voltage.append(np.array(i['pkl_data']['Voltage_measured']))
            discharge_current.append(np.array(i['pkl_data']['Current_measured']))
            discharge_temperature.append(np.array(i['pkl_data']['Temperature_measured']))
            discharge_capacity.append(np.array([i['pkl_data']['Capacity'][0]]))
        else:
            count += 1
    battery = {
        'charge_temperature': charge_temperature,
        'discharge_temperature': discharge_temperature,
        'discharge_capacity': discharge_capacity,
        'discharge_voltage': discharge_voltage,
        'charge_voltage': charge_voltage,
        'charge_current': charge_current,
        'discharge_current': discharge_current,
        'life': -1,
        'charge_protocol': 'CCCV',
        'material': 'lithium-ion battery'}
    return battery


Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']  # 4 个数据集的名字
dir_path = ''

NASA_battery = {}
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    NASA_battery['NASA_battery_'+name] = get_feature(data)
with open('../../pkl_data/NASA_battery.pkl', 'wb') as f:
    pickle.dump(NASA_battery, f)
