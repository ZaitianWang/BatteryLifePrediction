"""
读入dataset.pkl，计算特征均值方差等。
"""
import gc
import pickle

import numpy as np
from tqdm import tqdm


def process(datadict, name):
    battery_num = 0
    MIT_battery = {}
    for x in tqdm(datadict):
        data = datadict[x]
        life = int(data['cycle_life'][0])
        battery_num += 1
        charge_voltage, discharge_voltage, discharge_current, charge_current, charge_temperature, \
        discharge_temperature, discharge_capacity, charge_capacity = [], [], [], [], [], [], [], []
        for j in range(1, life - 1):
            v = data['cycles'][str(j)]['V']
            c = data['cycles'][str(j)]['I']
            t = data['cycles'][str(j)]['T']

            capa = data['cycles'][str(j)]['Qc']
            dis_capa = data['cycles'][str(j)]['Qdlin']  # 经过线性处理的数据，放电
            dis_T = data['cycles'][str(j)]['Tdlin']
            indices = np.where(c[50:] < 0)[0][1] - 1 + 50
            if indices == 0 or indices >= len(c):
                print(c)
                continue
            charge_voltage.append(v[:indices])
            charge_current.append(c[:indices])
            charge_temperature.append(t[:indices])
            charge_capacity.append(capa[:indices])
            discharge_voltage.append(v[indices:])
            discharge_current.append(c[indices:])
            discharge_capacity.append(dis_capa)
            discharge_temperature.append(dis_T)

        battery = {
            'charge_capacity': charge_capacity,
            'charge_temperature': charge_temperature,
            'discharge_temperature': discharge_temperature,
            'discharge_capacity': discharge_capacity,
            #    voltage is represented by capacity
            'discharge_voltage': discharge_voltage,
            'charge_voltage': charge_voltage,
            'charge_current': charge_current,
            'discharge_current': discharge_current,
            'life': life,
            'charge_protocol': 'CCCV',
            'material': 'lithium-ion battery'}
        MIT_battery['MIT_battery_' + str(battery_num)] = battery
    with open('../../pkl_data/' + name + '.pkl', 'wb') as f:
        pickle.dump(MIT_battery, f)


if __name__ == '__main__':

    trainMIT_battery = {}
    testMIT_battery = {}
    sec_testMIT_battery = {}

    dataset = pickle.load(open('dataset.pkl', 'rb'))
    for i in range(1, 83, 2):
        trainMIT_battery[i + 1] = dataset[i + 1]
        testMIT_battery[i] = dataset[i]
    trainMIT_battery[84] = dataset[84]
    for i in range(85, len(dataset) + 1, 1):
        sec_testMIT_battery[i] = dataset[i]
    del dataset
    gc.collect()
    print('process data...')
    process(trainMIT_battery, 'trainMIT_battery')
    process(sec_testMIT_battery, 'sec_testMIT_battery')
    process(testMIT_battery, 'testMIT_battery')
