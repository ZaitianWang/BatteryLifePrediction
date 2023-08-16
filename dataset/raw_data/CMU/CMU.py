import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def compute_life(c_capacity):
    min_time = len(min(c_capacity, key=lambda x: len(x))) - 1
    for i in range(5, len(c_capacity)):
        if c_capacity[i][min_time] <= 0.8 * c_capacity[5][min_time]:
            break
    return i


def split_cycle(dataframe):
    """
    :param is_end: When it is true, only the last set of pkl_data in each cycle is recorded
    :param dataframe: raw pkl_data
    :return: pkl_data spilt in cycle
    The goal is to split the pkl_data in each cycle,
    includes  current, voltage, discharge_capacity, charge_capacity and temperature
    """

    cycle_number = np.array(dataframe['cycleNumber'])
    index = np.where(np.diff(cycle_number) == 1)[0] + 1

    discharge_time = []
    discharge_voltage = []
    discharge_current = []
    discharge_capacity = []
    discharge_temperature = []
    charge_time = []
    charge_voltage = []
    charge_current = []
    charge_capacity = []
    charge_temperature = []

    all_time = np.array(dataframe['time_s'])
    all_charge_capacity = np.array(dataframe['QCharge_mA_h']) / 1000
    all_discharge_capacity = np.array(dataframe['QDischarge_mA_h']) / 1000
    all_current = np.array(dataframe['I_mA']) / 1000
    all_voltage = np.array(dataframe['Ecell_V'])
    all_temperature = np.array(dataframe['Temperature__C'])
    for j in range(len(index) - 1):
        time = all_time[index[j]:index[j + 1]]
        c = all_current[index[j]:index[j + 1]]
        v = all_voltage[index[j]:index[j + 1]]
        c_capa = all_charge_capacity[index[j]:index[j + 1]]
        d_capa = all_discharge_capacity[index[j]:index[j + 1]]
        t = all_temperature[index[j]:index[j + 1]]
        indices = np.argmax(c < 0)
        if indices == 0:
            continue
        charge_time.append(time[:indices])
        charge_voltage.append(v[:indices])
        charge_current.append(c[:indices])
        charge_capacity.append(c_capa[:indices])
        charge_temperature.append(t[:indices])
        discharge_time.append(time[indices:])
        discharge_voltage.append(v[indices:])
        discharge_current.append(c[indices:])
        discharge_capacity.append(d_capa[indices:])
        discharge_temperature.append(t[indices:])

    battery = {
        # 'charge_time': charge_time,
        # 'charge_current': charge_current,
        # 'charge_voltage': charge_voltage,
        # 'charge_capacity': charge_capacity,
        # 'charge_temperature': charge_temperature,
        'discharge_time': discharge_time,
        'discharge_current': discharge_current,
        'discharge_voltage':discharge_voltage,
        'discharge_capacity':discharge_capacity,
        'discharge_temperature': discharge_temperature,
        'life': compute_life(c_capacity=charge_capacity),
        'charge_protocol': 'CCCV',
        'material': 'lithium-ion battery'}
    return battery


if __name__ == '__main__':
    CMU_battery = {}
    # 
    for i in tqdm(range(1, 30+1)):
    # 
        try:
            filename = './[CMU]14226830/VAH' + str(i).rjust(2, '0') + '.csv'
            df = pd.read_csv(filename)
        except FileNotFoundError:
            continue
        cycle = split_cycle(df)
        CMU_battery['CMU_battery_' + str(i)] = cycle

    with open('./[CMU]14226830/pkl_data/CMU.pkl', 'wb') as f:
        pickle.dump(CMU_battery, f)
