import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_true_indices(lst):
    lst = np.array(lst)
    true_indices = np.where(lst)[0]
    diff = np.diff(true_indices, prepend=-1)
    return true_indices[diff != 1]


def process(all_data):
    discharge_voltage, discharge_current, discharge_capacity, discharge_temperature = [], [], [], []
    charge_voltage, charge_current, charge_capacity, charge_temperature = [], [], [], []
    index = np.where(np.diff(all_data[:, -1]) == 1)[0] + 1
    for j in range(1, len(index) - 1):
        cycle_data = all_data[index[j - 1]:index[j], :].T
        slice_index = np.argmax(cycle_data[3] > 0)
        if slice_index == 0:
            return {'charge_current': []}
        discharge_voltage.append(cycle_data[0][slice_index:])
        charge_voltage.append(cycle_data[0][:slice_index])
        discharge_current.append(cycle_data[1][slice_index:])
        charge_current.append(cycle_data[1][:slice_index])
        discharge_temperature.append(cycle_data[4][slice_index:])
        charge_temperature.append(cycle_data[4][:slice_index])
        discharge_capacity.append(cycle_data[2][slice_index:])
        charge_capacity.append(cycle_data[2][:slice_index])

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


if __name__ == '__main__':
    UNIBO_battery = {}
    c = ['voltage', 'current', 'charging_capacity', 'discharging_capacity', 'temperature', 'cycle_count']
    file = np.array(pd.read_csv('test_result.csv', usecols=c).values)
    battery_index = get_true_indices(file[:, -1] == 1)[1:]
    for i in tqdm(range(3, len(battery_index) - 1)):
        bat = process(file[battery_index[i]:battery_index[i + 1]])
        if 0 < len(bat['charge_current']) < 101:
            UNIBO_battery['UNIBO_battery_' + str(i)] = bat
    with open('../../pkl_data/UNIBO_battery.pkl', 'wb') as f:
        pickle.dump(UNIBO_battery, f)
