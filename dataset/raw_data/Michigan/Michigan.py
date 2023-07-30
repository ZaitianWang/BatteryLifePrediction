import pickle

import numpy as np
import pandas as pd


def processor(dataframe):
    cycle_number = np.array(dataframe['Cycle number'])
    index = np.where(np.diff(cycle_number) == 1)[0] + 1

    charge_voltage, discharge_capacity, charge_current = [], [], []
    discharge_voltage = []
    discharge_current = []
    discharge_temperature = []
    charge_temperature = []
    all_capacity = np.array(df['Capacity [Ah]'])
    all_current = np.array(df['Current [mA]']) / 1000
    all_voltage = np.array(df['Voltage [V]'])
    all_temperature = np.array(df['Temperature [C]'])
    for j in range(len(index) - 1):
        c = all_current[index[j]:index[j + 1]]
        v = all_voltage[index[j]:index[j + 1]]
        capa = all_capacity[index[j]:index[j + 1]]
        t = all_temperature[index[j]:index[j + 1]]
        indices = np.argmax(c < 0)
        if indices == 0:
            continue
        charge_voltage.append(v[:indices])
        charge_current.append(c[:indices])
        charge_temperature.append(t[:indices])

        discharge_voltage.append(v[indices:])
        discharge_current.append(c[indices:])
        discharge_capacity.append(capa[indices:])
        discharge_temperature.append(t[indices:])

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

    Michigan_battery = {}
    for i in range(1, 22):
        filename = str(i) + '/cycling_wExpansion.csv'
        df = pd.read_csv(filename)
        battery = processor(df)
        battery['life'] = -1
        battery['charge_protocol'] = 'CCCV'
        battery['material'] = 'lithium-ion battery'
        Michigan_battery['Michigan_battery_' + str(i)] = battery

    with open('../../pkl_data/Michigan_battery.pkl', 'wb') as f:
        pickle.dump(Michigan_battery, f)
