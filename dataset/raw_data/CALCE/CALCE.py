import glob
import os

import numpy as np
import pandas as pd

filePath = ''
for i in os.walk(filePath):
    dir_names = i[1]
    break

CALCE_battery = {}
count = 0
# for i in dir_names:
        # count += 1
        # df = pd.read_excel(file, sheet_name=1)
        # battery = {'current': np.array(df['Current(A)']), 'voltage': np.array(df['Voltage(V)']),
        #            'capacity': np.array(df['Discharge_Capacity(Ah)']), 'cycle_number': np.array(df['Cycle_Index'])}
        # battery['life'] = np.max(battery['cycle_number'])
        # battery['charge_protocol'] = 'CCCV'
        # battery['material'] = 'lithium-ion battery'
    # CALCE_battery['CALCE_battery_' + str(count)] = battery
print(CALCE_battery)
