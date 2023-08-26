- d&c_no_split_data
不区分充放电，5个特征，插值1024。
['discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature', ]
transformer mape 7.78
transformer embedding +xgboost 7.80, +svm 8.08, +knn 7.95

- dc_split_d_data
区分充放电且充放都用，10个特征，插值500
['charge_voltage', 'charge_Q', 'charge_current', 'charge_time', 'charge_temperature', 'discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature']
transformer mape 11.60

- dataset_08240730
区分充放电且只用放电，加上deltaQ、deltaV、ir共8个特征，插值500。
['discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature', 'ir', 'delta_Q', 'delta_V']
transformer mape 12.45
cnn mape >25

- dataset_08240855
区分充放电且只用放电，加上deltaQ、deltaV、ir共8个特征，插值1024。
['discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature', 'ir', 'delta_Q', 'delta_V']
transformer mape 12.43

- dataset_08242354
区分充放电且只用放电，5个特征，插值1024。
['discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature']
transformer mape 24.3

- dataset_08250910
区分充放电且充放都用，加上deltaQ、deltaV、ir共13个特征，插值1024
feature1 = ['charge_voltage', 'charge_Q', 'charge_current', 'charge_time', 'charge_temperature', 'discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature', 'ir', 'delta_Q', 'delta_V']
transformer mape 14.72
