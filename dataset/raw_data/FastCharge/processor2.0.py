"""
读入dataset.pkl，
提取 充电电压，放电电压，充电电流，放电电流，充电温度，放电温度，充电内阻，放电内阻
Qdlin，Tdlin
"""
import sys
sys.path.append('/home/wangzaitian/work/2307/battery/BatteryLifePrediction')
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
from src.AEDataset import myDataset
import pickle
import random
import torch
import numpy as np

First = 5
MaxTIME = 40
# MaxPoint = 500
MaxPoint = 1024

LIFE = []


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


def keep_increasing_np(*args):
    """
    :param args: 计算电流梯度，确保特征递增
    :return:
    """
    re = list(args)
    indices = np.where(np.abs(np.gradient(re[0])) < 0.3)[0]
    for i in range(len(re)):
        re[i] = re[i][indices]

    c = re[0]
    mask = (c < -0.02) | (c > 0.02)
    indices = np.where(mask)[0]
    for i in range(len(re)):
        re[i] = re[i][indices]
    return re


def plot_curve(c, v, t, c_label='c', v_label='v', x_label='Time', y_label='Values', save_name='test.jpg'):
    # 绘制曲线图
    plt.cla()
    plt.plot(t, c, label=c_label, color='blue', marker='o')
    plt.plot(t, v, label=v_label, color='red', marker='s')

    # 添加图例和标签
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 显示图形
    plt.savefig("../../../figure/" + save_name)


def process(datadict):
    battery_num = 0
    MIT_battery = {}
    keys = list(datadict.keys())
    for x in tqdm(keys):
        data = datadict[x]
        life = int(data['cycle_life'][0])
        if life < 200:
            print("battery {} 's life is too short!".format(x))
            continue
        else:
            LIFE.append(life)
        battery_num += 1
        charge_voltage, discharge_voltage, discharge_current, charge_current, charge_temperature, \
        discharge_temperature, discharge_Q, charge_Q, Qdlin, Tdlin, dQdV, charge_time, discharge_time, charge_mask, discharge_mask = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        dq_dv_var, V_var, IR, dq_dv_min, summary_ct, summary_Tavg, summary_QD = [], [], [], [], [], [], []
        delta_time, delta_v_var, delta_q_mean, delta_q_var, cycle_index = [], [], [], [], []
        delta_Q, delta_V = [], []
        for j in range(First, life - 1):
            time = data['cycles'][str(j)]['t']
            v = data['cycles'][str(j)]['V']
            c = data['cycles'][str(j)]['I']
            t = data['cycles'][str(j)]['T']
            qc = data['cycles'][str(j)]['Qc']
            qd = data['cycles'][str(j)]['Qd']
            if np.max(time) > 4 * MaxTIME or np.max(time) < 0.3 * MaxTIME:  # 删除时间过大的循环
                print(j)
                continue

            c, v, t, time, qc, qd = keep_increasing_np(c, v, t, time, qc, qd)

            ch_s = 0
            ch_e = np.where(c < -0.1)[0][0] - 1
            dis_s = np.where(c < -3.9)[0][0]
            dis_e = np.where(v[dis_s:-1] <= 2.0)[0][0] + dis_s

            # 递增的充电阶段 暂不使用
            _v, _c, _t, _q, _time = v[ch_s:ch_e], c[ch_s:ch_e], t[ch_s:ch_e], qc[ch_s:ch_e], time[ch_s:ch_e]
            _v, _c, _t, _q, _time, mask = pad_array([_v, _c, _t, _q, _time], True, MaxTIME)  # 将时间pad，其余变量用最后一个值填充
            _v, _c, _t, _q, _time = interpolate_array([_v, _c, _t, _q, _time])  # 按照时间将其他变量对齐

            # if j % 100 == 0:
            #     plot_curve(_v, _c, _time, save_name=str(battery_num) + '_' + str(j) + "charge.jpg")
            charge_voltage.append(_v * mask)
            charge_current.append(_c * mask)
            charge_temperature.append(_t * mask)
            charge_Q.append(_q * mask)
            charge_time.append(_time * mask)
            charge_mask.append(mask)

            #   递减的放电阶段
            _v, _c, _t, _q, _time = v[dis_s:dis_e], c[dis_s:dis_e], t[dis_s:dis_e], qd[dis_s:dis_e], time[dis_s:dis_e]
            _v, _c, _t, _q, _time, mask = pad_array([_v, _c, _t, _q, _time], False, MaxTIME)  # 将时间pad，其余变量用最后一个值填充
            _v, _c, _t, _q, _time = interpolate_array([_v, _c, _t, _q, _time])  # 按照时间将其他变量对齐

            if j == First:  # 记录第一次循环的特征
                q_init = _q
                v_init = _v
                time_init = _time
            delta_Q.append(_q - q_init)
            delta_V.append(_v - v_init)
            delta_time.append(time_init - _time)
            delta_q_mean.append(np.mean(_q - q_init))
            delta_q_var.append(np.var(_q - q_init))
            delta_v_var.append(np.var(_v - v_init))
            IR.append(100 * data['summary']['IR'][j])
            discharge_time.append(_time * mask)  # dataset2.0中使用序列
            discharge_voltage.append(_v * mask)
            discharge_current.append(_c * mask)
            discharge_temperature.append(_t * mask)
            discharge_Q.append(_q * mask)
            discharge_mask.append(mask)
            cycle_index.append(life - j)

            Qdlin.append(list(data['cycles'][str(j)]['Qdlin']))
            Tdlin.append(list(data['cycles'][str(j)]['Tdlin']))
            dQdV.append(list(data['cycles'][str(j)]['dQdV']))
            summary_ct.append(data['summary']['chargetime'][j])
            summary_Tavg.append(data['summary']['Tavg'][j])
            summary_QD.append(data['summary']['QD'][j])

        battery = {
            'delta_v_var': delta_v_var,
            'delta_time': delta_time,
            'dq_dv_var': dq_dv_var,
            'dq_dv_min': dq_dv_min,
            'IR': IR,
            'V_var': V_var,
            'delta_q_mean': delta_q_mean,
            'delta_q_var': delta_q_var,
            'cycle_index': cycle_index,
            'summary_ct': summary_ct,
            'summary_Tavg': summary_Tavg,
            'summary_QD': summary_QD,

            'charge_voltage': charge_voltage,
            'charge_current': charge_current,
            'charge_Q': charge_Q,
            'charge_temperature': charge_temperature,
            'charge_time': charge_time,
            'charge_mask': charge_mask,

            "delta_Q": delta_Q,
            "delta_V": delta_V,
            'ir': np.divide(np.array(discharge_voltage), np.array(discharge_current),
                            out=np.zeros_like(discharge_voltage), where=np.array(discharge_current) != 0),
            'discharge_voltage': discharge_voltage,
            'discharge_current': discharge_current,
            'discharge_Q': discharge_Q,
            'discharge_temperature': discharge_temperature,
            'discharge_time': discharge_time,
            'discharge_mask': discharge_mask,
            'life': life,
            'charge_protocol': 'CCCV',
            'material': 'lithium-ion battery'}
        MIT_battery['MIT_battery_' + str(battery_num)] = battery

    # 显示图形
    return MIT_battery


def gen_dataset(data, name):
    for i in range(len(LIFE)):
        print(i, "<<", LIFE[i])
    print('waiting..')
    window, shift = 100, 10  # 窗口大小和移动步长

    # feature1 = ['charge_voltage', 'charge_Q', 'charge_current', 'charge_time', 'charge_temperature', 'discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature']
    feature1 = ['discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature', 'ir', 'delta_Q', 'delta_V']
    feature1 = ['discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature']
    feature1 = ['charge_voltage', 'charge_Q', 'charge_current', 'charge_time', 'charge_temperature', 'discharge_voltage', 'discharge_Q', 'discharge_current', 'discharge_time', 'discharge_temperature', 'ir', 'delta_Q', 'delta_V']
    feature2 = ['delta_v_var']

    print("｜{}｜{}".format(feature1, feature2))
    input_data, input_data_summary, label = [], [], []  # 预测的输入，前100个循环预测总寿命
    train_data, train_data_summary, train_target = [], [], []  # 训练的输入，按照窗口间隔生成训练数据
    test_data, test_data_summary, test_target = [], [], []  # 训练的输入，按照窗口间隔生成训练数据

    keys = list(data.keys())
    random.seed(1)
    random.shuffle(keys)
    idx = int(len(keys) * 0.75)

    # for battery_name in train_idx:
    for battery_name in keys[:idx]:
        battery = data[battery_name]
        cycle_life = battery['cycle_index']
        temp = []
        temp_summary = []
        for i in feature1:
            temp.append(battery[i])
        for j in feature2:
            temp_summary.append(battery[j])
        temp = torch.FloatTensor(np.array(temp))
        temp_summary = torch.FloatTensor(np.array(temp_summary))
        for x in range(0, len(cycle_life) - window + 1, shift):
            train_data.append(temp[:, x:x + window, :])
            train_target.append(cycle_life[x:x + window])
            train_data_summary.append(temp_summary[:, x:x + window])

    print("测试集：", keys[idx:])

    for battery_name in keys[idx:]:
        battery = data[battery_name]
        cycle_life = battery['cycle_index']
        label.append(battery['life'] - window - First)
        temp = []
        temp_summary = []
        for x in feature1:
            temp.append(battery[x])
        for i in feature2:
            temp_summary.append(battery[i])
        temp = torch.FloatTensor(np.array(temp))
        temp_summary = torch.FloatTensor(np.array(temp_summary))
        for x in range(0, len(cycle_life) - window + 1, shift):
            test_data.append(temp[:, x:x + window, :])
            test_target.append(cycle_life[x:x + window])
            test_data_summary.append(temp_summary[:, x:x + window])

        input_data.append(temp[:, :window, :])
        input_data_summary.append(temp_summary[:, :window])

    train_dataset = myDataset(torch.stack(train_data), torch.stack(train_data_summary), torch.tensor(train_target))
    early_dataset = myDataset(torch.stack(input_data), torch.stack(input_data_summary), torch.tensor(label))
    test_dataset = myDataset(torch.stack(test_data), torch.stack(test_data_summary), torch.tensor(test_target))

    re = {'train_dataset': train_dataset, 'early_dataset': early_dataset, 'test_dataset': test_dataset}
    with open(name, 'wb') as f:
        pickle.dump(re, f)
    return re


if __name__ == '__main__':
    print('process data...')
    path = 'dataset/pkl_data/FastCharge/dataset_08250910.pkl'
    print(path.split('/')[-1])
    dataset = pickle.load(open('dataset/pkl_data/FastCharge/dataset.pkl', 'rb'))
    del dataset[15]
    gen_dataset(process(dataset), path)
