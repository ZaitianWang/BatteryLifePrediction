"""
读取原始数据，提取特征，并删除故障电池
"""
import h5py
import numpy as np
import pickle


def p1(matFilename):
    f = h5py.File(matFilename)
    batch = f['batch']

    num_cells = batch['summary'].shape[0]
    bat_dict = {}
    for i in range(num_cells):
        cl = f[batch['cycle_life'][i, 0]][:]
        policy = f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
        summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
        summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
        summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
        summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
        summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
        summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
        summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
            summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                   'cycle': summary_CY}
        cycles = f[batch['cycles'][i, 0]]
        cycle_dict = {}
        for j in range(cycles['I'].shape[0]):
            I = np.hstack(f[cycles['I'][j, 0]][:])
            Qc = np.hstack(f[cycles['Qc'][j, 0]][:])
            Qd = np.hstack(f[cycles['Qd'][j, 0]][:])
            Qdlin = np.hstack(f[cycles['Qdlin'][j, 0]][:])
            T = np.hstack(f[cycles['T'][j, 0]][:])
            Tdlin = np.hstack(f[cycles['Tdlin'][j, 0]][:])
            V = np.hstack(f[cycles['V'][j, 0]][:])
            dQdV = np.hstack(f[cycles['discharge_dQdV'][j, 0]][:])
            t = np.hstack(f[cycles['t'][j, 0]][:])
            cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
            cycle_dict[str(j)] = cd

        cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
        key = 'b1c' + str(i)
        bat_dict[key] = cell_dict

    return bat_dict


if __name__ == '__main__':
    filename = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
                '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
                '2018-04-12_batchdata_updated_struct_errorcorrect.mat']
    batch1 = p1(filename[0])
    batch2 = p1(filename[1])
    batch3 = p1(filename[2])

    # remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']
    numBat1 = len(batch1.keys())
    print(numBat1)

    batch2_keys = ['b1c7', 'b1c8', 'b1c9', 'b1c15', 'b1c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]
    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack(
                    (batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

    del batch2['b1c7']
    del batch2['b1c8']
    del batch2['b1c9']
    del batch2['b1c15']
    del batch2['b1c16']
    numBat2 = len(batch2.keys())
    print(numBat2)
    # remove noisy channels from batch3
    del batch3['b1c37']
    del batch3['b1c2']
    del batch3['b1c23']
    del batch3['b1c32']
    del batch3['b1c42']
    del batch3['b1c43']
    numBat3 = len(batch3.keys())
    print(numBat3)
    dataset = {}
    count = 1
    for i in batch1:
        dataset[count] = batch1[i]
        count += 1
    for i in batch2:
        dataset[count] = batch2[i]
        count += 1
    for i in batch3:
        dataset[count] = batch3[i]
        count += 1
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
