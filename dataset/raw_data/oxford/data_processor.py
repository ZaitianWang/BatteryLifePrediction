import numpy as np
import scipy.io


def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    print(data['Cell1']['annotations']
          )
    filename = matfile.split('/')[-1].split('.')[0]
    charge = data['Cell1'][0][0]
    voltage = data['Cell1'][0][0][1]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['pkl_data'] = str(col[i][0][0]), int(col[i][1][0]), str(
            col[i][2][0]), d2
        data.append(d1)

    return data


Battery_list = 'Oxford_Battery_Degradation_Dataset_1.mat'

loadMat(Battery_list)
