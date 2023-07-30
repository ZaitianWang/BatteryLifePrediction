import pickle

import torch
import torch.utils.data as data
import numpy as np


class AEDataset(data.Dataset):
    def __init__(self, x):
        super(AEDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


class PreDataset(data.Dataset):
    def __init__(self, dataset):
        super(PreDataset, self).__init__()
        self.x = []
        self.y = []
        for i in dataset:
            self.x.append(i[0])
            self.y.append(i[1])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


class earlyDataset(data.Dataset):
    def __init__(self, summary, cycles, labels, window_size, ):
        self.early_summary, self.early_cycles, self.early_labels = [], [], []
        super(earlyDataset, self).__init__()
        self.init_data(summary, cycles, labels, window_size)

    def __len__(self):
        return len(self.early_labels)

    def __getitem__(self, index):
        return self.early_summary[index], self.early_cycles[index], self.early_labels[index]

    def init_data(self, summary, cycles, labels, window_size):
        for i in range(len(cycles)):
            self.early_summary.append(summary[i][: window_size])
            self.early_cycles.append(cycles[i][:window_size])
            self.early_labels.append(labels[i][window_size])
        self.early_summary = np.array(self.early_summary)
        self.early_cycles = np.array(self.early_cycles)
        self.early_labels = np.array(self.early_labels)


class myDataset(data.Dataset):
    def __init__(self, cycles, summary, labels):
        super(myDataset, self).__init__()
        self.cycles, self.labels, self.summary = cycles, labels, summary
        self.cycle_len = cycles.shape[1]
        self.summary_len = summary.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.cycles[index], self.summary[index],self.labels[index]
