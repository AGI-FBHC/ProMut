import numpy as np
import torch.utils.data as Data


class NpzDataset(Data.Dataset):  #用于处理.npz文件格式的数据。
    def __init__(self, data_file, label_file,is_training=False):
        self.data = np.load(data_file)['arr_0']
        self.label = np.load(label_file)['arr_0']


    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]


        return x, y

    def __len__(self):
        return len(self.label)
