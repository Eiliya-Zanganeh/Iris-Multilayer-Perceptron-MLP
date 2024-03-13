import pandas as pd
from torch.utils.data import Dataset as TorchDataSet
from torch import Tensor


class DataSet(TorchDataSet):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.data.loc[self.data['species'] == 'Iris-setosa', 'species'] = 0  # 0 -> Iris-setosa
        self.data.loc[self.data['species'] == 'Iris-versicolor', 'species'] = 1  # 0 -> Iris-versicolor
        self.data.loc[self.data['species'] == 'Iris-virginica', 'species'] = 2  # 1 -> Iris-virginica

        self.data = self.data.apply(pd.to_numeric)

        self.data = self.data.values

        self.X = Tensor(self.data[:, :4]).float()
        self.Y = Tensor(self.data[:, 4]).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
