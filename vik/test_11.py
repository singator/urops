import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']


class IrisDataset(Dataset):
    """ Iris Dataset """

    def __init__(self, pd_dframe, transform = None):
        """
        Args:
            pd_dframe (pd dataframe): pandas data frame containing the data.
            transform (callable, optional): Optional transform to be applied to the 
            data.
        """
        self.data = pd_dframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        return sample

if __name__ == '__main__':
    train_pd = pd.read_csv('../data/iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
    test_pd = pd.read_csv('../data/iris_test.csv', names=CSV_COLUMN_NAMES, header=0)

    # Create data loaders for training and test sets.
    train_torch_dataset = IrisDataset(train_pd)
    train_torch_loader = DataLoader(train_torch_dataset, batch_size=4, 
            shuffle=True, num_workers=2)

    test_torch_dataset = IrisDataset(test_pd)
    test_torch_loader = DataLoader(test_torch_dataset, batch_size=10, 
            shuffle=True, num_workers=2)

    # Test loader
    for x in test_torch_loader:
        print(x)

