""" 
Running the primis dataset using pytorch on a CPU.

- Dataset (ok for now).
- Dataloader (ok for now).
- sample/split (ok for now).
- read data (ok for now).
- normalise? (ok for now).
- work with smaller dataset
- set up CNN model
"""

import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Hyperparameters and such:
in_fname = '../data/primis_small.npy'

class pkDataset(Dataset):
    """ Parking Lot Dataset Constructor"""

    def __init__(self, feat_labels, transform=None):
        """
        Args:
            feat_labels (list of len 2): a list containing features in pos 0
            and labels in pos 1.
            transform (boolean): This should be True if we wish to normalise
            the data into [0 .. 1]. This is done manually for now. We should 
            read up on the transforms and PILL to see how they do it. Why can't
            we call self.transform(sample)??
        """
        self.data = feat_labels
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x_sample = self.data[0][idx]
        y_sample = self.data[1][idx]

        y_sample = y_sample.reshape(-1,)
        y_sample = np.array(y_sample)

        x_sample = x_sample.reshape(-1, 32, 32, 3)
        x_sample = np.array(x_sample.transpose((0, 3, 1, 2)))

        if self.transform:
            x_sample = x_sample / 256.0

        return {'x': torch.FloatTensor(x_sample),
                'y': torch.LongTensor(y_sample)}

if __name__ == '__main__':

    with open(in_fname, 'rb') as f:
        pX,pY = pickle.load(f)

    # Split the data into training, validation and test sets. For now, we use
    # percentage 80/10/10.
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2228)
    for tr_id,te_id in split.split(pX, pY):
      train_x = pX[tr_id]
      train_y = pY[tr_id]

      test_x =  pX[te_id]
      test_y = pY[te_id]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=2229)
    for tr_id,te_id in split.split(test_x, test_y):
      val_x = test_x[tr_id]
      val_y = test_y[tr_id]

      test_x = test_x[te_id]
      test_y = test_y[te_id]

    print('Dataset sizes for train, validation and test' + 
            ' sets are {}, {} and {}.'.format(len(train_x), len(val_x), 
                len(test_x)))

    # Create datasets and their loaders
    train_data = pkDataset([train_x, train_y], True)
    train_loader = DataLoader(train_data, batch_size=10,
            shuffle=True, num_workers=2)

    val_data = pkDataset([val_x, val_y], True)
    val_loader = DataLoader(val_data, batch_size=10,
            shuffle=False, num_workers=2)

    test_data = pkDataset([test_x, test_y], True)
    test_loader = DataLoader(test_data, batch_size=10,
            shuffle=False, num_workers=2)

    nnet = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            )

    for i, xy in enumerate(test_loader):
        x = Variable(xy['x']).view(-1, 3, 32, 32)
        outputs=nnet(x)

    print(outputs.data)

#    all_data_loader = DataLoader(all_data, batch_size = 16,
#            shuffle = True, num_workers=4)
#
#    ii = iter(all_data_loader)
#    print(next(ii))
