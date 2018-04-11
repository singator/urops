""" 
Running the primis dataset using pytorch on a CPU.

- Dataset (ok for now).
- Dataloader (ok for now).
- sample/split (ok for now).
- read data (ok for now).
- normalise? (ok for now).
- work with smaller dataset
- set up CNN model
- try decaying learning rate, different optimizer
- run on FH using:
floyd run --gpu+ --env pytorch-0.3 --data urops/datasets/primis_npy:/data 'python primis_pytorch_fh_gpu.py'
"""

import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Hyperparameters and such:
in_fname = '/data/primis_big.npy'
num_epochs = 50
num_steps = 100
learning_rate = 1e-6  # 0.001 seemed to work ok.
bs_test_val = 64
bs_training = 128
eval_every = 10

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

class Net(nn.Module):
    """ First attempt at CNN on primis """
    
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 2304)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

#    print('Dataset sizes for train, validation and test' + 
#            ' sets are {}, {} and {}.'.format(len(train_x), len(val_x), 
#                len(test_x)))
    print('-----')
    print('Running on GPU!')
    print('-----')

    # Create datasets and their loaders
    train_data = pkDataset([train_x, train_y], True)
    train_loader = DataLoader(train_data, batch_size=bs_training,
            shuffle=True, num_workers=2)

    val_data = pkDataset([val_x, val_y], True)
    val_loader = DataLoader(val_data, batch_size=bs_test_val,
            shuffle=False, num_workers=2)
    val_iter = iter(val_loader)

    test_data = pkDataset([test_x, test_y], True)
    test_loader = DataLoader(test_data, batch_size=bs_test_val,
            shuffle=False, num_workers=2)

    model = Net()
    model.cuda()
    print('Model instantiated')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    global_step = 0

    for epoch in range(num_epochs):
        for i,xy in enumerate(train_loader):
            x = Variable(xy['x'].cuda()).view(-1, 3, 32, 32)
            y = Variable(xy['y'].cuda()).view(-1)

            # Forward + Backward + Optimise
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            global_step += 1
            print('{{ "metric": "Training Loss", "value": {:.3f} }}'.format(loss.data[0]))
        
        if (epoch + 1) % eval_every == 0:
            try:
                xy = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                xy = next(val_iter)

            x = Variable(xy['x'].cuda(), volatile=True).view(-1, 3, 32, 32)
            y = xy['y'].cuda().view(-1)

            outputs = model(x)
            _,predicted = torch.max(outputs.data, 1)

            # print(predicted)
            correct = (predicted == y).sum()
            accuracy = correct / bs_test_val
            print('{{"metric": "Validation Accuracy", "value": {:.3f} }}'.format(accuracy))
