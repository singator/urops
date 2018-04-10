import pandas as pd
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

# Hyperparameters:
input_size = 4
hidden_size = 10
num_classes = 3
num_epochs = 600
training_bs = 60
test_bs = 15
learning_rate = 0.03
eval_every = 25

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
        samp_mat = sample.as_matrix().reshape(-1, 5)
        x = torch.Tensor(samp_mat[:, :-1])
        y = torch.LongTensor(samp_mat[:, -1])
        sample_dict = {'x': x, 'y':y}
#        sample_dict = {'batch': sample.values}
        return sample_dict

class Net(nn.Module):
    """ Simple Deep Learning Network """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    train_pd = pd.read_csv('/data/iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
    test_pd = pd.read_csv('/data/iris_test.csv', names=CSV_COLUMN_NAMES, header=0)

    # Create data loaders for training and test sets.
    train_torch_dataset = IrisDataset(train_pd)
    train_torch_loader = DataLoader(train_torch_dataset, batch_size=training_bs, 
            shuffle=True, num_workers=2)

    test_torch_dataset = IrisDataset(test_pd)
    test_torch_loader = DataLoader(test_torch_dataset, batch_size=test_bs,
            shuffle=True, num_workers=2)

    print('------')
    print('Running on a GPU')
    print('------')

    net = Net(input_size, hidden_size, num_classes)
    if torch.cuda.is_available():
        net = net.cuda()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    global_step = 0

    for epoch in range(num_epochs):
        for i,xy in enumerate(train_torch_loader):
            x = Variable(xy['x'].cuda()).view(-1, 4)
            y = Variable(xy['y'].cuda()).view(-1)

            # Forward + Backward + Optimise
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            global_step += 1
        
        print('Epoch {}, step {} completed. Loss: {:.3f}'.format(epoch+1, 
          global_step, loss.data[0]))

        # Evaluate on VALIDATION set:
        if (epoch + 1) % eval_every == 0:
            total = test_bs

            xy = next(iter(test_torch_loader))
            x = Variable(xy['x'].cuda()).view(-1, 4)
            y = xy['y'].cuda().view(-1)
            outputs = net(x)
            print('reached here')
            _,predicted = torch.max(outputs.data, 1)
            correct = (predicted == y).sum()
            accuracy = correct / total
            print('Accuracy on val set: {:.2f}'.format(accuracy))