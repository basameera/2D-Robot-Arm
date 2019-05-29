"""
Sameera Sandaruwan
"""

# Importing PyTorch tools
# from torch import nn, optim, cuda

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import cuda

# Importing other libraries
import numpy as np
import matplotlib.pyplot as plt
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Model
class NeuralNet(nn.Module):

    def __init__(self, optimizer=optim.SGD, lr=0.01, criterion=nn.NLLLoss(), test_criterion=nn.NLLLoss(reduction='sum'), use_cuda=None):

        # Basics
        super(NeuralNet, self).__init__()

        # Settings
        self.optim_type = optimizer
        self.optimizer  = None
        self.lr      = lr
        self.criterion  = criterion
        self.test_criterion  = test_criterion

        # Use CUDA?
        self.use_cuda = use_cuda if (use_cuda != None and cuda.is_available()) else cuda.is_available()

        # Current loss and loss history
        self.train_loss      = 0
        self.valid_loss      = 0
        self.train_loss_hist = []
        self.valid_loss_hist = []

        # Initializing all layers
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        # Running startup routines
        self.startup_routines()

    def startup_routines(self):
        self.optimizer = self.optim_type(self.parameters(), lr=self.lr)
        if self.use_cuda:
            self.cuda()

    def predict(self, input):

        # Switching off autograd
        with torch.no_grad():

            # Use CUDA?
            if self.use_cuda:
                input = input.cuda()

            # Running inference
            return self.forward(input)

    # DONE
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def fit_step(self, training_loader):

        # Preparations for fit step
        self.train_loss = 0 # Resetting training loss
        self.train()        # Switching to autograd
        
        for batch_idx, (data, target) in enumerate(training_loader):
            
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            # Clearing gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.forward(data)

            # Calculating loss
            loss = self.criterion(output, target)
            self.train_loss += loss.item() # Adding to epoch loss

            # Backward pass and optimization
            loss.backward()                      # Backward pass
            self.optimizer.step()                # Optimizing weights

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(training_loader.dataset),
                    100. * batch_idx / len(training_loader), loss))

        # Adding loss to history
        self.train_loss_hist.append(self.train_loss / len(training_loader))

    def validation_step(self, validation_loader):
        self.eval()
        # Preparations for validation step
        self.valid_loss = 0 # Resetting validation loss
        correct = 0
        # Switching off autograd
        with torch.no_grad():

            # Looping through data
            for input, target in validation_loader:

                # Use CUDA?
                if self.use_cuda:
                    input = input.cuda()
                    target = target.cuda()

                # Forward pass
                output = self.forward(input)

                # Calculating loss
                loss = self.test_criterion(output, target)
                self.valid_loss += loss.item() # Adding to epoch loss

                # accuracy
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            # Adding loss to history
            self.valid_loss_hist.append(self.valid_loss / len(validation_loader))

    def fit(self, training_loader, validation_loader=None, epochs=2, show_progress=True, save_best=False):

        # Helpers
        best_validation = 1e5

        # Looping through epochs
        for epoch in range(epochs):
            self.fit_step(training_loader) # Optimizing
            if validation_loader != None:  # Perform validation?
                self.validation_step(validation_loader) # Calculating validation loss

        # Switching to eval
        self.eval()

# 
def main():
    batch_size = 64
    test_batch_size = 1000
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = NeuralNet()
    model.fit(train_loader, test_loader)
    

# 
if __name__ == '__main__':
    main()