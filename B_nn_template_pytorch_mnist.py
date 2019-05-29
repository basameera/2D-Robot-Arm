"""
Sameera Sandaruwan
"""

# Importing PyTorch tools
import torch
from torch import nn, optim, cuda

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

    def __init__(self, optimizer=optim.SGD, alpha=0.01, criterion=nn.CrossEntropyLoss(), use_cuda=None):

        # Basics
        super(NeuralNet, self).__init__()

        # Settings
        self.optim_type = optimizer
        self.optimizer  = None
        self.alpha      = alpha
        self.criterion  = criterion

        # Use CUDA?
        self.use_cuda = use_cuda if (use_cuda != None and cuda.is_available()) else cuda.is_available()

        # Current loss and loss history
        self.train_loss      = 0
        self.valid_loss      = 0
        self.train_loss_hist = []
        self.valid_loss_hist = []

        # Initializing states with placeholder tensors
        self.softmax_state = torch.tensor([0], dtype=torch.float64)

        # Initializing all layers
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        # Running startup routines
        self.startup_routines()

    def startup_routines(self):
        self.optimizer = self.optim_type(self.parameters(), lr=self.alpha)
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

        # Looping through data
        for input, softmax_target in training_loader:

            # Use CUDA?
            if self.use_cuda:
                input = input.cuda()
                softmax_target = softmax_target.cuda()

            # Clearing gradients
            self.optimizer.zero_grad()           

            # Forward pass
            self.forward(input)

            # Calculating loss
            loss = self.criterion(self.softmax_state, softmax_target)
            self.train_loss += loss.item() # Adding to epoch loss

            # Backward pass and optimization
            loss.backward()                      # Backward pass
            self.optimizer.step()                # Optimizing weights
            

        # Adding loss to history
        self.train_loss_hist.append(self.train_loss / len(training_loader))

    def validation_step(self, validation_loader):

        # Preparations for validation step
        self.valid_loss = 0 # Resetting validation loss

        # Switching off autograd
        with torch.no_grad():

            # Looping through data
            for input, softmax_target in validation_loader:

                # Use CUDA?
                if self.use_cuda:
                    input = input.cuda()
                    softmax_target = softmax_target.cuda()

                # Forward pass
                self.forward(input)

                # Calculating loss
                loss = self.criterion(self.softmax_state, softmax_target)
                self.valid_loss += loss.item() # Adding to epoch loss

            # Adding loss to history
            self.valid_loss_hist.append(self.valid_loss / len(validation_loader))

    def fit(self, training_loader, validation_loader=None, epochs=10, show_progress=True, save_best=False):

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
    model = NeuralNet()
    print(model.eval())

# 
if __name__ == '__main__':
    main()