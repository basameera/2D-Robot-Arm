'''
Sameera Sandaruwan
'''

# Importing PyTorch tools
import torch
from torch import nn, optim, cuda

# Importing other libraries
import numpy as np
import matplotlib.pyplot as plt
import time

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
        self.lin_1 = nn.Linear(
            in_features  = 1,
            out_features = 50,
            bias         = True,
        )
        self.relu = nn.ReLU(
            inplace = False,
        )
        self.lin_2 = nn.Linear(
            in_features  = 1,
            out_features = 1,
            bias         = True,
        )
        self.softmax = nn.Softmax()

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

    def forward(self, input):
        input = self.lin_1(input)                # Linear
        input = self.relu(input)                 # ReLU
        input = self.lin_2(input)                # Linear
        self.softmax_state = self.softmax(input) # Softmax
        return self.softmax_state

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

            # Forward pass
            self.forward(input)

            # Calculating loss
            loss = self.criterion(self.softmax_state, softmax_target)
            self.train_loss += loss.item() # Adding to epoch loss

            # Backward pass and optimization
            loss.backward()                      # Backward pass
            self.optimizer.step()                # Optimizing weights
            self.optimizer.zero_grad()           # Clearing gradients

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

        # Possibly prepping progress message
        if show_progress:
            epoch_l = max(len(str(epochs)), 2)
            print('Training model...')
            print('%sEpoch   Training loss   Validation loss   Duration   Time remaining' % ''.rjust(2 * epoch_l - 4, ' '))
            t = time.time()

        # Looping through epochs
        for epoch in range(epochs):
            self.fit_step(training_loader) # Optimizing
            if validation_loader != None:  # Perform validation?
                self.validation_step(validation_loader) # Calculating validation loss

            # Possibly printing progress
            if show_progress:
                eta_s = (time.time() - t) * (epochs - epoch)
                eta = '%sm %ss' % (round(eta_s / 60), 60 - round(eta_s % 60))
                print('%s/%s' % (str(epoch + 1).rjust(epoch_l, ' '), str(epochs).ljust(epoch_l, ' ')),
                    '| %s' % str(round(self.train_loss_hist[-1], 8)).ljust(13, ' '),
                    '| %s' % str(round(self.valid_loss_hist[-1], 8)).ljust(15, ' '),
                    '| %ss' % str(round(time.time() - t, 3)).rjust(7, ' '),
                    '| %s' % eta.ljust(14, ' '))
                t = time.time()

            # Possibly saving model
            if save_best:
                if self.valid_loss_hist[-1] < best_validation:
                    self.save('best_validation')
                    best_validation = self.valid_loss_hist[-1]

        # Switching to eval
        self.eval()

    def plot_hist(self):

        # Adding plots
        plt.plot(self.train_loss_hist, color='blue', label='Training loss')
        plt.plot(self.valid_loss_hist, color='springgreen', label='Validation loss')

        # Axis labels
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Displaying plot
        plt.show()

    def save(self, name='model.pth'):
        if not '.pth' in name: name += '.pth'
        torch.save({
            'lin_1': self.lin_1,
            'relu': self.relu,
            'lin_2': self.lin_2,
            'softmax': self.softmax,
            'train_loss':      self.train_loss,
            'valid_loss':      self.valid_loss,
            'train_loss_hist': self.train_loss_hist,
            'valid_loss_hist': self.valid_loss_hist,
            'optim_type':      self.optim_type,
            'alpha':           self.alpha,
            'criterion':       self.criterion,
            'use_cuda':        self.use_cuda
        }, name)

    @staticmethod
    def load(name='model.pth'):
        if not '.pth' in name: name += '.pth'
        checkpoint = torch.load(name)
        model = NeuralNet(
            optimizer = checkpoint['optim_type'],
            alpha     = checkpoint['alpha'],
            criterion = checkpoint['criterion'],
            use_cuda  = checkpoint['use_cuda']
        )
        model.lin_1 = checkpoint['lin_1']
        model.relu = checkpoint['relu']
        model.lin_2 = checkpoint['lin_2']
        model.softmax = checkpoint['softmax']
        model.train_loss      = checkpoint['train_loss']
        model.valid_loss      = checkpoint['valid_loss']
        model.train_loss_hist = checkpoint['train_loss_hist']
        model.valid_loss_hist = checkpoint['valid_loss_hist']
        model.startup_routines()
        return model