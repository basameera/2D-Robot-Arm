"""
Sameera Sandaruwan

To Do;
* add more cmd args

=== Features ===
* Fit
* Fit step
* Validation 
* Save, Save best model
* Predict
* Show progress
* Plotting
* CMD Args

"""
from __future__ import print_function
import argparse

# Importing PyTorch tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import cuda

# Importing other libraries
import numpy as np
import matplotlib.pyplot as plt

# Model
class NeuralNet(nn.Module):

    def __init__(self, optimizer=optim.SGD, lr=0.01, criterion=nn.NLLLoss(), valid_criterion=nn.NLLLoss(reduction='sum'), use_cuda=None):

        # Basics
        super(NeuralNet, self).__init__()
        self.model_name = __class__.__name__

        # Settings
        self.optim_type = optimizer
        self.optimizer  = None
        self.lr      = lr
        self.criterion  = criterion
        self.valid_criterion  = valid_criterion

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
            return self(input)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def fit_step(self, training_loader, epoch, show_progress=False):

        # Preparations for fit step
        self.train_loss = 0 # Resetting training loss
        self.train()        # Switching to autograd
        
        for batch_idx, (data, target) in enumerate(training_loader):
            
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            # Clearing gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self(data)

            # Calculating loss
            loss = self.criterion(output, target)
            self.train_loss += loss.item() # Adding to epoch loss

            # Backward pass and optimization
            loss.backward()                      # Backward pass
            self.optimizer.step()                # Optimizing weights

            if show_progress:
                if batch_idx % int(len(training_loader)*0.05) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(training_loader.dataset),
                        100. * batch_idx / len(training_loader), loss))

        # Adding loss to history
        self.train_loss_hist.append(self.train_loss / len(training_loader))

    def validation_step(self, validation_loader, show_progress=False):
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
                loss = self.valid_criterion(output, target)
                self.valid_loss += loss.item() # Adding to epoch loss

                # accuracy
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            self.valid_loss /= len(validation_loader.dataset)

            # Adding loss to history
            self.valid_loss_hist.append(self.valid_loss / len(validation_loader))
        
        if show_progress:
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.valid_loss, correct, len(validation_loader.dataset),
                100. * correct / len(validation_loader.dataset)))

    def fit(self, training_loader, validation_loader=None, epochs=2, show_progress=True, save_best=False, save_plot=False):

        # Helpers
        best_validation = 1e5

        # Looping through epochs
        for epoch in range(epochs):
            self.fit_step(training_loader, epoch, show_progress) # Optimizing
            if validation_loader != None:  # Perform validation?
                self.validation_step(validation_loader, show_progress) # Calculating validation loss

            # Possibly saving model
            if save_best:
                if self.valid_loss_hist[-1] < best_validation:
                    self.save('best_validation_'+str(epoch))
                    best_validation = self.valid_loss_hist[-1]
        
        # Switching to eval
        self.eval()

        # save plot
        if save_plot:
            self.plot_hist()

    def save(self, name='model.pth'):
        if not '.pth' in name: name += '.pth'
        torch.save(self.state_dict(), name)
    
    def plot_hist(self, plot_name='plot_name'):

        if not '.png' in plot_name: plot_name += '.png'
        
        plt.figure()

        # Adding plots
        plt.plot(self.train_loss_hist, color='blue', label='Training loss')
        plt.plot(self.valid_loss_hist, color='red', label='Validation loss')

        # Axis labels
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # Displaying plot
        plt.savefig(plot_name)

    @staticmethod
    def load(name='model.pth'):
        if not '.pth' in name: name += '.pth'

        # the_model = TheModelClass(*args, **kwargs)
        # the_model.load_state_dict(torch.load(PATH))
            
        model = NeuralNet()
        model.load_state_dict(torch.load(name))
        model.startup_routines()
        return model

# cmd args
def cmdArgs():
    parser = argparse.ArgumentParser(description='PyTorch NN Template\n- by Bassandaruwan')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the trained Model')
    parser.add_argument('--save-best', action='store_true', default=False,
                        help='For Saving the current Best Model')
    return parser.parse_args()

# Let's run
def main():
    args = cmdArgs()
    print(args)

    batch_size = args.batch_size
    valid_batch_size = args.test_batch_size


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    valid_data = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=valid_batch_size, shuffle=False, **kwargs)

    # Instantiate mode
    model = NeuralNet()

    # Train model
    model.fit(train_loader, valid_loader, epochs=args.epochs, save_best=args.save_best, show_progress=False, save_plot=False)
    
    # save model
    if args.save_model:
        print("saving model")
        model.save()

    # load model
    # model = NeuralNet().load()

    # test model
    print('******************************')

    # for input, target in valid_loader:
    #     output = model.predict(input)
    #     print(torch.argmax(output[0]), target[0])
    #     break

# main 
if __name__ == '__main__':
    main()