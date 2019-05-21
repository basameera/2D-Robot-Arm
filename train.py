
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def pmin(data):
    print(data.dtype, end=' ')
    print(type(data), end=' ')
    if isinstance(data, np.ndarray):
        print(data.shape)
    print()

# https://stackoverflow.com/questions/13897316/approximating-the-sine-function-with-a-neural-network
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 1, 300, 1
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
class TheModelClass(nn.Module):
    def __init__(self, D_in=1, H=10, D_out=1):
        super(TheModelClass, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass(D_in, H, D_out).to(device=device)
model.eval()

# Init loass
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []

# Print model's state_dict
print("=== Model ===")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("=== Optimizer ===")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    

for t in tqdm(range(500)):
    # gen random data
    Xin = np.random.randint(360*3, size=1000).reshape(1000,1)
    Yout = np.sin(np.deg2rad(Xin))
    
    x = torch.tensor(Xin, dtype=dtype, device=device)

    y = torch.tensor(Yout, dtype=dtype, device=device)
#     print(np.min(y.cpu().numpy()))
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
#     if t%1000==0:
#         print(t, loss.item())
    loss_history.append(loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

# plot train loss
f1 = plt.figure()
plt.plot(loss_history, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Train loss')
plt.legend()
f1.savefig('train_loss.png')


# save model
model_name = 'sinNN_new'
torch.save(model.state_dict(), model_name)

# load model
model_sin = TheModelClass(D_in, H, D_out).to(device=device)
model_sin.load_state_dict(torch.load(model_name))
model_sin.eval()

# prediction with loaded model
X_test = np.random.randint(360, size=1000).reshape(1000,1)
Y_test = np.sin(np.deg2rad(X_test))
y_pred = model_sin(torch.tensor(X_test, dtype=dtype, device=device))

# plot prediction
f2 = plt.figure()
plt.scatter(X_test, Y_test, label='Target')
plt.scatter(X_test, y_pred.cpu().data, label='Prediction')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.legend()
f2.savefig('prediction.png')