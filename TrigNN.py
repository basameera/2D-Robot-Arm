import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define NN
class TrigNN(nn.Module):
    def __init__(self, D_in=1, H=300, D_out=1):
        super(TrigNN, self).__init__()
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

# Define Trig Model
class TrigModel:
    def __init__(self):
        self.name = 'TrigonometryNN'
        self.dtype = torch.float
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def Sin(self, model='sinNN'):
        self.model_sin = TrigNN().to(device=self.device)
        self.model_sin.load_state_dict(torch.load(model))
        return self.model_sin
    
    def SinEval(self):
        return self.Sin().eval()
    
    def Cos(self, model='cosNN'):
        self.model_cos = TrigNN().to(device=self.device)
        self.model_cos.load_state_dict(torch.load(model))
        return self.model_cos
    
    def CosEval(self):
        return self.Cos().eval()
    
    def toTorchTensor(self, x):
        return torch.tensor(x, dtype=self.dtype, device=self.device)
        
    def toNumpy(self, x):
        return x.cpu().data