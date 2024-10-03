from analyzer.cli import runCli
from analyzer.cli import main

import torch
import torch.nn as nn
import torch.nn.functional as F

# from . import setup_logging

class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            self.fc1 = nn.Linear(14,7)
            self.fc2 = nn.Linear(7,1)
        def forward(self,x):
            x = F.relu(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            return x
        
if __name__ == "__main__":
    main()
