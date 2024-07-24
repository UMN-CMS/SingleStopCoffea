from analyzer.cli import runCli

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import setup_logging

class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            self.fc1 = nn.Linear(13,13)
            self.fc2 = nn.Linear(13,3)
        def forward(self,x):
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x),dim=1)
            return x
        
if __name__ == "__main__":
    args = runCli()
    setup_logging(default_level=args.log_level)

    if hasattr(args, "func"):
        args.func(args)
