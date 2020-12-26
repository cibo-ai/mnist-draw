import torch
from torch import nn
from torch.nn import functional as F

class Conv2d_model(nn.Module):
  def __init__(self):
    super(Conv2d_model, self).__init__()
    # Convolutional layers
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(12, affine=True)
    self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=6, stride=2, padding=2, bias=False)
    self.bn2 = nn.BatchNorm2d(24, affine=True)
    self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
    self.bn3 = nn.BatchNorm2d(32, affine=True)
    
    # Dense layers
    self.fc1 = nn.Linear(in_features=32*7*7, out_features=200, bias=False)
    self.bn4 = nn.BatchNorm1d(200, affine=True)
    self.dr = nn.Dropout(p=0.3)
    self.fc2 = nn.Linear(in_features=200, out_features=10, bias=False)
  
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    
    # Flatten the input to feed into the dense layers
    x = x.flatten(start_dim=1)
    
    x = F.relu(self.bn4(self.fc1(x)))
    x = self.dr(x)
    x = F.log_softmax(self.fc2(x), dim=1)
    return x
