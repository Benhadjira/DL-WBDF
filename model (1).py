import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, patch_size, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.patch_size =  patch_size
        self.dropout_rate = dropout_rate

        self.depthwise = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, groups=1, bias=False)
        self.pointwise = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, bias=True)

        self.conv1x1 = nn.Conv2d(32, 8, kernel_size=1, bias=True)

        self.flatten_size = (patch_size[0] // 2) * (patch_size[1] // 2) * 8  # Adjusted based on max pooling
        self.fc1 = nn.Linear(self.flatten_size, 10)  # Small FC layer
        self.fc2 = nn.Linear(10, 2)  # Output layer

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.sigmoid(x)
        x = F.dropout2d(x, self.dropout_rate, training=self.training)
        x = F.max_pool2d(x, 2)

        x = self.conv1x1(x)
        x = F.elu(x)
        x = F.dropout2d(x, self.dropout_rate, training=self.training)

        x = x.view(x.size(0), -1) 
        x = F.elu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)

        return x