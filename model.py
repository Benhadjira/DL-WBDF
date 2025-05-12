import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return x

class CNN(nn.Module):
    def __init__(self, patch_size, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.patch_size = patch_size
        self.dropout_rate = dropout_rate

        self.sepconv1 = SeparableConvBlock(1, 16)
        self.sepconv2 = SeparableConvBlock(16, 32)
        self.sepconv3 = SeparableConvBlock(32, 64)

        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(dropout_rate)

        flatten_size = (patch_size[0] // 2) * (patch_size[1] // 2) * 64
        self.fc1 = nn.Linear(flatten_size, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.sepconv1(x)
        x = self.dropout2d(x)
        x = self.sepconv2(x)
        x = self.dropout2d(x)
        x = self.sepconv3(x)
        x = self.pool(x)
        x = self.dropout2d(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x
