import torch
import torch.nn as nn
#from torchsummary import summary

class Model(nn.Module):
    def __init__(self, input_channels=3):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(10, 10, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(10, 10, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(10, 10, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10 * 75 * 75, 128)
        self.fc2 = nn.Linear(128, 23)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = Model(input_channels=3)

#summary(model, (1, 300, 300))

