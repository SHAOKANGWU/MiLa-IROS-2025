import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import copy
import torchvision.models as models
torch.set_default_tensor_type(torch.DoubleTensor)


class MyCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=2),
            nn.BatchNorm2d(40),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(480, 200),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(200, output_dim))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out



class MyCNN_2(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(256, 200),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(200, output_dim))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class MyCNN_3(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.first_layer = copy.deepcopy(vgg19.features[:1])

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(4608, 200),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(200, output_dim))

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.conv2(out)
        # out = self.conv3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class GoalCondiCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.spacial_softmax = nn.Softmax2d()
        ########################################################


        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(1008*2, 1000),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(1000, 200),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(200, output_dim))

    def forward(self, current_img, goal_img):

        out = self.first_layer(current_img)
        out = self.conv1(out)
        out = self.conv2(out)
        out1 = self.conv3(out)
        out1 = self.spacial_softmax(out1)
        out1 = out1.reshape(out1.size(0), -1)

        out = self.first_layer(goal_img)
        out = self.conv1(out)
        out = self.conv2(out)
        out2 = self.conv3(out)
        out2 = self.spacial_softmax(out2)
        out2 = out2.reshape(out2.size(0), -1)

        out = torch.stack([out1, out2], dim=1).reshape(out2.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


if __name__ == '__main__':
    image = torch.randn(size=[1, 3, 280, 320])
    image_2 = image[:, :, ::2, ::2]
    image_3 = image[:, :, ::3, ::3]
    image_4 = image[:, :, ::4, ::4]
    model = GoalCondiCNN(3)
    data1 = torch.randn([10, 3, 140, 160])
    data2 = torch.randn([10, 3, 140, 160])
    res = model(data1, data2)
    print(1)