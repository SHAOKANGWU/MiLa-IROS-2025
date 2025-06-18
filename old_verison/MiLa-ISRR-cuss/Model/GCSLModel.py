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
from torch.nn.parameter import Parameter

# TODO: We can use residual block here.

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class GCSL_Model(nn.Module):
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
            nn.Linear(1008*2, 100),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(100+3+1, 100),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU())

        self.fc4 = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(50, output_dim))



    def conv_forward(self, current_img, goal_img):
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
        return out1, out2


    def forward(self, current_img, goal_img, current_state):
        # compute [current image, goal image] embeddings
        out1, out2 = self.conv_forward(current_img, goal_img)
        out = torch.stack([out1, out2], dim=1).reshape(out2.size(0), -1)
        out = self.fc1(out)

        # stack images embeddings and current state
        out = torch.hstack([out, current_state])

        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out


class MAML_Model(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.spacial_softmax = SpatialSoftmax(height=30, width=37, channel=32)
        ########################################################

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64*2, 64),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64+3+1, 64),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU())

        self.fc4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(64, output_dim))


    def conv_forward(self, current_img, goal_img):
        out = self.first_layer(current_img)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out1 = self.conv4(out)
        out1 = self.spacial_softmax(out1)

        out = self.first_layer(goal_img)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out2 = self.conv4(out)
        out2 = self.spacial_softmax(out2)
        return out1, out2


    def forward(self, current_img, goal_img, current_state):
        # compute [current image, goal image] embeddings
        out1, out2 = self.conv_forward(current_img, goal_img)
        out = torch.stack([out1, out2], dim=1).reshape(out2.size(0), -1)
        out = self.fc1(out)

        # stack images embeddings and current state
        out = torch.hstack([out, current_state])
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out



class Controller_NN(nn.Module):
    def __init__(self, output_dim):
        super(Controller_NN, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.modified_pretrained = nn.Sequential(*list(resnet18.children())[:-1])
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(512 * 2, 512)
        self.linear2 = nn.Linear(512+3+1, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear = nn.Linear(32, output_dim)

    def forward(self, current_img, goal_img, robot_state):
        out_1 = self.modified_pretrained(current_img)
        out_2 = self.modified_pretrained(goal_img)
        out_combined = torch.cat((out_1, out_2), 1)
        imgs_embedding = self.relu(self.linear1(out_combined.squeeze(2).squeeze(2)))
        out = torch.hstack([imgs_embedding, robot_state])

        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.relu(self.linear4(out))
        out = self.linear5(out)
        out = self.linear(out)
        return out


if __name__ == '__main__':

    # model = GCSL_Model(3)
    # data1 = torch.randn([40, 3, 140, 170])
    # data2 = torch.randn([40, 3, 140, 170])
    # data3 = torch.randn([40, 3])
    # res = model(data1, data2, data3)


    # Candidate Models 1
    # model = Controller_NN(3)
    # data1 = torch.randn([40, 3, 140, 170])
    # data2 = torch.randn([40, 3, 140, 170])
    # data3 = torch.randn([40, 3+1])
    # res = model(data1, data2, data3)

    # Candidate Models 2
    model = WatchTryModel(3)
    data1 = torch.randn([40, 3, 140, 170])
    data2 = torch.randn([40, 3, 140, 170])
    data3 = torch.randn([40, 3+1])
    res = model(data1, data2, data3)

    # model = MAML(3)
    # print(1)