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

        # pos_x, pos_y = np.meshgrid(
        #     np.linspace(-1., 1., self.height),
        #     np.linspace(-1., 1., self.width)
        # )

        ############################ torch API ############################
        pos_x_tensor, pos_y_tensor = torch.meshgrid(
            torch.linspace(-1., 1., self.height),
            torch.linspace(-1., 1., self.width), indexing='ij'
        )
        pos_x_tensor, pos_y_tensor = pos_x_tensor.T, pos_y_tensor.T
        ############################ torch API ############################

        # pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        # pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()

        ############################ torch API ############################
        pos_x = pos_x_tensor.reshape(self.height * self.width)
        pos_y = pos_y_tensor.reshape(self.height * self.width)
        ############################ torch API ############################

        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):

        feature = feature.view(-1, self.height * self.width)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class MAML_Model_Baseline(nn.Module):
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
            nn.Linear(64+3+1, 1000),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(1000, 500),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU())

        self.fc4 = nn.Sequential(
            nn.Linear(500, 100),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(100, output_dim))


    def conv_forward(self, current_imgs):
        out = self.first_layer(current_imgs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.spacial_softmax(out)
        return out


    def forward(self, current_imgs, current_state):
        out = self.conv_forward(current_imgs)
        out = torch.hstack([out, current_state])
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out



class GCMAML_Model(nn.Module):
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
            nn.Linear(64*2 + 3 + 1, 1000),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU())


        # only for meta-training
        self.fc4 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(500, 100),
            nn.ReLU())

        self.fc6 = nn.Sequential(
            nn.Linear(100, output_dim))


        # only for meta-test
        self.fc7 = nn.Sequential(
            nn.Linear(500, 100),
            nn.ReLU())

        self.fc8 = nn.Sequential(
            nn.Linear(100, output_dim))

        # self.initialization()

    def initialization(self):
        nn.init.kaiming_uniform_(self.conv1[0].weight, nonlinearity='relu')
        nn.init.constant_(self.conv1[0].bias, 0)

        nn.init.kaiming_uniform_(self.conv2[0].weight, nonlinearity='relu')
        nn.init.constant_(self.conv2[0].bias, 0)

        nn.init.kaiming_uniform_(self.conv3[0].weight, nonlinearity='relu')
        nn.init.constant_(self.conv3[0].bias, 0)

        nn.init.kaiming_uniform_(self.conv4[0].weight, nonlinearity='relu')
        nn.init.constant_(self.conv4[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc1[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc1[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc2[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc2[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc3[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc3[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc4[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc4[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc5[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc5[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc6[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc6[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc7[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc7[0].bias, 0)

        nn.init.kaiming_uniform_(self.fc8[0].weight, nonlinearity='relu')
        nn.init.constant_(self.fc8[0].bias, 0)


    def conv_forward(self, imgs):
        out = self.first_layer(imgs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out1 = self.spacial_softmax(out)
        return out1


    def forward(self, imgs, state, meta_testing):
        out = self.conv_forward(imgs)
        batch_imgs = out[0:-1]
        goal_img = out[-1:].repeat(batch_imgs.size(0), 1)
        goal_embedding = goal_img

        # if not meta_testing:
        #     goal_embedding = goal_img
        # else:
        #     goal_embedding = torch.zeros_like(goal_img)

        context = torch.cat((batch_imgs, goal_embedding), dim=1)
        assert len(state) == len(context)
        out = torch.cat((state, context), dim=1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        if not meta_testing:
            out = self.fc4(out)
            out = self.fc5(out)
            out = self.fc6(out)
        else:
            out = self.fc7(out)
            out = self.fc8(out)
        return out


if __name__ == '__main__':
    data1 = torch.randn([40+1, 3, 140, 170])
    data2 = torch.randn([40, 3+1])

    # model = MAML_Model_Baseline(3)
    # res = model(data1, data2)

    model = GCMAML_Model(3)
    res = model(data1, data2, meta_testing=True)

    print(res)