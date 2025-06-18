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
from torch.nn.parameter import Parameter
torch.set_default_tensor_type(torch.DoubleTensor)


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


class MetaGoalCondiCNN(nn.Module):
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
            nn.Linear(200, 100),
            nn.ReLU())


        # only for meta-training
        self.fc4 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(50, output_dim))


        # only for meta-test
        self.fc6 = nn.Sequential(
            nn.Linear(100, output_dim))


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


    def forward(self, current_img, goal_img, meta_testing):
        out1, out2 = self.conv_forward(current_img, goal_img)

        if not meta_testing:
            out = torch.stack([out1, out2], dim=1).reshape(out2.size(0), -1)
        else:
            out3 = torch.zeros_like(out2, device=out2.device, dtype=out2.dtype)
            out = torch.stack([out1, out3], dim=1).reshape(out2.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        # in here, add more layers to distinguish meta-train and meta-test
        if not meta_testing:
            out = self.fc4(out)
            out = self.fc5(out)
        else:
            out = self.fc6(out)
        return out


class MetaGoalCondiCNN_without_batchnorm(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2),
            nn.ReLU(),
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
            nn.Linear(200, 100),
            nn.ReLU())


        # only for meta-training
        self.fc4 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(50, output_dim))


        # only for meta-test
        self.fc6 = nn.Sequential(
            nn.Linear(100, output_dim))


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


    def forward(self, current_img, goal_img, meta_testing):
        out1, out2 = self.conv_forward(current_img, goal_img)

        if not meta_testing:
            out = torch.stack([out1, out2], dim=1).reshape(out2.size(0), -1)
        else:
            out3 = torch.zeros_like(out2, device=out2.device, dtype=out2.dtype)
            out = torch.stack([out1, out3], dim=1).reshape(out2.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        # in here, add more layers to distinguish meta-train and meta-test
        if not meta_testing:
            out = self.fc4(out)
            out = self.fc5(out)
        else:
            out = self.fc6(out)
        return out


class MetaGoalCondi_MAML_Model(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.spacial_softmax = SpatialSoftmax(height=30, width=37, channel=32)
        ########################################################

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64*2, 1000),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(1000, 200),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU())


        self.fc4 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(50, output_dim))

        self.fc6 = nn.Sequential(
            nn.Linear(100, output_dim))


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


    def forward(self, current_img, goal_img, meta_testing):
        # compute [current image, goal image] embeddings
        out1, out2 = self.conv_forward(current_img, goal_img)

        if not meta_testing:
            out = torch.stack([out1, out2], dim=1).reshape(out2.size(0), -1)
        else:
            out3 = torch.zeros_like(out2, device=out2.device, dtype=out2.dtype)
            out = torch.stack([out1, out3], dim=1).reshape(out2.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        # in here, add more layers to distinguish meta-train and meta-test
        if not meta_testing:
            out = self.fc4(out)
            out = self.fc5(out)
        else:
            out = self.fc6(out)
        return out


if __name__ == '__main__':
    image = torch.randn(size=[1, 3, 280, 320])
    image_2 = image[:, :, ::2, ::2]
    image_3 = image[:, :, ::3, ::3]
    image_4 = image[:, :, ::4, ::4]
    # model = MetaGoalCondiCNN(3)
    # data1 = torch.randn([1, 3, 140, 170])
    # data2 = torch.randn([1, 3, 140, 170])
    # res = model(data1, data2, meta_testing=True)

    model = MetaGoalCondi_MAML_Model(3)
    data1 = torch.randn([40, 3, 140, 170])
    data2 = torch.randn([40, 3, 140, 170])
    res = model(data1, data2, meta_testing=True)
    # print(1)