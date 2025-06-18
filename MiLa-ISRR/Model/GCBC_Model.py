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


class SpatialSoftmax(nn.Module):
    """
    C, W, H => C*2 # for each channel, get good point [x,y]
    """

    def __init__(self, n_rows, n_cols):
        super(SpatialSoftmax, self).__init__()

        x_map = torch.zeros((n_rows, n_cols))
        y_map = torch.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                x_map[i, j] = (i - n_rows / 2) / n_rows
                y_map[i, j] = (j - n_cols / 2) / n_cols
        x_map = x_map.reshape((-1))
        y_map = y_map.reshape((-1))

        self.register_buffer('x_map', x_map)
        self.register_buffer('y_map', y_map)
        # self.x_map = x_map.to("cuda")
        # self.y_map = y_map.to("cuda")

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # batch, C, W*H
        x = F.softmax(x, 2)  # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map)  # batch, C
        fp_y = torch.matmul(x, self.y_map)  # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x  # batch, C*2


class GCBC_Model_NoGripper(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # vgg19 = models.vgg19(pretrained=True).features[:1]
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.spacial_softmax = SpatialSoftmax(32, 39)

        bias_dim = 10
        self.bias_transform = nn.Parameter(torch.zeros([1, bias_dim]), requires_grad=True)
        ########################################################

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64*4 + 3 + bias_dim, 256),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(256, 100),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.ReLU())

        self.fc4 = nn.Sequential(
            nn.Linear(100, output_dim))


    def conv_forward(self, imgs):
        conv_out = self.first_layer(imgs)
        conv_out = F.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.conv1(conv_out)
        conv_out = F.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.conv2(conv_out)
        conv_out = F.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.conv3(conv_out)
        conv_out = F.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.spacial_softmax(conv_out)
        return conv_out


    def forward(self, imgs, goal_imgs, state):
        out_imgs = self.conv_forward(imgs)
        out_goal_imgs = self.conv_forward(goal_imgs)
        out = torch.cat((out_imgs, out_goal_imgs), 1)


        batch_bias = self.bias_transform.repeat(len(imgs), 1)
        out = torch.cat((out, state, batch_bias), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out



if __name__ == '__main__':
    data1 = torch.randn([40, 3, 140, 170])
    data2 = torch.randn([40, 3])

    model = GCBC_Model_NoGripper(3)
    res = model(data1, data1, data2)

    # model = MAML_Model_2Head(3)
    # res = model(data1, data2, meta_testing=True)

    print(res)