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



class MAML_Model(nn.Module):
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
        ########################################################

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64*2 + 3 + 1, 128),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU())

        self.fc4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(128, output_dim))


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


    def forward(self, imgs, state):
        out = self.conv_forward(imgs)
        out = torch.cat((state, out), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out


class MAML_BT_Model(nn.Module):
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
            nn.Linear(64*2 + 3 + 1 + bias_dim, 100),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(100, 100),
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


    def forward(self, imgs, state):
        out = self.conv_forward(imgs)
        batch_bias = self.bias_transform.repeat(len(imgs), 1)
        out = torch.cat((out, state, batch_bias), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


class MAML_BT_Model_NoGripper(nn.Module):
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
            nn.Linear(64*2 + 3 + bias_dim, 100),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(100, 100),
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


    def forward(self, imgs, state):
        out = self.conv_forward(imgs)
        batch_bias = self.bias_transform.repeat(len(imgs), 1)
        out = torch.cat((out, state, batch_bias), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


class MAML_Model_2Head(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # vgg19 = models.vgg19(pretrained=True).features[:1]
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.spacial_softmax = SpatialSoftmax(39, 39)
        ########################################################

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64*2*2 + 3, 128*2),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128*2, 128*2),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128*2, 128),
            nn.ReLU())

        # only for meta-training
        self.fc4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(128, output_dim))

        # only for meta-test
        self.fc6 = nn.Sequential(
            nn.Linear(128, output_dim))


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


    def forward(self, imgs, state, meta_testing):
        out = self.conv_forward(imgs)
        img_out = out[0:-1]
        goal_out = out[-1].repeat(len(img_out), 1)
        out = torch.cat((img_out, goal_out), dim=1)

        # state shape (batch, 3 (current state))
        out = torch.cat((out, state), dim=1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        if not meta_testing:
            out = self.fc4(out)
            out = self.fc5(out)
        else:
            out = self.fc6(out)
        return out


class G_MAML_Model_2Head(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # vgg19 = models.vgg19(pretrained=True).features[:1]
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.spacial_softmax = SpatialSoftmax(39, 39)
        ########################################################

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64*2*2, 128*2),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128*2, 128*2),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128*2, 128),
            nn.ReLU())

        # only for meta-training
        self.fc4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU())

        self.fc5 = nn.Sequential(
            nn.Linear(128, output_dim))

        # only for meta-test
        self.fc6 = nn.Sequential(
            nn.Linear(128, output_dim))


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


    def forward(self, imgs, meta_testing):
        out = self.conv_forward(imgs)
        img_out = out[0:-1]
        goal_out = out[-1].repeat(len(img_out), 1)
        out = torch.cat((img_out, goal_out), dim=1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        if not meta_testing:
            out = self.fc4(out)
            out = self.fc5(out)
        else:
            out = self.fc6(out)
        return out



class MAML_BT_Model_NoGripper_MoreWider(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # vgg19 = models.vgg19(pretrained=True).features[:1]
        self.first_layer = copy.deepcopy(vgg19.features[:1])
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.spacial_softmax = SpatialSoftmax(39, 39)

        bias_dim = 10
        self.bias_transform = nn.Parameter(torch.zeros([1, bias_dim]), requires_grad=True)
        ########################################################

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(64*2 + 3 + bias_dim, 100),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(100, 100),
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


    def forward(self, imgs, state):
        out = self.conv_forward(imgs)
        batch_bias = self.bias_transform.repeat(len(imgs), 1)
        out = torch.cat((out, state, batch_bias), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out



class DAML_GoalCondiCNN(nn.Module):
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


class DAML_GoalCondiCNN_Cus(nn.Module):
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
            nn.Linear(1296*2+3, 1000),
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


    def forward(self, current_img, goal_img, current_state, meta_testing):
        out1, out2 = self.conv_forward(current_img, goal_img)

        if not meta_testing:
            out = torch.stack([out1, out2], dim=1).reshape(out2.size(0), -1)
        else:
            out3 = torch.zeros_like(out2, device=out2.device, dtype=out2.dtype)
            out = torch.stack([out1, out3], dim=1).reshape(out2.size(0), -1)

        out = torch.hstack([out, current_state])

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


class VisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        state_len = 3
        bias_len = 10

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2) # RGB -> h with kernel of 3
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(24, 12, kernel_size=3, stride=2)

        fc_inshape = 4800 + state_len + bias_len
        #breakpoint()
        self.fc1 = nn.Linear(fc_inshape, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 3)
        self.fc5 = nn.Linear(200, 3)

        self.bias = nn.Parameter(torch.empty(10).uniform_(-1, 1)) # arbitrarily decided on the initialization


    def forward(self, img, state_vec, meta_testing):
        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))
        flat_img = img.reshape(img.shape[0], -1) # flatten except batch dim
        bias = self.bias.unsqueeze(0).expand(state_vec.shape[0], self.bias.shape[0]) # expand to (B,10)
        img_and_state = torch.cat([flat_img, state_vec, bias], dim=1)
        img_and_state = F.relu(self.fc1(img_and_state))
        img_and_state = F.relu(self.fc2(img_and_state))

        if not meta_testing:
            img_and_state = F.relu(self.fc3(img_and_state))
            img_and_state = self.fc4(img_and_state)

        else:
            img_and_state = self.fc5(img_and_state)
        return img_and_state



class VisualModel_Norm_Spatial(nn.Module):
    def __init__(self):
        super().__init__()
        state_len = 3
        bias_len = 3

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.spacial_softmax = SpatialSoftmax(9, 9)

        fc_inshape = 16*2 + state_len + bias_len
        self.fc1 = nn.Linear(fc_inshape, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 3)
        self.fc5 = nn.Linear(200, 3)

        self.bias = nn.Parameter(torch.empty(bias_len).uniform_(-1, 1)) # arbitrarily decided on the initialization


    def forward(self, img, state_vec, meta_testing):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        # flat_img = img.view(img.shape[0], -1) # flatten except batch dim
        flat_img = self.spacial_softmax(img)

        bias = self.bias.unsqueeze(0).expand(state_vec.shape[0], self.bias.shape[0]) # expand to (B,10)
        img_and_state = torch.cat([flat_img, state_vec, bias], dim=1)
        img_and_state = F.relu(self.fc1(img_and_state))
        img_and_state = F.relu(self.fc2(img_and_state))

        if not meta_testing:
            img_and_state = F.relu(self.fc3(img_and_state))
            img_and_state = self.fc4(img_and_state)

        else:
            img_and_state = self.fc5(img_and_state)
        return img_and_state



class VisualModel_Norm_Softmax2d(nn.Module):
    def __init__(self):
        super().__init__()
        state_len = 3

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.spacial_softmax = nn.Softmax2d()

        fc_inshape = 1296 + state_len
        self.fc1 = nn.Linear(fc_inshape, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 3)
        self.fc5 = nn.Linear(200, 3)


    def forward(self, img, state_vec, meta_testing):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        # flat_img = img.view(img.shape[0], -1) # flatten except batch dim
        flat_img = self.spacial_softmax(img).reshape(img.shape[0], -1)

        img_and_state = torch.cat([flat_img, state_vec], dim=1)
        img_and_state = F.relu(self.fc1(img_and_state))
        img_and_state = F.relu(self.fc2(img_and_state))

        if not meta_testing:
            img_and_state = F.relu(self.fc3(img_and_state))
            img_and_state = self.fc4(img_and_state)

        else:
            img_and_state = self.fc5(img_and_state)
        return img_and_state


class VisualModel_Softmax2d_Deep(nn.Module):
    def __init__(self):
        super().__init__()
        state_len = 3

        #################### vision network ####################
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.first_layer = copy.deepcopy(vgg19.features[:1])

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.spacial_softmax = nn.Softmax2d()

        fc_inshape = 1296 + state_len
        self.fc1 = nn.Linear(fc_inshape, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 200)

        # Head 1: meta-testing is False
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 3)

        # Head 2: meta-testing is True
        self.fc6 = nn.Linear(200, 3)


    def forward(self, img, state_vec, meta_testing):
        img = self.first_layer(img)
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        flat_img = self.spacial_softmax(img).reshape(img.shape[0], -1)

        img_and_state = torch.cat([flat_img, state_vec], dim=1)
        img_and_state = F.relu(self.fc1(img_and_state))
        img_and_state = F.relu(self.fc2(img_and_state))
        img_and_state = F.relu(self.fc3(img_and_state))

        if not meta_testing:
            img_and_state = F.relu(self.fc4(img_and_state))
            img_and_state = self.fc5(img_and_state)

        else:
            img_and_state = self.fc6(img_and_state)
        return img_and_state



if __name__ == '__main__':
    data1 = torch.randn([100, 3, 168, 170])
    data2 = torch.randn([100, 3])

    model = VisualModel_Norm_Softmax2d()
    res = model(data1, data2, meta_testing=True)

    # model = MAML_Model_2Head(3)
    # res = model(data1, data2, meta_testing=True)

    print(res)