import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import torchvision.models as models
torch.set_default_tensor_type(torch.DoubleTensor)

class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.pre_trained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.pre_trained_model.fc.in_features
        self.pre_trained_model.fc = nn.Linear(in_features, 250)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)

    def forward(self, img):
        x = self.pre_trained_model(img)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LH_DMP(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.pre_trained_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = self.pre_trained_model.fc.in_features
        self.pre_trained_model.fc = nn.Linear(in_features, 250)

        # used for goal prediction
        self.goal_fc1 = nn.Linear(250, 100)
        self.goal_fc2 = nn.Linear(100, 50)
        self.goal_fc3 = nn.Linear(50, 3)    # x, y, z (3 dimension)

        # used for weights prediction
        self.logits_fc1 = nn.Linear(250, 101)
        self.logits_fc2 = nn.Linear(101, 50)
        self.logits_fc3 = nn.Linear(50, output_dim)  # 3 skills

        # used for tau prediction
        self.tau_fc1 = nn.Linear(250, 102)
        self.tau_fc2 = nn.Linear(102, 50)
        self.tau_fc3 = nn.Linear(50, 1)  # mean (1 dimension) + std (1 dimension)

    def forward(self, img):
        x = self.pre_trained_model(img)

        # predict goal
        g = F.relu(self.goal_fc1(x))
        g = F.relu(self.goal_fc2(g))
        goal = self.goal_fc3(g)

        # predict logits
        p = F.relu(self.logits_fc1(x))
        p = F.relu(self.logits_fc2(p))
        logits = self.logits_fc3(p)

        # predict tau
        t = F.relu(self.tau_fc1(x))
        t = F.relu(self.tau_fc2(t))
        # tau_mean, and tau_log_std can be in any range (R).
        tau = self.tau_fc3(t)
        # tau_mean = tau[:, 0:1]
        # tau_log_std = tau[:, 1:2]
        return goal, logits, tau    #tau_mean, tau_log_std

class GripperModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.pre_trained_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = self.pre_trained_model.fc.in_features
        self.pre_trained_model.fc = nn.Linear(in_features, 250)

        # used for weights prediction
        self.gripper_fc1 = nn.Linear(250, 100)
        self.gripper_fc2 = nn.Linear(100+state_dim, 50)
        self.gripper_fc3 = nn.Linear(50, 10)
        self.gripper_fc4 = nn.Linear(10, 2)  # close or open the gripper


    def forward(self, img, states):
    # def forward(self, img):
        x = self.pre_trained_model(img)
        x = F.relu(self.gripper_fc1(x))
        x = torch.concat([x, states], dim=1)
        x = F.relu(self.gripper_fc2(x))
        x = F.relu(self.gripper_fc3(x))
        logits = self.gripper_fc4(x)

        return logits


class MIL(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.pre_trained_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = self.pre_trained_model.fc.in_features
        self.pre_trained_model.fc = nn.Linear(in_features, 250)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100 + state_dim, 50)

        self.pre_update_fc = nn.Linear(50, action_dim)
        self.post_update_fc = nn.Linear(50, action_dim)

    def forward(self, img, state, post_update):
        x = self.pre_trained_model(img)
        x = F.relu(self.fc1(x))
        x = torch.concat([x, state], dim=1)
        x = F.relu(self.fc2(x))

        if post_update is False:
            x = self.pre_update_fc(x)
        else:
            x = self.post_update_fc(x)
        return x


def visualize_img():
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # image: 640 (width) * 480 (height)
    image = Image.open("../data/dataset/meta-train/demo1/0-Color.png")
    # 3, height, width
    img_tensor = transform(image)
    img_ = np.transpose(img_tensor.numpy(), [1, 2, 0])

    # print("width {}, height {}".format(image.width, image.height))
    # img_rgb = image.convert('RGB')
    img_array = np.array(img_)   # height, width, 3

    plt.imshow(img_)
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image = Image.open("../data/dataset/meta-train/demo1/0-Color.png")
    img_tensor = transform(image)
    batch_data = torch.unsqueeze(img_tensor, 0)
    model = CNN(output_dim=5)
    for param in model.parameters():
        param.requires_grad = False

    model.pre_trained_model.fc.requires_grad_(True)
    model.fc1.requires_grad_(True)
    model.fc2.requires_grad_(True)
    model.fc3.requires_grad_(True)

    optim = Adam(model.parameters(), lr=0.001)

    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += 1
    print("number of learnable parameters {} (including both weights and bias)".format(count))

    predict = model(batch_data)
    index = torch.argmax(predict)
    print(index)