import torch
import torch.nn as nn
import torch.nn.functional as F


class BC_Model(nn.Module):
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

        fc_inshape = 1296 * 2 + state_len
        self.fc1 = nn.Linear(fc_inshape, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 3)


    def forward(self, img, goal_img, state_vec):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        flat_img = self.spacial_softmax(img).reshape(img.shape[0], -1)

        goal_img = self.conv1(goal_img)
        goal_img = self.conv2(goal_img)
        goal_img = self.conv3(goal_img)
        flat_goal_img = self.spacial_softmax(goal_img).reshape(img.shape[0], -1)

        img_and_state = torch.cat([flat_img, flat_goal_img, state_vec], dim=1)
        img_and_state = F.relu(self.fc1(img_and_state))
        img_and_state = F.relu(self.fc2(img_and_state))

        img_and_state = F.relu(self.fc3(img_and_state))
        img_and_state = self.fc4(img_and_state)
        return img_and_state
