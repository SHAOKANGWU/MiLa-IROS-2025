import sys
sys.path.append("../../")
from torch.utils.data import Dataset
import os
import copy
from PIL import Image
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from util import setRandomSeed
from torch.nn.utils import clip_grad_value_
from util.run_aguments import save_args
from torchvision.transforms import v2
import random
from util.gmr import *
import argparse
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from Model import GCSL_Model
import time
from matplotlib.pyplot import MultipleLocator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50000, help="training epoch")
    parser.add_argument("--seed", type=int, default=1, help="the random seed")
    parser.add_argument("--batch", type=int, help="batch size (tasks numbers)")
    parser.add_argument("--save-name", type=str, help="the path to save your model")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"], help="device")
    parser.add_argument("--sample-size", type=int, default=40, help="sample image size for each demonstration")
    parser.add_argument("--down-sample-factor", type=int, default=2, help="down sample factor for images")
    parser.add_argument("--interval", type=int, default=3, help="interval for two consecutive images")

    #################### meta outer update learnign rate ####################
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for the BC policy")
    parser.add_argument("--idx", type=int, default=0, help="index")

    args = parser.parse_args()
    return args


random.seed(3)

train_task = [i for i in range(1, 163, 1)]      # task 1 - 162
validation_task = [i for i in range(1, 19, 1)]  # task 1 - 18

train_dataset, validation_dataset = {}, {}

for task in train_task:
    # each task has 10 demos, we use 8 for training, 1 for validation and 1 for testing
    if 1 <= task <= 18:
        demos = [i for i in range(1, 7, 1)]
    else:
        demos = [i for i in range(1, 9, 1)]
    random.shuffle(demos)
    train_dataset[task] = demos

for task in validation_task:
    # each task has 10 demos, we use 8 for training, 1 for validation and 1 for testing
    if 1 <= task <= 6:
        demos = [i for i in range(1, 7, 1)]
    else:
        demos = [i for i in range(1, 9, 1)]
    random.shuffle(demos)
    validation_dataset[task] = demos


def train_demos(batch_size):

    tasks = list(train_dataset.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(train_dataset[task], 1)[0]
        batch_demos.append(demo)

    return batch_tasks, batch_demos


def valid_demos(batch_size):

    tasks = list(validation_dataset.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(validation_dataset[task], 1)[0]
        batch_demos.append(demo)

    return batch_tasks, batch_demos


args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)
DOWN_SAMPLE_FACTOR = args.down_sample_factor

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


post_transforms = v2.RandomChoice([
    v2.ColorJitter(brightness=.4, hue=0.0),     # adjust brightness is a good choice
    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),    # gaussian blur is a good choice
    AddGaussianNoise(0, 0.03),   # gaussian noise is also a good choice,
    v2.ColorJitter(brightness=0.0, hue=0.0),    # do nothing
], p=[0.25, 0.25, 0.25, 0.25])


transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.double),
])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def display_image(image_tensor):
    image_numpy = image_tensor.cpu().data.numpy()
    image = np.transpose(image_numpy, [1, 2, 0])
    plt.imshow(image)
    plt.show()


def get_traj_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data

def read_img_from_path(img_path):
    image = transform(Image.open(img_path)).to(device)
    # image = post_transforms(image)
    image_tensor = image[:, 55:335, 220:560].unsqueeze(0)
    image_tensor = image_tensor[:, :, ::DOWN_SAMPLE_FACTOR, ::DOWN_SAMPLE_FACTOR]
    return image_tensor


def get_demo_from_raw_data(mode, task_idx, demo_idx, sample_size, interval):

    gripper_data = np.array(get_traj_data(
        "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/gripper.txt".format(mode, task_idx, demo_idx)))
    gripper_data = torch.tensor(gripper_data)[0:-90].to(device)

    traj = torch.tensor(
        get_traj_data(
            "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task_idx, demo_idx))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    assert len(traj) == len(gripper_data)

    imgs_folder_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/img/color".format(mode, task_idx, demo_idx)
    imgs_name_list = os.listdir(imgs_folder_path)
    sorted_img_list = sorted(imgs_name_list, key=lambda x: int(x.split(".")[0]))
    imgs_name_list = sorted_img_list[0:len(traj)]
    # the images path for the whole demonstration
    imgs_path_list = [os.path.join(imgs_folder_path, name) for name in imgs_name_list]

    # Sample a batch of indexes from this demonstration (Batch = N)
    sample_idx = np.array(list(range(0, len(imgs_path_list)-interval-1, interval)))

    # read the goal image from this demonstration, repeat it with shape N * 3 * width * height
    clearn_goal_image = transform(Image.open(imgs_path_list[-1])).to(device)
    goal_obs = []
    for _ in sample_idx:
        # goal_image = post_transforms(clearn_goal_image)
        goal_image = clearn_goal_image[:, 55:335, 220:560].unsqueeze(0)
        goal_image = goal_image[:, :, ::DOWN_SAMPLE_FACTOR, ::DOWN_SAMPLE_FACTOR][0]
        goal_obs.append(goal_image)
    goal_image_tensor = torch.stack(goal_obs)

    # read the current image (sample indexes) from this demonstration
    img_obs = []
    for idx in sample_idx:
        img_tensor = read_img_from_path(imgs_path_list[idx])[0]
        img_obs.append(img_tensor)
    current_img = torch.stack(img_obs)

    # display_image(goal_image_tensor[0])
    assert len(imgs_path_list) == len(traj)
    current_state = traj[sample_idx]
    next_state = traj[sample_idx + interval]
    gripper_tensor = gripper_data[sample_idx].reshape(-1, 1)

    assert len(current_img) == len(goal_image_tensor) == len(current_state) == len(next_state) == len(gripper_tensor)
    return current_img, goal_image_tensor, current_state, next_state, gripper_tensor


def main():
    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)
    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))
    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # trainable models
    model = GCSL_Model(output_dim=3).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    max_eval_loss = 1e9

    state_dict = torch.load("saved/GCSL-increment/model_eval.pth")
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["opt"])


    # random sample tasks and demos from our dataset. In this procedure, we can do batch learning.
    train_batch_tasks, train_batch_demos = train_demos(1000)
    idx = 1 # args.idx

    train_task, train_demo = train_batch_tasks[idx], train_batch_demos[idx]
    current_img, goal_image_tensor, current_state, next_state, gripper_tensor = get_demo_from_raw_data("train",
                                   train_task, train_demo, sample_size=args.sample_size, interval=args.interval)

    state = current_state[0].unsqueeze(0)
    predicted_traj = []
    for i in range(len(current_img)):
        extended_state = torch.hstack([state, gripper_tensor[i].unsqueeze(0)])
        prediction = model(current_img[i].unsqueeze(0), goal_image_tensor[i].unsqueeze(0), extended_state)
        predict_next_state = state + prediction
        state = predict_next_state
        predicted_traj.append(predict_next_state[0])

    # torch.save(predicted_traj, "predicted_traj")
    # torch.save(next_state, "target_traj")

    # plot 3D trajectory data
    ax = plt.axes(projection='3d')

    x_scale = MultipleLocator(0.1)
    y_scale = MultipleLocator(0.1)
    z_scale = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_scale)
    ax.yaxis.set_major_locator(y_scale)
    ax.zaxis.set_major_locator(z_scale)
    ax.set_xlim(-0.2, 0.8)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.3, 0.7)

    predicted_traj = torch.stack(predicted_traj).cpu().data.numpy()
    target_traj = next_state.cpu().data.numpy()

    ax.plot3D(predicted_traj[:, 0], predicted_traj[:, 1], predicted_traj[:, 2], label="predict traj")
    ax.plot3D(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], label="target traj")
    plt.legend()
    plt.show()


setRandomSeed(args.seed)
main()