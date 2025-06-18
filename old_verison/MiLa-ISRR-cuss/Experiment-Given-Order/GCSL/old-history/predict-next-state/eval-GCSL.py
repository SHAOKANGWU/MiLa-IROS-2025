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


test_task = [i for i in range(1, 17, 1)]        # task 1 - 16
test_dataset = {}
for task in test_task:
    # each task has 10 demos, we use 8 for training, 1 for validation and 1 for testing
    if 1 <= task <= 4:
        demos = [i for i in range(1, 7, 1)]
    else:
        demos = [i for i in range(1, 9, 1)]
    test_dataset[task] = demos

def test_demos(batch_size):

    tasks = list(test_dataset.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(test_dataset[task], 1)[0]
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

    traj = torch.tensor(
        get_traj_data(
            "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task_idx, demo_idx))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)

    imgs_folder_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/img/color".format(mode, task_idx, demo_idx)
    imgs_name_list = os.listdir(imgs_folder_path)
    sorted_img_list = sorted(imgs_name_list, key=lambda x: int(x.split(".")[0]))
    imgs_name_list = sorted_img_list[0:len(traj)]
    # the images path for the whole demonstration
    imgs_path_list = [os.path.join(imgs_folder_path, name) for name in imgs_name_list]

    # Sample a batch of indexes from this demonstration (Batch = N)
    sample_idx = np.array(random.sample(list(range(0, len(imgs_path_list)-interval-1)), sample_size))

    # read the goal image from this demonstration, repeat it with shape N * 3 * width * height
    clearn_goal_image = transform(Image.open(imgs_path_list[-1])).to(device)
    goal_obs = []
    for _ in sample_idx:
        goal_image = post_transforms(clearn_goal_image)
        goal_image = goal_image[:, 55:335, 220:560].unsqueeze(0)
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

    assert len(current_img) == len(goal_image_tensor) == len(current_state) == len(next_state)
    return current_img, goal_image_tensor, current_state, next_state

def main():
    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)
    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))
    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # trainable models
    model = GCSL_Model(output_dim=3).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    state_dict = torch.load("saved/GCSL/model_eval.pth")
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["opt"])

    # random sample tasks and demos from our dataset. In this procedure, we can do batch learning.
    test_batch_tasks, test_batch_demos = test_demos(args.batch)

    total_loss1 = 0     # loss between prediction (next state) and next state
    total_loss2 = 0     # loss between prediction (next state) and current state
    for idx in range(args.batch):
        test_task, test_demo = test_batch_tasks[idx], test_batch_demos[idx]
        current_img, goal_image_tensor, current_state, next_state = get_demo_from_raw_data("test",
                                       test_task, test_demo, sample_size=args.sample_size, interval=args.interval)
        with torch.no_grad():
            prediction = model(current_img, goal_image_tensor, current_state)
            loss1 = F.mse_loss(100 * prediction, 100 * next_state, reduction="sum") / args.sample_size
            loss2 = F.mse_loss(100 * prediction, 100 * current_state, reduction="sum") / args.sample_size
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            print('loss between prediction (next state) and next state {:.2f}\n'.format(loss1))
            print('loss between prediction (next state) and current state {:.2f}\n'.format(loss2))

    total_loss1 /= args.batch
    total_loss2 /= args.batch

    print('total average loss between prediction (next state) and next state {:.2f}\n'.format(total_loss1))
    print('total average loss between prediction (next state) and current state {:.2f}\n'.format(total_loss2))

setRandomSeed(args.seed)
main()