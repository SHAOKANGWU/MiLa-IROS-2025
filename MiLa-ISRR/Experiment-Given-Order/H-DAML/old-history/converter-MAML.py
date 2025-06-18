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
from Model import MAML_BT_Model_NoGripper
import time
import learn2learn as l2l

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
    parser.add_argument("--loss-factor", type=int, default=1, help="multiplcation factor for loss")

    #################### meta outer update learnign rate ####################
    parser.add_argument("--inner-lr", type=float, default=0.001, help="inner update learning rate")
    parser.add_argument("--outer-lr", type=float, default=0.001, help="outer update learning rate")
    parser.add_argument("--adaptation-steps", type=int, default=1, help="number of adaptation steps in inner loop update")

    args = parser.parse_args()
    return args


random.seed(3)

train_dataset = {
    "origin_block_gray_container": [1, 19, 21, 23, 25, 27, 29, 31, 33],
    "origin_block_white_container": [10, 91, 93, 95, 97, 99, 101, 103, 105],

    "green_block_gray_container": [2, 20, 35, 37, 39, 41, 43, 45, 47],
    "green_block_white_container": [11, 92, 107, 109, 111, 113, 115, 117, 119],

    "yellow_block_gray_container": [3, 22, 36, 49, 51, 53, 55, 57, 59],
    "yellow_block_white_container": [12, 94, 108, 121, 123, 125, 127, 129, 131],

    "blue_block_gray_container": [4, 24, 38, 50, 61, 63, 65, 67, 69],
    "blue_block_white_container": [13, 96, 110, 122, 133, 135, 137, 139, 141],

    "pink_cup_gray_container": [5, 26, 40, 52, 62, 71, 73, 75, 77],
    "pink_cup_white_container": [14, 98, 112, 124, 134, 143, 145, 147, 149],

    "light_green_cup_gray_container": [6, 28, 42, 54, 64, 72, 79, 81, 83],
    "light_green_cup_white_container": [15, 100, 114, 126, 136, 144, 151, 153, 155],

    "light_yellow_cup_gray_container": [7, 30, 44, 56, 66, 74, 80, 85, 87],
    "light_yellow_cup_white_container": [16, 102, 116, 128, 138, 146, 152, 157, 159],

    "dark_red_basket_gray_container": [8, 32, 46, 58, 68, 76, 82, 86, 89],
    "dark_red_basket_white_container": [17, 104, 118, 130, 140, 148, 154, 158, 161],

    "ocean_blue_basket_gray_container": [9, 34, 48, 60, 70, 78, 84, 88, 90],
    "ocean_blue_basket_white_container": [18, 106, 120, 132, 142, 150, 156, 160, 162]
}

validation_dataset = {
    "white_cup_gray_container": [1, 7, 9],

    "gray_cup_white_container": [4, 13, 15],

    "block_combination_1_gray_container": [2, 8, 11],
    "block_combination_1_white_container": [5, 14, 17],

    "block_combination_2_gray_container": [3, 10, 12],
    "block_combination_2_white_container": [6, 16, 18],
}

def train_demos(batch_size):
    task_keys = list(train_dataset.keys())

    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task_name = random.choice(task_keys)

        meta_task = random.choices(train_dataset[task_name], k=2)
        meta_demo = []

        for task in meta_task:
            # get number of demos from the task
            task_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/train/task{}".format(task)
            num_of_demos = len(os.listdir(task_path))
            # random sample a demo
            demos = [i for i in range(1, num_of_demos+1, 1)]
            random.shuffle(demos)

            demo = random.choices(demos, k=1)[0]
            meta_demo.append(demo)

        batch_tasks.append(meta_task)
        batch_demos.append(meta_demo)

    return batch_tasks, batch_demos


def valid_demos(batch_size):
    task_keys = list(validation_dataset.keys())

    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task_name = random.choice(task_keys)    # e.g. white_cup_gray_container

        meta_task = random.choices(validation_dataset[task_name], k=2)  # e.g. [1, 7]
        meta_demo = []

        for task in meta_task:
            # get number of demos from the task
            task_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/validation/task{}".format(task)
            num_of_demos = len(os.listdir(task_path))
            # random sample a demo
            demos = [i for i in range(1, num_of_demos+1, 1)]
            random.shuffle(demos)

            demo = random.choices(demos, k=1)[0]
            meta_demo.append(demo)

        batch_tasks.append(meta_task)
        batch_demos.append(meta_demo)

    return batch_tasks, batch_demos

test_dataset = {
    "red_block_pink_container": [1, 5, 7, 9],

    "light_blue_cup_pink_container": [2, 6, 11, 13],

    "block_combination_1_pink_container": [3, 8, 12, 15],

    "block_combination_2_pink_container": [4, 10, 14, 16]
}


def test_demos(batch_size):
    task_keys = list(test_dataset.keys())

    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task_name = random.choice(task_keys)    # e.g. white_cup_gray_container

        meta_task = random.choices(test_dataset[task_name], k=2)  # e.g. [1, 7]
        meta_demo = []

        for task in meta_task:
            # get number of demos from the task
            task_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/test/task{}".format(task)
            num_of_demos = len(os.listdir(task_path))
            # random sample a demo
            demos = [i for i in range(1, num_of_demos+1, 1)]
            random.shuffle(demos)

            demo = random.choices(demos, k=1)[0]
            meta_demo.append(demo)

        batch_tasks.append(meta_task)
        batch_demos.append(meta_demo)

    return batch_tasks, batch_demos


args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)
DOWN_SAMPLE_FACTOR = args.down_sample_factor
dt = 0.0333

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


post_transforms = v2.RandomChoice([
    v2.ColorJitter(brightness=.2, hue=0.0),     # adjust brightness is a good choice
    AddGaussianNoise(0, 0.02),   # gaussian noise is also a good choice,
    v2.ColorJitter(brightness=0.0, hue=0.0),    # do nothing
], p=[0.33, 0.33, 0.34])


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


def classify_gripper_data(gripper_data_array, close_width=0.04, open_width=0.05):
    # given the close width and open width of the gripper data, classify it into binary data (0 or 1), where 0 means open, 1 means close
    open_index, close_index = -1, -1
    gripper_data_status = np.zeros_like(gripper_data_array)
    for index, data in enumerate(gripper_data_array):
        if data <= close_width:
            close_index = index
            break

    reverse_gripper_data_array = gripper_data_array[::-1]
    for index, data in enumerate(reverse_gripper_data_array):
        if data <= open_width:
            open_index = len(gripper_data_array) - index
            break

    gripper_data_status[close_index:open_index] = 1
    return gripper_data_status


def segment_tau(tau_data):
    # tau_data is a binary data
    first_skill_end_idx = int(torch.argmax(tau_data))
    second_skill_end_idx = int(torch.argmin(tau_data[first_skill_end_idx:]) + first_skill_end_idx)
    third_skill_end_idx = len(tau_data)
    tau1 = dt * first_skill_end_idx
    tau2 = dt * second_skill_end_idx - tau1
    tau3 = dt * third_skill_end_idx - tau2 - tau1
    return torch.tensor(tau1, device=device), torch.tensor(tau2, device=device), torch.tensor(tau3, device=device)


def get_segment_traj(task, demo, mode):
    gripper_data = np.array(get_traj_data(
        "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/gripper.txt".format(mode, task, demo)))
    inner_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
    tau1, tau2, tau3 = segment_tau(inner_gripper)
    meta_train_traj = torch.tensor(
        get_traj_data(
            "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    skill1_traj = meta_train_traj[0:int(tau1/dt)].cpu().data.numpy()
    skill2_traj = meta_train_traj[int(tau1/dt): int((tau1 + tau2)/dt)].cpu().data.numpy()
    skill3_traj = meta_train_traj[int((tau1 + tau2)/dt):].cpu().data.numpy()
    return skill1_traj, skill2_traj, skill3_traj


def read_img_from_path(img_path, mode):
    image = transform(Image.open(img_path)).to(device)
    if mode == "validation" or mode == "test":
        pass
    elif mode == "train":
        image = post_transforms(image)
    else:
        raise NotImplementedError("mode must be train, validation, or test")
    image_tensor = image[:, 55:335, 220:560].unsqueeze(0)
    image_tensor = image_tensor[:, :, ::DOWN_SAMPLE_FACTOR, ::DOWN_SAMPLE_FACTOR]
    return image_tensor


def get_demo_from_raw_data(mode, task_idx, demo_idx, sample_size, interval, traj_range):

    start_index, end_index = traj_range

    gripper_data = np.array(get_traj_data(
        "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/gripper.txt".format(mode, task_idx, demo_idx)))
    gripper_data = torch.tensor(gripper_data)[0:-90].to(device)
    gripper_data = gripper_data[start_index:end_index]

    traj = torch.tensor(
        get_traj_data(
            "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task_idx, demo_idx))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    traj = traj[start_index:end_index]

    assert len(traj) == len(gripper_data)

    imgs_folder_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/img/color".format(mode, task_idx, demo_idx)
    imgs_name_list = os.listdir(imgs_folder_path)
    sorted_img_list = sorted(imgs_name_list, key=lambda x: int(x.split(".")[0]))
    imgs_name_list = sorted_img_list[start_index:end_index]
    # the images path for the whole demonstration
    imgs_path_list = [os.path.join(imgs_folder_path, name) for name in imgs_name_list]

    # Sample a batch of indexes from this demonstration (Batch = N)
    # sample_idx = np.array(random.sample(list(range(0, len(imgs_path_list)-interval-1)), sample_size))

    # TODO: input all of the demonstration to NN
    sample_idx = list(range(0, len(imgs_path_list)-3-1))
    step = random.choice(list(range(0, interval)))
    sample_idx = np.array(sample_idx[step::interval])

    # read the current image (sample indexes) from this demonstration
    img_obs = []
    for idx in sample_idx:
        img_tensor = read_img_from_path(imgs_path_list[idx], mode=mode)[0]
        img_obs.append(img_tensor)
    current_img = torch.stack(img_obs)

    # display_image(goal_image_tensor[0])
    assert len(imgs_path_list) == len(traj)
    current_state = traj[sample_idx]
    next_state = traj[sample_idx + 3]   # TODO: We predict the next state in the future with 0.1 s
    gripper_tensor = gripper_data[sample_idx].reshape(-1, 1)

    assert len(current_img) == len(current_state) == len(next_state) == len(gripper_tensor)
    return current_img, current_state, next_state, gripper_tensor


def train(task_idx, demo_idx, mode, model, subtask, meta_testing):
    traj1, traj2, traj3 = get_segment_traj(task_idx, demo_idx, mode)

    if subtask == 0:
        traj_range = (0, len(traj1))
    elif subtask == 1:
        traj_range = (len(traj1), len(traj1)+len(traj2))
    elif subtask == 2:
        traj_range = (len(traj1)+len(traj2), len(traj1)+len(traj2)+len(traj3))
    else:
        raise NotImplementedError("subtask must be 0, 1 or 2")

    current_img, current_state, next_state, _ = get_demo_from_raw_data(mode, task_idx, demo_idx,
                            sample_size=args.sample_size, interval=args.interval, traj_range=traj_range)

    # the prediction is the delta state (prediction = s_{t+1} - s_t)
    if not meta_testing:
        prediction = model(current_img, current_state)

        target = next_state - current_state
        # compute MSE loss in cm
        loss = args.loss_factor * F.mse_loss(prediction, target, reduction="sum") / len(prediction) #args.sample_size
        return loss

    else:
        start_pos = current_state[0:1]
        predict_state = []
        delta_state = []

        for i in range(len(next_state)):
            # predict the increment based on current state
            predict = model(current_img[i:i+1], start_pos)
            delta_state.append(predict.detach())

            # get the next state
            start_pos = torch.clamp(predict.detach(), -0.05, 0.05) + start_pos
            predict_state.append(start_pos)

        # save as tensor
        predict_state = torch.cat(predict_state)
        delta_state = torch.cat(delta_state)

        target = next_state - current_state
        loss = args.loss_factor * F.mse_loss(delta_state, target, reduction="sum") / len(delta_state) #args.sample_size
        return loss, predict_state, next_state


def main():
    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)
    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))
    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # trainable models
    model = MAML_BT_Model_NoGripper(output_dim=3).to(device) # Controller_NN
    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=False, allow_unused=True)
    optimizer = Adam(maml.parameters(), lr=args.outer_lr)
    max_eval_loss = 1e9
    adaptation_steps = args.adaptation_steps
    maml.train()

    state_dict = torch.load("saved/maml-bt-delta-23/model_eval.pth")
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["opt"])

    saved_data = {"predict_traj": [], "target_traj": [], "loss": [],
                  "order": [], "task": [], "demo": [],
                  }

    # train_task, train_demo = 1, 1   # for place
    train_task, train_demo = 137, 5   # for push
    subtask = 2

    learner = maml.clone()
    # Inner update
    for step in range(adaptation_steps):
        loss1 = train(train_task, train_demo, "train", learner, subtask, meta_testing=False)
        learner.adapt(loss1)
        state_dict = {"model": learner.module.state_dict()}
        torch.save(state_dict, "meta-update-once-goal-model")

    update_once_model = MAML_BT_Model_NoGripper(output_dim=3).to(device)
    state_dict = torch.load("meta-update-once-goal-model")
    update_once_model.load_state_dict(state_dict["model"])
    traced_script_module = torch.jit.script(update_once_model)
    traced_script_module.save("meta-goal-model.pt")


setRandomSeed(args.seed)
main()