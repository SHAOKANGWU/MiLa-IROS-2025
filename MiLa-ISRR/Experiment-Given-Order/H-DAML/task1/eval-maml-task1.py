import sys
sys.path.append("../../../")
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
from Model import VisualModel, VisualModel_Norm_Spatial, VisualModel_Norm_Softmax2d, VisualModel_Softmax2d_Deep
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
    parser.add_argument("--multi-dt", type=int, help="multiplcation factor for dt")


    #################### meta outer update learnign rate ####################
    parser.add_argument("--inner-lr", type=float, default=0.001, help="inner update learning rate")
    parser.add_argument("--outer-lr", type=float, default=0.001, help="outer update learning rate")
    parser.add_argument("--adaptation-steps", type=int, default=1, help="number of adaptation steps in inner loop update")
    parser.add_argument("--subtask", type=int, help="subtask index")

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
        demo = random.sample(train_dataset[task], 2)
        batch_demos.append(demo)

    return batch_tasks, batch_demos


def valid_demos(batch_size):

    tasks = list(validation_dataset.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(validation_dataset[task], 2)
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
        demo = random.sample(test_dataset[task], 2)
        batch_demos.append(demo)

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
    image_tensor = image[:, 0:335, 220:560].unsqueeze(0)
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
    next_state = traj[sample_idx + args.multi_dt]   # TODO: We predict the next state in the future with 0.1 s
    gripper_tensor = gripper_data[sample_idx].reshape(-1, 1)

    goal_img_tensor = read_img_from_path(imgs_path_list[-1], mode="test")

    assert len(current_img) == len(current_state) == len(next_state) == len(gripper_tensor)
    return current_img, current_state, next_state, gripper_tensor, goal_img_tensor


def train(task_idx, demo_idx, mode, model, subtask, meta_testing, iter_predict):
    traj1, traj2, traj3 = get_segment_traj(task_idx, demo_idx, mode)

    if subtask == 0:
        traj_range = (0+50, len(traj1)+5)
    elif subtask == 1:
        traj_range = (len(traj1)+30, len(traj1)+len(traj2)) # There is some stop during two tasks.
    elif subtask == 2:
        traj_range = (len(traj1)+len(traj2)+30, len(traj1)+len(traj2)+len(traj3))
    else:
        raise NotImplementedError("subtask must be 0, 1 or 2")

    current_img, current_state, next_state, _, goal_img = get_demo_from_raw_data(mode, task_idx, demo_idx,
                            sample_size=args.sample_size, interval=args.interval, traj_range=traj_range)

    if not iter_predict:
        prediction = model(current_img, current_state, meta_testing)

        target = next_state - current_state

        loss = args.loss_factor * F.mse_loss(prediction, target, reduction="sum") / len(prediction)
        return loss, prediction, target

    else:
        start_pos = current_state[0:1]
        prediction = []
        for i in range(len(next_state)):
            predict = model(current_img[i:i+1], start_pos, meta_testing)
            predict = torch.clamp(predict, -0.03, 0.03)
            start_pos = (predict + start_pos).detach()
            prediction.append(start_pos)
        prediction = torch.cat(prediction)  # predicted trajectory

        target = next_state
        loss = args.loss_factor * F.mse_loss(prediction, target, reduction="sum") / len(prediction)  # args.sample_size
        return loss, prediction, target



def main():
    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)
    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))
    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # trainable models
    model = VisualModel_Softmax2d_Deep().to(device)

    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=False, allow_unused=True)
    optimizer = Adam(maml.parameters(), lr=args.outer_lr)

    adaptation_steps = args.adaptation_steps
    maml.train()

    state_dict = torch.load("saved/maml-softmax2d-delta-task1-hyper1/model_eval.pth")
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["opt"])

    optimizer.zero_grad()

    train_batch_tasks, train_batch_demos = test_demos(args.batch)
    meta_trian_loss = 0

    saved_data = {"predict_traj": [], "target_traj": [], "loss": [], "order": [], "task": [], "demo": []}


    for idx in range(args.batch):
        train_task, train_demo = train_batch_tasks[idx], train_batch_demos[idx][0]
        val_task, val_demo = train_batch_tasks[idx], train_batch_demos[idx][1]
        subtask = args.subtask # random.choice([0, 1, 2])

        learner = maml.clone()
        # Inner update
        for step in range(adaptation_steps):
            loss1, _, _ = train(train_task, train_demo, "test", learner, subtask, meta_testing=False, iter_predict=False)
            learner.adapt(loss1) #, grad_clip_value=30)

        with torch.no_grad():
            loss, predict_traj, target_traj = train(val_task, val_demo, "test", learner, subtask, meta_testing=True, iter_predict=True)
            meta_trian_loss += loss.item()

        saved_data["task"].append([train_task, val_task])
        saved_data["demo"].append([train_demo, val_demo])
        saved_data["loss"].append(loss.item())

        saved_data["predict_traj"].append(predict_traj)
        saved_data["target_traj"].append(target_traj)

    meta_trian_loss /= args.batch
    torch.save(saved_data, "test-iter")
    print("loss {:.6f}".format(meta_trian_loss))

setRandomSeed(args.seed)
main()