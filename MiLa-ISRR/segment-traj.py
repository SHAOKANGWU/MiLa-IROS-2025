# TODO: In this script, the alpha in d^tr and d^val must be the same.

import os
import copy

from PIL import Image
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from Model import DMPIntegrator
from util import setRandomSeed, GaussianBasisFunction
from torch.nn.utils import clip_grad_value_
from util.run_aguments import save_args
from torchvision.transforms import v2
import random
from util.gmr import *
import learn2learn as l2l
import argparse
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50000, help="training epoch")
    parser.add_argument("--seed", type=int, default=1, help="the random seed")
    parser.add_argument("--batch", type=int, help="batch size (tasks numbers)")
    parser.add_argument("--save-name", type=str, help="the path to save your model")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"], help="device")
    parser.add_argument("--sample-size", type=int, default=30, help="policy sample size")
    parser.add_argument("--lr-classifier", type=float, default=0.0005, help="classifier learning rate")

    #################### meta inner update learnign rate ####################
    parser.add_argument("--inner-goal", type=float, default=0.00001, help="inner update learning rate for goal")
    #########################################################################

    #################### meta outer update learnign rate ####################
    parser.add_argument("--outer-goal", type=float, default=0.0001, help="outer update learning rate for goal")


    parser.add_argument("--dt", type=float, default=0.0333, help="collect data interval, do not change it")
    parser.add_argument("--beta", type=float, default=1.0, help="soft max coefficient, do not change it")

    # ------------------------------ DMP hyperparameters ------------------------------ #
    parser.add_argument("--kp", type=float, default=150.0, help="DMP kp, do not change it")
    parser.add_argument("--kv", type=float, default=25.0, help="DMP kv, do not change it")
    parser.add_argument("--alpha", type=float, default=5, help="DMP alpha, do not change it")
    parser.add_argument("--basis-num", type=float, default=8, help="DMP alpha, do not change it")
    # --------------------------------------------------------------------------------- #

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

DOWN_SAMPLEING = 1

# load MP weights
MP1 = torch.tensor(np.load("Skill-Repertoire/mp1.npy"), device=device)  # shape: (basis_num, 3)
MP2 = torch.tensor(np.load("Skill-Repertoire/mp2.npy"), device=device)  # shape: (basis_num, 3)
MP3 = torch.tensor(np.load("Skill-Repertoire/mp3.npy"), device=device)  # shape: (basis_num, 3)

cut_off_mp1 = 17 / 0.04 * 0.0333
cut_off_mp2 = 7.85 / 0.04 * 0.0333
cut_off_mp3 = 9.5 / 0.04 * 0.0333
dt = 0.0333

MP1_PINV = torch.load("Skill-Repertoire/gmm1_pinv")
MP2_PINV = torch.load("Skill-Repertoire/gmm2_pinv")
MP3_PINV = torch.load("Skill-Repertoire/gmm3_pinv")
d_in = range(0, 1)
d_out = range(1, 4)

TAU_MIN = 4.6
TAU_MAX = 8.3
SKILLS = {
            0: {"mp": MP1, "PINV": MP1_PINV, "cut_off_t": cut_off_mp1},
            1: {"mp": MP2, "PINV": MP2_PINV, "cut_off_t": cut_off_mp2},
            2: {"mp": MP3, "PINV": MP3_PINV, "cut_off_t": cut_off_mp3}
}

SKILLS_ORDER = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
# Now you can retrieve skill and its parameters by SKILLS[i] ()

basisFunc = GaussianBasisFunction(basis_num=args.basis_num, alpha=args.alpha, device=device)
integrator = DMPIntegrator(kp=args.kp, kv=args.kv, alpha=args.alpha, basisfunc=basisFunc)


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


post_transforms = v2.RandomChoice([
    v2.ColorJitter(brightness=.4, hue=0.0),     # adjust brightness is a good choice
    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),    # gaussian blur is a good choice
    AddGaussianNoise(0, 0.03),   # gaussian noise is also a good choice,
    v2.ColorJitter(brightness=0.0, hue=0.0),    # do nothing
], p=[0.25, 0.25, 0.25, 0.25])


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



def search_tau(goal, trajectory):
    mse_dist = torch.sum((trajectory - goal) ** 2, dim=1)
    idx = torch.argmin(mse_dist)
    return idx


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


def get_dmp_traj_and_pinv(initPos, initVel, goal, k, mp, mp_pinv, cut_off_t, tau):
    DMP_pos, _, _ = integrator(initPos, initVel, goal=goal, k=k, dt=DOWN_SAMPLEING * dt, weights=mp, tau=tau, device=device)

    # get pinverse covariance here
    mp_t_list = []
    mp_scale = cut_off_t / tau.item()
    for i in range(k-1):
        mp_t = (i + 1) * mp_scale * dt * DOWN_SAMPLEING
        mp_t_list.append(mp_t)

    pinv_t_list = np.array(list(mp_pinv.keys()))
    pinv_tensor = torch.zeros([k, 3, 3], device=device)

    for index, t in enumerate(mp_t_list):
        idx = np.argmin((t - pinv_t_list) ** 2)
        pinv_tensor[index] = mp_pinv[pinv_t_list[idx]]

    pinv_tensor[-1] = pinv_tensor[-2]

    return DMP_pos, pinv_tensor



def train(task, demo, mode, skill_order, g1, g2, g3):
    # TODO: here we do not have access to the skill order
    meta_train_traj = torch.tensor(
        get_traj_data("/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    target_traj = copy.deepcopy(meta_train_traj)

    tau1_idx = search_tau(g1.detach(), target_traj[int(TAU_MIN/dt):int(TAU_MAX/dt)])
    tau1 = tau1_idx * dt + TAU_MIN
    k1 = int(tau1.item() / dt) + 1

    tau2_idx = search_tau(g2.detach(), target_traj[int((tau1 + TAU_MIN)/dt):int((tau1 + TAU_MAX)/dt)])
    tau2 = tau2_idx * dt + TAU_MIN
    k2 = int(tau2.item() / dt) + 1

    tau3 = len(target_traj) * dt - tau1 - tau2
    k3 = int(tau3.item() / dt) + 1

    k1, k2, k3 = int(tau1.item() / dt / DOWN_SAMPLEING) + 1, int(tau2.item() / dt / DOWN_SAMPLEING) + 1, int(tau3.item() / dt / DOWN_SAMPLEING) + 1

    skill_index_1, skill_index_2, skill_index_3 = skill_order
    initPos = target_traj[0]
    initVel = torch.zeros_like(initPos, device=device)
    DMP_pos_1, skill1_pinv = get_dmp_traj_and_pinv(initPos, initVel, g1, k1, SKILLS[skill_index_1]["mp"],
                           SKILLS[skill_index_1]["PINV"], SKILLS[skill_index_1]["cut_off_t"], tau1)

    initPos = DMP_pos_1[-1]#.detach() # DMP_pos_1[-1].detach()# target_traj[k1]
    initVel = torch.zeros_like(initPos, device=device)
    DMP_pos_2, skill2_pinv = get_dmp_traj_and_pinv(initPos, initVel, g2, k2, SKILLS[skill_index_2]["mp"],
                           SKILLS[skill_index_2]["PINV"], SKILLS[skill_index_2]["cut_off_t"], tau2)

    initPos = DMP_pos_2[-1]#.detach()  # DMP_pos_2[-1].detach() # target_traj[k1+k2]
    initVel = torch.zeros_like(initPos, device=device)
    DMP_pos_3, skill3_pinv = get_dmp_traj_and_pinv(initPos, initVel, g3, k3, SKILLS[skill_index_3]["mp"],
                           SKILLS[skill_index_3]["PINV"], SKILLS[skill_index_3]["cut_off_t"], tau3)

    # calculate loss
    pinv_covariance = torch.vstack([skill1_pinv, skill2_pinv, skill3_pinv])
    predict_length = len(DMP_pos_1) + len(DMP_pos_2) + len(DMP_pos_3)

    meta_train_traj = target_traj[0::DOWN_SAMPLEING]
    target_length = len(meta_train_traj)
    predict_pos = torch.concat([DMP_pos_1, DMP_pos_2, DMP_pos_3])

    real_targ_pos = torch.zeros([predict_length, 3], device=device)
    real_targ_pos[0:target_length] = copy.deepcopy(meta_train_traj)
    for i in range(predict_length - target_length):
        real_targ_pos[target_length + i] = copy.deepcopy(meta_train_traj[-1])

    subtask1_loss, subtask2_loss, subtask3_loss = 0, 0, 0

    for i in range(predict_length):
        # loss = (predict_pos[i] - real_targ_pos[i]).reshape(1, -1) @ pinv_covariance[i] @ (predict_pos[i] - real_targ_pos[i]).reshape(-1, 1)
        loss = (predict_pos[i] - real_targ_pos[i]).reshape(1, -1) @ (predict_pos[i] - real_targ_pos[i]).reshape(-1, 1)

        # compute each sub-task loss
        if i < len(DMP_pos_1):
            subtask1_loss += torch.sum(loss)

        if len(DMP_pos_1) <= i < len(DMP_pos_1) + len(DMP_pos_2):
            subtask2_loss += torch.sum(loss)

        if i >= len(DMP_pos_1) + len(DMP_pos_2):
            subtask3_loss += torch.sum(loss)

    if skill_index_1 == 0:
        subtask1_loss *= 5

    if skill_index_2 == 0:
        subtask2_loss *= 5

    if skill_index_3 == 0:
        subtask3_loss *= 5

    total_loss = 1000 * (subtask1_loss + subtask2_loss + subtask3_loss) / predict_length

    return total_loss


def get_unique_list(list_of_list):
    unique_list = []
    for list in list_of_list:
        if list not in unique_list:
            unique_list.append(list)
    return unique_list


def main():

    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)
    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))
    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    skill_orders = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                order = [i, j, k]
                skill_orders.append(order)

    for epoch in range(0, args.epoch):
        # do training on training-set
        meta_train_loss = 0
        # sample a batch of paired demons
        train_batch_tasks, train_batch_demos = train_demos(args.batch)

        train_skill_order = np.zeros([args.batch, 3])

        for task_idx in range(args.batch):
            val_task, val_demo = train_batch_tasks[task_idx], train_batch_demos[task_idx][1]

            skill_order_loss = []
            skill1_traj, skill2_traj, skill3_traj = get_segment_traj(val_task, val_demo, mode="train")
            print("tau1 {}, tau2 {}, tau3 {}".format(len(skill1_traj) * dt, len(skill2_traj) * dt, len(skill3_traj) * dt))

            g1, g2, g3 = (torch.from_numpy(skill1_traj[-1]).to(device), torch.from_numpy(skill2_traj[-1]).to(device),
                          torch.from_numpy(skill3_traj[-1]).to(device))
            g1 += torch.tensor([0.0, 0.0, 0.0], device=device)

            skill_orders = [[0, 1, 2]]
            for skill_order in skill_orders:
                loss = train(val_task, val_demo, mode="train", skill_order=skill_order, g1=g1, g2=g2, g3=g3)
                skill_order_loss.append(loss)

            optimal_loss = min(skill_order_loss)
            optimal_order = skill_orders[skill_order_loss.index(optimal_loss)]
            train_skill_order[task_idx] = np.array(optimal_order)
            print("order {}, loss {}".format(optimal_order, optimal_loss))

        meta_train_loss /= args.batch

setRandomSeed(args.seed)
main()