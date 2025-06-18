import os
import copy
import sys
sys.path.append("../../")

from PIL import Image
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from Model import DMPIntegrator, MetaGoalCondiCNN
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


args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)

DOWN_SAMPLEING = 1

# load MP weights
MP1 = torch.tensor(np.load("../../Skill-Repertoire/mp1.npy"), device=device)  # shape: (basis_num, 3)
MP2 = torch.tensor(np.load("../../Skill-Repertoire/mp2.npy"), device=device)  # shape: (basis_num, 3)
MP3 = torch.tensor(np.load("../../Skill-Repertoire/mp3.npy"), device=device)  # shape: (basis_num, 3)

cut_off_mp1 = 17 / 0.04 * 0.0333
cut_off_mp2 = 7.85 / 0.04 * 0.0333
cut_off_mp3 = 9.5 / 0.04 * 0.0333
dt = 0.0333

MP1_PINV = torch.load("../../Skill-Repertoire/gmm1_pinv")
MP2_PINV = torch.load("../../Skill-Repertoire/gmm2_pinv")
MP3_PINV = torch.load("../../Skill-Repertoire/gmm3_pinv")
d_in = range(0, 1)
d_out = range(1, 4)

TAU_MIN = 4.6
TAU_MAX = 8.3
SKILLS = {
            0: {"mp": MP1, "PINV": MP1_PINV, "cut_off_t": cut_off_mp1},
            1: {"mp": MP2, "PINV": MP2_PINV, "cut_off_t": cut_off_mp2},
            2: {"mp": MP3, "PINV": MP3_PINV, "cut_off_t": cut_off_mp3}
}


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
    dt = args.dt
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



def predict_goal(task, demo, goal_model, current_frame, down_sample_factor=2, mode="wrong", meta_test=None, goal_img_index=None):
    img_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/img/color/{}.png".format(mode, task, demo, goal_img_index)
    img_tensor = transform(Image.open(img_path)).to(device)
    img_tensor = post_transforms(img_tensor)
    imgs_tensor = img_tensor[:, 55:335, 220:560].unsqueeze(0)
    # down sample the image by 2
    goal_img = imgs_tensor[:, :, ::down_sample_factor, ::down_sample_factor]

    img_path = "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/img/color/{}.png".format(mode, task, demo, current_frame)
    img_tensor = transform(Image.open(img_path)).to(device)
    img_tensor = post_transforms(img_tensor)
    imgs_tensor = img_tensor[:, 55:335, 220:560].unsqueeze(0)
    # down sample the image by 2
    Img = imgs_tensor[:, :, ::down_sample_factor, ::down_sample_factor]
    goal = goal_model(Img, goal_img, meta_test)[0]
    return goal, Img


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


def predict_order(task, demo, goal_model, mp_classifier, mode, meta_test, sample_size):
    # TODO: here we do not have access to the skill order
    meta_train_traj = torch.tensor(
        get_traj_data("/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    target_traj = copy.deepcopy(meta_train_traj)

    # First, we need to predict 3 goals, irrelative to the tau
    g1, img1 = predict_goal(task, demo, goal_model, current_frame=0, down_sample_factor=2, mode=mode,
                      meta_test=meta_test, goal_img_index=len(target_traj))
    logits1 = mp_classifier(img1)
    prob1 = F.softmax(logits1, dim=-1)
    tau1_idx = search_tau(g1.detach(), target_traj[int(TAU_MIN/dt):int(TAU_MAX/dt)])
    tau1 = tau1_idx * dt + TAU_MIN
    k1 = int(tau1.item() / dt) + 1

    g2, img2 = predict_goal(task, demo, goal_model, current_frame=k1, down_sample_factor=2, mode=mode,
                      meta_test=meta_test, goal_img_index=len(target_traj))
    logits2 = mp_classifier(img2)
    prob2 = F.softmax(logits2, dim=-1)
    tau2_idx = search_tau(g2.detach(), target_traj[int((tau1 + TAU_MIN)/dt):int((tau1 + TAU_MAX)/dt)])
    tau2 = tau2_idx * dt + TAU_MIN
    k2 = int(tau2.item() / dt) + 1

    g3, img3 = predict_goal(task, demo, goal_model, current_frame=k1+k2, down_sample_factor=2, mode=mode,
                      meta_test=meta_test, goal_img_index=len(target_traj))
    logits3 = mp_classifier(img3)
    prob3 = F.softmax(logits3, dim=-1)
    tau3 = len(target_traj) * dt - tau1 - tau2
    k3 = int(tau3.item() / dt) + 1

    # remember also output the probability of softmax
    softmax_prob = torch.vstack([prob1, prob2, prob3])

    return torch.vstack([img1, img2, img3]), softmax_prob



def train(task, demo, goal_model, mode, meta_test, skill_order):
    # TODO: here we do not have access to the skill order
    meta_train_traj = torch.tensor(
        get_traj_data("/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    target_traj = copy.deepcopy(meta_train_traj)

    # First, we need to predict 3 goals, irrelative to the tau
    g1, img1 = predict_goal(task, demo, goal_model, current_frame=0, down_sample_factor=2, mode=mode,
                      meta_test=meta_test, goal_img_index=len(target_traj))
    tau1_idx = search_tau(g1.detach(), target_traj[int(TAU_MIN/dt):int(TAU_MAX/dt)])
    tau1 = tau1_idx * dt + TAU_MIN
    k1 = int(tau1.item() / dt) + 1

    g2, img2 = predict_goal(task, demo, goal_model, current_frame=k1, down_sample_factor=2, mode=mode,
                      meta_test=meta_test, goal_img_index=len(target_traj))
    tau2_idx = search_tau(g2.detach(), target_traj[int((tau1 + TAU_MIN)/dt):int((tau1 + TAU_MAX)/dt)])
    tau2 = tau2_idx * dt + TAU_MIN
    k2 = int(tau2.item() / dt) + 1

    g3, img3 = predict_goal(task, demo, goal_model, current_frame=k1+k2, down_sample_factor=2, mode=mode,
                      meta_test=meta_test, goal_img_index=len(target_traj))
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
        loss = (predict_pos[i] - real_targ_pos[i]).reshape(1, -1) @ pinv_covariance[i] @ (predict_pos[i] - real_targ_pos[i]).reshape(-1, 1)

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

    total_loss = (subtask1_loss + subtask2_loss + subtask3_loss) / predict_length

    return total_loss, torch.vstack([img1, img2, img3])

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

    # trainable models
    goal_model = MetaGoalCondiCNN(output_dim=3).to(device)
    maml = l2l.algorithms.MAML(goal_model, lr=args.inner_goal, first_order=False, allow_unused=True)  # inner update learning rate
    optimizer_goal = Adam(maml.parameters(), lr=args.outer_goal)                    # outer update learning rate

    # TODO: load model -> Here we provide a pretrained model.
    state_dict = torch.load("saved/pretrained-model/nonNovel/model_eval.pth")
    goal_model.load_state_dict(state_dict["goal_model"])
    optimizer_goal.load_state_dict(state_dict["goal_opt"])

    optimizer_goal.zero_grad()

    adaptation_steps = 1
    max_eval_loss = 1e6

    for epoch in range(0, args.epoch):
        optimizer_goal.zero_grad()

        # do training on training-set
        meta_train_loss = 0
        # sample a batch of paired demons
        train_batch_tasks, train_batch_demos = train_demos(args.batch)

        start_time = time.time()
        for task_idx in range(args.batch):
            train_task, train_demo = train_batch_tasks[task_idx][0], train_batch_demos[task_idx][0]
            val_task, val_demo = train_batch_tasks[task_idx][1], train_batch_demos[task_idx][1]

            learner = maml.clone()
            skill_order = [0, 1, 2]
            for step in range(adaptation_steps):
                loss1, _ = train(train_task, train_demo, learner, mode="train", meta_test=False, skill_order=skill_order)
                learner.adapt(loss1)
            loss, _ = train(val_task, val_demo, learner, mode="train", meta_test=True, skill_order=skill_order)

            loss.backward()
            meta_train_loss += loss.item()

        meta_train_loss /= args.batch


        if (epoch+1) % 100 == 0:
            meta_valid_loss = 0
            validation_batch = 300

            valid_batch_tasks, valid_batch_demos = valid_demos(validation_batch)
            valid_batch_prob = torch.zeros([3, 3]).to(device)  # [num_tasks, num_mps], where each value is a probability

            for task_idx in range(validation_batch):
                train_task, train_demo = valid_batch_tasks[task_idx][0], valid_batch_demos[task_idx][0]
                val_task, val_demo = valid_batch_tasks[task_idx][1], valid_batch_demos[task_idx][1]

                skill_order = [0, 1, 2]
                learner = maml.clone()
                for step in range(adaptation_steps):
                    loss1, _ = train(train_task, train_demo, learner, mode="validation", meta_test=False,
                                     skill_order=skill_order)
                    learner.adapt(loss1)
                loss, _ = train(val_task, val_demo, learner, mode="validation", meta_test=True, skill_order=skill_order)

                meta_valid_loss += loss.item()

            meta_valid_loss /= validation_batch
            valid_batch_prob /= validation_batch

            logs = 'epoch {}, Meta Valid Loss {:.2f}\n'.format(epoch, meta_valid_loss)
            print(logs, end="")
            file.write(logs)
            file.flush()

            if max_eval_loss > meta_valid_loss:
                max_eval_loss = meta_valid_loss
                state_dict = {"goal_model": goal_model.state_dict(),
                              "goal_opt": optimizer_goal.state_dict()}
                torch.save(state_dict, os.path.join(saved_folder_name, "model_eval.pth"))


        state_dict = {"goal_model": goal_model.state_dict(),
                      "goal_opt": optimizer_goal.state_dict()}
        torch.save(state_dict, os.path.join(saved_folder_name, "model.pth"))

        logs = '\n'
        print(logs, end="")
        file.write(logs)
        logs = 'epoch {}\n'.format(epoch)
        print(logs, end="")
        file.write(logs)
        logs = 'Meta Train Loss {:.2f}\n'.format(meta_train_loss)
        print(logs, end="")
        file.write(logs)
        file.flush()

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if not p.grad is None:
                p.grad.data.mul_(1.0 / args.batch)
        optimizer_goal.step()

        end_time = time.time()
        print("time consumed:", end_time - start_time)

setRandomSeed(args.seed)
main()