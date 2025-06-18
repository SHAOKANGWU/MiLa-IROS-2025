import os
import copy
import random
import torch
from PIL import Image
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from Model import CNN, DMPIntegrator
from util import setRandomSeed, GaussianBasisFunction, sample_demos, train_demos, test_demos
from torch.nn.utils import clip_grad_value_
from util.run_aguments import get_args, save_args
from gmr import *

# checklist before you run the below code
# 1. MP (mp*.npy)
# 2. Covariance corresponding to the MP file (mp*_GMM.npy)
# 3. cutoff time for 2
# 4. dataset saved path
# 5. trajectory end index -90 (we discard the last 90 frames of each video)
# 6. crop image into your needed shape (see function `train`)
# 7. the prior for tau (tau1, tau2, and tau3 prior)

args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)

# load MP weights
MP1 = torch.tensor(np.load("saved/mp1.npy"), device=device)  # shape: (basis_num, 3)
MP2 = torch.tensor(np.load("saved/mp2.npy"), device=device)  # shape: (basis_num, 3)
MP3 = torch.tensor(np.load("saved/mp3.npy"), device=device)  # shape: (basis_num, 3)

cut_off_mp1 = 16
cut_off_mp2 = 9
cut_off_mp3 = 10

MP1_GMM = torch.load("saved/mp1_GMM.npy")
MP2_GMM = torch.load("saved/mp2_GMM.npy")
MP3_GMM = torch.load("saved/mp3_GMM.npy")
d_in = range(0, 1)
d_out = range(1, 4)

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.double),
])

def get_traj_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data


def classify_gripper_data(gripper_data_array, close_width=0.04, open_width=0.04):
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


def inner_update(stored_weights, model, lr):
    for name, param in model.named_parameters():
        stored_weights[name] = copy.deepcopy(param.data)  # store param data by deep copy
        if not param.grad is None:
            param.data.sub_(param.grad.data * lr)
    return stored_weights


def load_back(stored_weights, model):
    for name, param in model.named_parameters():
        param.data = stored_weights[name]  # restore weights


def train(task, demo, integrator, tau1, tau2, tau3):
    dim = 3
    dt = args.dt
    index = 0

    # imgs_tensor: [3, height, width]
    # xyz_tensor: [frame_length, 3]
    xyz_tensor = torch.tensor(
        get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)

    # trajectory data
    initPosG = copy.deepcopy(xyz_tensor[0])
    trajectory_length = len(xyz_tensor)

    PosG = initPosG
    VelG = torch.zeros_like(PosG, device=device)

    saved_data = {"index": 0, "pos": [], "tau": [], "predict_goal": [], "achieved_goal": [], "MP1": [], "MP2": [], "MP3": [],
                  "cov": [], "loss": []}
    total_loss = 0
    # generate trajectory length until > target trajectory length
    for _ in range(3):

        if _ == 0:
            tau = tau1
            amplified_prob = torch.tensor([1, 0, 0])
            goal = xyz_tensor[int(tau1/0.04)] #+ torch.tensor([0.02, 0.02, 0])
        elif _ == 1:
            tau = tau2
            amplified_prob = torch.tensor([0, 1, 0])
            goal = xyz_tensor[int( (tau1+tau2)/0.04 )]
        else:
            tau = tau3
            amplified_prob = torch.tensor([0, 0, 1])
            goal = xyz_tensor[-1]

        # Synthesis Weights
        synthesis_weights = torch.zeros_like(MP1, device=device).reshape(-1)
        for i in range(3):  # 4 denotes the number of skills
            if i == 0:
                synthesis_weights += MP1.reshape(-1) * amplified_prob[0]
            elif i == 1:
                synthesis_weights += MP2.reshape(-1) * amplified_prob[1]
            else:
                synthesis_weights += MP3.reshape(-1) * amplified_prob[2]
        synthesis_weights = synthesis_weights.reshape(MP1.shape[0], MP1.shape[1])

        k = int(tau.item() / dt) + 1
        index += k
        # generate predicted trajectory by MP
        DMP_pos, DMP_vel, DMP_acc = integrator(PosG, VelG, goal=goal, k=k, dt=dt, weights=synthesis_weights, tau=tau, device=device)

        # TODO: calculate s_list (trajectory length - 1)
        mp1_t_list, mp2_t_list, mp3_t_list = [], [], []
        mp1_scale = cut_off_mp1 / tau.item()
        mp2_scale = cut_off_mp2 / tau.item()
        mp3_scale = cut_off_mp3 / tau.item()
        for i in range(k - 1):
            mp1_t = (i+1) * mp1_scale * dt
            mp2_t = (i+1) * mp2_scale * dt
            mp3_t = (i+1) * mp3_scale * dt
            mp1_t_list.append(mp1_t)
            mp2_t_list.append(mp2_t)
            mp3_t_list.append(mp3_t)

        # TODO: the actual forcing terms for skill 1, 2, 3 are obtained from GMM 1, 2, 3
        # TODO: compress time
        _, skill1_cov_list = gmr(MP1_GMM, mp1_t_list, d_in, d_out, len(mp1_t_list))
        _, skill2_cov_list = gmr(MP2_GMM, mp2_t_list, d_in, d_out, len(mp2_t_list))
        _, skill3_cov_list = gmr(MP3_GMM, mp3_t_list, d_in, d_out, len(mp3_t_list))
        # skill3_cov_list = skill3_cov_list / 4

        pos_cov_list = []
        for i in range(k-1):
            pos_cov = (amplified_prob[0].detach().cpu().data.numpy() * skill1_cov_list[i] +
                       amplified_prob[1].detach().cpu().data.numpy() * skill2_cov_list[i] +
                       amplified_prob[2].detach().cpu().data.numpy() * skill3_cov_list[i])
            pos_cov_list.append(pos_cov)
        pos_cov_list.append(pos_cov_list[-1])
        saved_data["cov"].append( torch.tensor(np.stack(pos_cov_list), device=device) )

        if index < trajectory_length:
            PosG = xyz_tensor[index]
            VelG = torch.zeros_like(PosG)
        # ------------------------------------------------------------------------------------------------------- #

        saved_data["index"] += k
        saved_data["pos"].append(DMP_pos)
        saved_data["tau"].append(tau.item())
        saved_data["predict_goal"].append(goal.cpu().data.numpy())
        saved_data["achieved_goal"].append(DMP_pos[-1].cpu().data.numpy())
        saved_data["MP1"].append(amplified_prob[0].item())
        saved_data["MP2"].append(amplified_prob[1].item())
        saved_data["MP3"].append(amplified_prob[2].item())
        # update index

    # calculate loss
    length = saved_data["index"]
    predict_pos = torch.concat(saved_data["pos"])

    real_targ_pos = torch.zeros([length, dim], device=device)
    real_targ_pos[0:trajectory_length] = copy.deepcopy(xyz_tensor)
    for i in range(length - trajectory_length):
        real_targ_pos[trajectory_length + i] = copy.deepcopy(xyz_tensor[-1])

    covariance = torch.concat(saved_data["cov"])

    skill_len_index = 0
    select_index = [0]
    for i in range(len(saved_data["cov"])):
        skill_len_index += len(saved_data["cov"][i])
        select_index.append(skill_len_index)
    loss_array = np.zeros(len(predict_pos))

    for i in range(len(covariance)):
        loss = (predict_pos[i] - real_targ_pos[i]).reshape(1, -1) @ torch.pinverse(covariance[i]) @ (predict_pos[i] - real_targ_pos[i]).reshape(-1, 1)
        total_loss += loss
        loss_array[i] = loss.item()

    for i in range(len(saved_data["cov"])):
        skill_loss = np.sum(loss_array[select_index[i]:select_index[i+1]])
        saved_data["loss"].append(skill_loss / len(covariance))
        # saved_data["loss"].append(skill_loss / len(saved_data["cov"][i]))


    total_loss = total_loss / len(covariance)

    return total_loss, saved_data, predict_pos


def main():

    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)

    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))

    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # DMP functions
    basisFunc = GaussianBasisFunction(basis_num=args.basis_num, alpha=args.alpha, device=device)
    integrator = DMPIntegrator(kp=args.kp, kv=args.kv, alpha=args.alpha, basisfunc=basisFunc)

    max_eval_loss = 1e6

    # training epoch
    for epoch in range(args.epoch):
        # random sample 1 task, and its corresponding 2 demos (1 for meta-train, 1 for meta-test)
        batch = args.batch
        batch_tasks, batch_demos = train_demos(batch)

        meta_train_loss = 0
        for train_batch in range(batch):
            task, demo = batch_tasks[train_batch], batch_demos[train_batch][0]
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            meta_train_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            meta_train_traj = torch.tensor(
                get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
            ).reshape(-1, 21)[:, 14:17][0:-90].to(device)

            tau1, tau2, tau3 = segment_tau(meta_train_gripper)
            loss, saved_data, _ = train(task, demo, integrator, tau1, tau2, tau3)

            meta_train_loss += loss

        meta_train_loss /= batch

        logs = "epoch: {}, task {}, demo {} meta train BC loss {:.2f}\n".format(epoch, batch_tasks[-1],
                                          batch_demos[-1][0], meta_train_loss.item() )
        print(logs, end="")
        file.write(logs)
        logs = "number of tau {}, tau: {}\n".format(len(saved_data["tau"]), saved_data["tau"])
        print(logs, end="")
        file.write(logs)
        logs = "predict goal {}\n".format(saved_data["predict_goal"])
        print(logs, end="")
        file.write(logs)
        logs = "achieved goal: {}\n".format(saved_data["achieved_goal"])
        print(logs, end="")
        file.write(logs)
        logs = "MP1 prob: {}\n".format(saved_data["MP1"])
        print(logs, end="")
        file.write(logs)
        logs = "MP2 prob: {}\n".format(saved_data["MP2"])
        print(logs, end="")
        file.write(logs)
        logs = "MP3 prob: {}\n".format(saved_data["MP3"])
        print(logs, end="")
        file.write(logs)
        logs = "each part loss: {}\n\n".format(saved_data["loss"])
        print(logs, end="")
        file.write(logs)
        file.flush()

        # after execuate one inner update, the wights in NN model become \theta'
        # use \theta' to calculate the outer loop loss

        meta_test_loss = 0
        for test_batch in range(batch):
            task, demo = batch_tasks[test_batch], batch_demos[test_batch][1]

            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            meta_test_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            meta_test_traj = torch.tensor(
                get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
            ).reshape(-1, 21)[:, 14:17][0:-90].to(device)

            tau1, tau2, tau3 = segment_tau(meta_test_gripper)
            loss, saved_data, _ = train(task, demo, integrator, tau1, tau2, tau3)

            meta_test_loss += loss

        meta_test_loss /= batch

        logs = "epoch: {}, task {}, demo {} meta test task BC loss {:.2f}\n".format(epoch, batch_tasks[-1],
                                                          batch_demos[-1][1], meta_test_loss.item())
        print(logs, end="")
        file.write(logs)
        logs = "number of tau {}, tau: {}\n".format(len(saved_data["tau"]), saved_data["tau"])
        print(logs, end="")
        file.write(logs)
        logs = "predict goal {}\n".format(saved_data["predict_goal"])
        print(logs, end="")
        file.write(logs)
        logs = "achieved goal: {}\n".format(saved_data["achieved_goal"])
        print(logs, end="")
        file.write(logs)
        logs = "MP1 prob: {}\n".format(saved_data["MP1"])
        print(logs, end="")
        file.write(logs)
        logs = "MP2 prob: {}\n".format(saved_data["MP2"])
        print(logs, end="")
        file.write(logs)
        logs = "MP3 prob: {}\n".format(saved_data["MP3"])
        print(logs, end="")
        file.write(logs)
        logs = "each part loss: {}\n\n".format(saved_data["loss"])
        print(logs, end="")
        file.write(logs)
        file.flush()

        torch.save(meta_train_traj, os.path.join(saved_folder_name, "meta_train_traj"))
        torch.save(saved_data["pos"], os.path.join(saved_folder_name, "predict_traj"))
        torch.save(meta_test_traj, os.path.join(saved_folder_name, "meta_test_traj"))


setRandomSeed(args.seed)
main()