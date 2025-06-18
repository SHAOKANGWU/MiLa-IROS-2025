import os
import copy
import random
import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from Model import DMPIntegrator
from util import setRandomSeed, GaussianBasisFunction, sample_demos
from torch.nn.utils import clip_grad_value_
from util.run_aguments import get_args, save_args
import torchvision.models as models
from gmr import *

args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)

MP1_GMM = torch.load("saved/skill1_gmm_dict")
MP2_GMM = torch.load("saved/skill2_gmm_dict")
MP3_GMM = torch.load("saved/skill3_gmm_dict")
d_in = range(0, 1)
d_out = range(1, 4)

def find_index(s, s_list):
    results = torch.tensor(s_list, device=device) - s
    results = torch.abs(results)
    index = torch.argmin(results)
    return index


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


def inner_update(stored_weights, model, lr):
    for name, param in model.named_parameters():
        stored_weights[name] = copy.deepcopy(param.data)  # store param data by deep copy
        if not param.grad is None:
            param.data.sub_(param.grad.data * lr)
    return stored_weights


def load_back(stored_weights, model):
    for name, param in model.named_parameters():
        param.data = stored_weights[name]  # restore weights


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


def compt_dmp_pos_cov(tau, prob, s_list, s_fs1_cov_list, s_fs2_cov_list, s_fs3_cov_list, goal, initPos):
    traj_length = len(s_list)
    assert len(s_list) == len(s_fs1_cov_list) == len(s_fs2_cov_list) == len(s_fs3_cov_list)
    assert prob.shape == goal.shape == initPos.shape

    s_fs_cov_list = []
    for i in range(traj_length):
        fs_cov = prob[0] * s_fs1_cov_list[i] + prob[1] * s_fs2_cov_list[i] + prob[2] * s_fs3_cov_list[i]
        s_fs_cov_list.append(fs_cov)

    pos_cov_list = []
    matA = torch.diag(goal-initPos)
    matA[-1][-1] = 1
    for i in range(traj_length):
        # pos_cov = 1 / (tau ** 4) * (s_list[i] ** 2) * (matA @ fs_cov_list[i] @ matA)
        pos_cov = 1 / (tau ** 4) * (matA @ s_fs_cov_list[i] @ matA)
        pos_cov_list.append(pos_cov)
    return pos_cov_list


def train(imgs_tensor, xyz_tensor, goal_model, MP1, MP2, MP3, integrator, beta, tau1, tau2, tau3):
    dim = 3
    dt = args.dt
    # imgs_tensor: [frame_length, 3, height, width]
    # xyz_tensor: [frame_length, 3]
    imgs_tensor = imgs_tensor.to(device)
    xyz_tensor = xyz_tensor.to(device)
    # trajectory data
    initPosG = copy.deepcopy(xyz_tensor[0])
    trajectory_length = len(xyz_tensor)

    PosG = initPosG
    VelG = torch.zeros_like(PosG, device=device)
    index = [0]

    saved_data = {"index": 0, "pos": [], "tau": [], "predict_goal": [], "achieved_goal": [], "MP1": [], "MP2": [], "MP3": [], "cov": []}
    total_loss = 0
    for _ in range(3):
        # Here we assume only predict one batch
        goal = goal_model(imgs_tensor[index])[0]

        if _ == 0:
            tau = tau1
            amplified_prob = torch.tensor([1, 0, 0])
        elif _ == 1:
            tau = tau2
            amplified_prob = torch.tensor([0, 1, 0])
        else:
            tau = tau3
            amplified_prob = torch.tensor([0, 0, 1])


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

        k = int(tau / dt) + 1
        index[0] += k
        # generate predicted trajectory by MP
        DMP_pos, DMP_vel, DMP_acc = integrator(PosG, VelG, goal=goal, k=k, dt=dt, weights=synthesis_weights, tau=tau, device=device)

        # todo: calculate covariance
        pos_cov_list = []
        for i in range(k - 1):
            s = torch.exp(-args.alpha / tau * (i + 1) * dt)
            idx = find_index(s, MP1_GMM["s"])
            cov1 = torch.tensor(MP1_GMM["cov"][idx], device=device)

            idx = find_index(s, MP2_GMM["s"])
            cov2 = torch.tensor(MP2_GMM["cov"][idx], device=device)

            idx = find_index(s, MP3_GMM["s"])
            cov3 = torch.tensor(MP3_GMM["cov"][idx], device=device)

            cov = amplified_prob[0].detach() * cov1 + amplified_prob[1].detach() * cov2 + amplified_prob[
                2].detach() * cov3

            pos_cov_list.append(cov)

        pos_cov_list.append(pos_cov_list[-1])
        saved_data["cov"].append(torch.stack(pos_cov_list))

        # two different assumption for training
        # ---------- 1. Assume the predicted goal is corresponding to our training data ---------- #
        # TODO: compare detach with not detach
        # PosG[i] = DMP_pos[-1].detach()
        # VelG[i] = DMP_vel[-1].detach()
        # ---------------------------------------------------------------------------------------- #
        # ------- 2. At the end of each tau, we sample a new image and robot state from our training data ------- #
        if index[0] < trajectory_length:
            PosG = xyz_tensor[index[0]]
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

    for i in range(len(covariance)):
        loss = (predict_pos[i] - real_targ_pos[i]).reshape(1, -1) @ torch.pinverse(covariance[i]) @ (predict_pos[i] - real_targ_pos[i]).reshape(-1, 1)
        total_loss += loss

    total_loss = total_loss / len(covariance)

    return total_loss, saved_data, predict_pos

def main():

    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)

    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))

    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # trainable models
    goal_model = CNN(output_dim=3).to(device)

    # optimizer
    optimizer_goal = Adam(goal_model.parameters(), lr=0.00005)

    # DMP functions
    basisFunc = GaussianBasisFunction(basis_num=args.basis_num, alpha=args.alpha, device=device)
    integrator = DMPIntegrator(kp=args.kp, kv=args.kv, alpha=args.alpha, basisfunc=basisFunc)

    # load MP weights
    MP1 = torch.tensor(np.load("saved/mp1.npy"), device=device)  # shape: (basis_num, 3)
    MP2 = torch.tensor(np.load("saved/mp2.npy"), device=device)  # shape: (basis_num, 3)
    MP3 = torch.tensor(np.load("saved/mp3.npy"), device=device)  # shape: (basis_num, 3)

    beta = args.beta

    # training epoch
    for epoch in range(args.epoch):
        # random sample 1 task, and its corresponding 2 demos (1 for meta-train, 1 for meta-test)
        batch = args.batch
        batch_tasks, batch_demos = sample_demos(batch)

        meta_train_loss = 0
        for train_batch in range(batch):
            meta_train_imgs = torch.load("saved/5task3skill/task{}Demo{}Img".format(batch_tasks[train_batch],
                                 batch_demos[train_batch][0]))
            meta_train_traj = torch.load("saved/5task3skill/task{}Demo{}Traj".format(batch_tasks[train_batch],
                                  batch_demos[train_batch][0]))
            meta_train_gripper = torch.load("saved/5task3skill/task{}Demo{}Gripper".format(batch_tasks[train_batch],
                                  batch_demos[train_batch][0]))

            tau1, tau2, tau3 = segment_tau(meta_train_gripper)
            loss, saved_data, _ = train(meta_train_imgs, meta_train_traj, goal_model,
                    MP1, MP2, MP3, integrator, beta=beta, tau1=tau1, tau2=tau2, tau3=tau3)
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
        logs = "MP3 prob: {}\n\n".format(saved_data["MP3"])
        print(logs, end="")
        file.write(logs)
        file.flush()

        optimizer_goal.zero_grad()

        meta_train_loss.backward()

        clip_grad_value_(goal_model.parameters(), 10)

        # store_weights keep the origin NN weights \theta. \theta' = \theta - lr * \theta.grad
        stored_goal_weights, stored_logits_weights, stored_tau_weights = {}, {}, {}
        stored_goal_weights = inner_update(stored_goal_weights, goal_model, lr=0.00005)

        # after execuate one inner update, the wights in NN model become \theta'
        # use \theta' to calculate the outer loop loss
        optimizer_goal.zero_grad()

        meta_test_loss = 0
        for test_batch in range(batch):
            meta_test_imgs = torch.load("saved/5task3skill/task{}Demo{}Img".format(batch_tasks[test_batch],
                                    batch_demos[test_batch][1]))
            meta_test_traj = torch.load("saved/5task3skill/task{}Demo{}Traj".format(batch_tasks[test_batch],
                                    batch_demos[test_batch][1]))
            meta_test_gripper = torch.load("saved/5task3skill/task{}Demo{}Gripper".format(batch_tasks[test_batch],
                                    batch_demos[test_batch][1]))
            tau1, tau2, tau3 = segment_tau(meta_test_gripper)

            loss, saved_data, predict_pos = train(meta_test_imgs, meta_test_traj, goal_model,
                                                     MP1, MP2, MP3, integrator, beta=beta, tau1=tau1, tau2=tau2, tau3=tau3)
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
        logs = "MP3 prob: {}\n\n".format(saved_data["MP3"])
        print(logs, end="")
        file.write(logs)
        file.flush()

        load_back(stored_goal_weights, goal_model)

        optimizer_goal.zero_grad()

        # save model here
        state_dict = {"goal_model": goal_model.state_dict(), "goal_opt": optimizer_goal.state_dict()}
        torch.save(state_dict, os.path.join(saved_folder_name, "model.pth"))
        torch.save(meta_train_traj, os.path.join(saved_folder_name, "meta_train_traj"))
        torch.save(saved_data["pos"], os.path.join(saved_folder_name, "predict_traj"))
        torch.save(meta_test_traj, os.path.join(saved_folder_name, "meta_test_traj"))

        meta_test_loss.backward()

        clip_grad_value_(goal_model.parameters(), 10)

        optimizer_goal.step()

setRandomSeed(args.seed)
main()