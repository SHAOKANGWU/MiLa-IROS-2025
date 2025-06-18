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
from Model import CNN, DMPIntegrator
from util import setRandomSeed, GaussianBasisFunction
from torch.nn.utils import clip_grad_value_
from util.run_aguments import get_args, save_args

args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)


def inner_update(stored_weights, model, lr):
    for name, param in model.named_parameters():
        stored_weights[name] = copy.deepcopy(param.data)  # store param data by deep copy
        if not param.grad is None:
            param.data.sub_(param.grad.data * lr)
    return stored_weights


def load_back(stored_weights, model):
    for name, param in model.named_parameters():
        param.data = stored_weights[name]  # restore weights


def segment_skills(tau_data):
    # tau_data is a binary data
    first_skill_end_idx = int(torch.argmax(tau_data))
    second_skill_end_idx = int(torch.argmin(tau_data[first_skill_end_idx:]) + first_skill_end_idx)
    third_skill_end_idx = len(tau_data)
    skills_idx = [0, first_skill_end_idx, second_skill_end_idx, third_skill_end_idx]
    return skills_idx



def train(imgs_tensor, xyz_tensor, goal_model, integrator, tau, MP):
    dt = args.dt
    imgs_tensor = imgs_tensor.to(device)
    xyz_tensor = xyz_tensor.to(device)
    # trajectory data
    initPosG = copy.deepcopy(xyz_tensor[0])
    trajectory_length = len(xyz_tensor)

    PosG = initPosG
    VelG = torch.zeros_like(PosG, device=device)
    index = [0]

    saved_data = {"pos": [], "predict_goal": [], "achieved_goal": []}
    total_loss = 0
    # Here we assume only predict one batch
    goal = goal_model(imgs_tensor[index])[0]

    k = int(tau / dt) + 1
    # generate predicted trajectory by MP
    DMP_pos, DMP_vel, DMP_acc = integrator(PosG, VelG, goal=goal, k=k, dt=dt, weights=MP, tau=tau, device=device)

    saved_data["pos"].append(DMP_pos)
    saved_data["predict_goal"].append(goal.cpu().data.numpy())
    saved_data["achieved_goal"].append(DMP_pos[-1].cpu().data.numpy())
    # update index

    # calculate loss
    predict_pos = DMP_pos
    length = predict_pos.shape[0]
    real_targ_pos = torch.zeros_like(predict_pos, device=device)
    real_targ_pos[0:trajectory_length] = copy.deepcopy(xyz_tensor)
    for i in range(length - trajectory_length):
        real_targ_pos[trajectory_length + i] = copy.deepcopy(xyz_tensor[-1])

    # We only care about the position, instead of velocity
    total_loss += 10000 * torch.mean((predict_pos - real_targ_pos) ** 2)
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

    # training epoch
    for epoch in range(args.epoch):
        # random sample 1 task, and its corresponding 2 demos (1 for meta-train, 1 for meta-test)
        tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        demos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # sample skill
        skills = [0, 1, 2]
        skill = random.sample(skills, 1)[0]

        batch_tasks = []
        batch_demos = []
        batch = args.batch
        for _ in range(batch):
            task = random.sample(tasks, 1)[0]
            demo = random.sample(demos, 2)
            batch_tasks.append(task)
            batch_demos.append(demo)

        meta_train_loss = 0
        for train_batch in range(batch):
            meta_train_imgs = torch.load("saved/20task3skill/task{}Demo{}Img".format(batch_tasks[train_batch],
                                 batch_demos[train_batch][0]))
            meta_train_traj = torch.load("saved/20task3skill/task{}Demo{}Traj".format(batch_tasks[train_batch],
                                  batch_demos[train_batch][0]))
            meta_train_gripper = torch.load("saved/20task3skill/task{}Demo{}Gripper".format(batch_tasks[train_batch],
                                  batch_demos[train_batch][0]))

            skills_idx = segment_skills(meta_train_gripper)
            start_idx, end_idx = skills_idx[skill], skills_idx[skill+1]
            imgs_tensor = meta_train_imgs[start_idx:end_idx]
            xyz_tensor = meta_train_traj[start_idx:end_idx]
            tau = (end_idx - start_idx) * args.dt
            MP = [MP1, MP2, MP3][skill]

            loss, saved_data, _ = train(imgs_tensor, xyz_tensor, goal_model, integrator, tau, MP)
            meta_train_loss += loss

        meta_train_loss /= batch

        logs = "epoch: {}, task {}, demo {} meta train BC loss {:.2f}\n".format(epoch, batch_tasks[-1],
                                          batch_demos[-1][0], meta_train_loss.item() )
        print(logs, end="")
        file.write(logs)
        logs = "predict goal {}\n".format(saved_data["predict_goal"])
        print(logs, end="")
        file.write(logs)
        logs = "achieved goal: {}\n".format(saved_data["achieved_goal"])
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
            meta_test_imgs = torch.load("saved/20task3skill/task{}Demo{}Img".format(batch_tasks[test_batch],
                                    batch_demos[test_batch][1]))
            meta_test_traj = torch.load("saved/20task3skill/task{}Demo{}Traj".format(batch_tasks[test_batch],
                                    batch_demos[test_batch][1]))
            meta_test_gripper = torch.load("saved/20task3skill/task{}Demo{}Gripper".format(batch_tasks[test_batch],
                                    batch_demos[test_batch][1]))

            skills_idx = segment_skills(meta_test_gripper)
            start_idx, end_idx = skills_idx[skill], skills_idx[skill+1]
            imgs_tensor = meta_test_imgs[start_idx:end_idx]
            xyz_tensor = meta_test_traj[start_idx:end_idx]
            tau = (end_idx - start_idx) * args.dt
            MP = [MP1, MP2, MP3][skill]

            loss, saved_data, _ = train(imgs_tensor, xyz_tensor, goal_model, integrator, tau, MP)

            meta_test_loss += loss

        meta_test_loss /= batch

        logs = "epoch: {}, task {}, demo {} meta test task BC loss {:.2f}\n".format(epoch, batch_tasks[-1],
                                                          batch_demos[-1][1], meta_test_loss.item())
        print(logs, end="")
        file.write(logs)
        logs = "predict goal {}\n".format(saved_data["predict_goal"])
        print(logs, end="")
        file.write(logs)
        logs = "achieved goal: {}\n".format(saved_data["achieved_goal"])
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