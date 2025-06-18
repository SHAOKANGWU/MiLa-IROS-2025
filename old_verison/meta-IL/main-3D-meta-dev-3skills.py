import os
import copy
import random
import torch
from torch.optim import Adam, SGD
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from Model import CNN, DMPIntegrator
from util import setRandomSeed, GaussianBasisFunction, sample_demos, train_demos, test_demos
from torch.nn.utils import clip_grad_value_
from util.run_aguments import get_args, save_args


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


transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.double),
])

# load MP weights
MP1 = torch.tensor(np.load("saved/mp1.npy"), device=device)  # shape: (basis_num, 3)
MP2 = torch.tensor(np.load("saved/mp2.npy"), device=device)  # shape: (basis_num, 3)
MP3 = torch.tensor(np.load("saved/mp3.npy"), device=device)  # shape: (basis_num, 3)

def get_traj_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data


def inner_update(stored_weights, model, lr):
    for name, param in model.named_parameters():
        stored_weights[name] = copy.deepcopy(param.data)  # store param data by deep copy
        if not param.grad is None:
            param.data.sub_(param.grad.data * lr)
    return stored_weights


def load_back(stored_weights, model):
    for name, param in model.named_parameters():
        param.data = stored_weights[name]  # restore weights


def train(task, demo, goal_model, logits_model, tau_model, integrator, beta):
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

    saved_data = {"index": 0, "pos": [], "tau": [], "predict_goal": [], "achieved_goal": [], "MP1": [], "MP2": [], "MP3": []}
    total_loss = 0
    target_tau = trajectory_length * dt
    # generate trajectory length until > target trajectory length
    while not saved_data["index"] >= trajectory_length:
        # Here we assume only predict one batch
        img_path = "meta-train/task{}/demo{}/img/{}.png".format(task, demo, index)
        img_tensor = transform(Image.open(img_path)).to(device)
        imgs_tensor = img_tensor[:, 50:330, 250:570].unsqueeze(0)
        goal = goal_model(imgs_tensor)[0]
        logits = logits_model(imgs_tensor)[0]
        tau = tau_model(imgs_tensor)[0]

        # ------------------Tau Prior--------------------- #
        tau1 = torch.clip(7.7 + torch.exp(0.5 * tau[0]) * torch.randn(1, device=device)[0], min=5, max=11)
        tau2 = torch.clip(7.4 + torch.exp(0.5 * tau[1]) * torch.randn(1, device=device)[0], min=5, max=11)
        tau3 = torch.clip(10.4 + torch.exp(0.5 * tau[1]) * torch.randn(1, device=device)[0], min=8, max=15)
        # ------------------------------------------------ #

        # # ------------------Tau Prior--------------------- #
        # tau = torch.abs(tau_model(imgs_tensor[index])[0])
        # tau1 = 5 + tau[0] * torch.randn(1, device=device)[0]
        # tau2 = 5 + tau[1] * torch.randn(1, device=device)[0]
        # tau3 = 5 + tau[2] * torch.randn(1, device=device)[0]
        # # ------------------------------------------------ #

        # Skills probability
        amplified_prob = F.softmax(beta * logits, dim=-1)

        # ---------------- Synthesis Tau -------------------- #
        tau = amplified_prob[0] * tau1 + amplified_prob[1] * tau2 + amplified_prob[2] * tau3
        if target_tau - tau - sum(saved_data["tau"]) <= 5:  # TODO: Modify here, a logical error!
            tau = torch.tensor(target_tau - sum(saved_data["tau"]), device=device)
        # --------------------------------------------------- #

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

        # two different assumption for training
        # ---------- 1. Assume the predicted goal is corresponding to our training data ---------- #
        # TODO: compare detach with not detach
        # PosG[i] = DMP_pos[-1].detach()
        # VelG[i] = DMP_vel[-1].detach()
        # ---------------------------------------------------------------------------------------- #
        # ------- 2. At the end of each tau, we sample a new image and robot state from our training data ------- #
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

    # We only care about the position, instead of velocity
    # total_loss += 10000 * torch.mean((predict_pos - real_targ_pos) ** 2)

    loss = (predict_pos - real_targ_pos) ** 2
    # TODO: WE DO NOT KNOW THE TAU HERE, BUT WE ACTUALLY ONLY CONCERNS ABOUT THE START AND END PORTION OF EACH TAU
    current_tau = 0
    select_list = []
    for saved_tau in saved_data["tau"]:
        start_part = [int(current_tau / 0.04), int((current_tau + saved_tau * 0.2) / 0.04)]
        end_part = [int((current_tau + saved_tau * 0.8) / 0.04), int((current_tau + saved_tau) / 0.04)]

        start_index = [_ for _ in range(start_part[0], start_part[1], 1)]
        end_index = [_ for _ in range(end_part[0], end_part[1], 1)]

        select_list = select_list + start_index + end_index

        current_tau += saved_tau

    assert select_list[-1] < len(loss)
    selective_loss = loss[select_list]
    total_loss += 10000 * torch.mean(selective_loss)

    return total_loss, saved_data, predict_pos


def main():

    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)

    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))

    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # trainable models
    tau_model = CNN(output_dim=3).to(device)
    logits_model = CNN(output_dim=3).to(device)
    goal_model = CNN(output_dim=3).to(device)

    # optimizer
    optimizer_tau = Adam(tau_model.parameters(), lr=0.0001)
    optimizer_logits = Adam(logits_model.parameters(), lr=0.0001)
    optimizer_goal = Adam(goal_model.parameters(), lr=0.00005)

    # DMP functions
    basisFunc = GaussianBasisFunction(basis_num=args.basis_num, alpha=args.alpha, device=device)
    integrator = DMPIntegrator(kp=args.kp, kv=args.kv, alpha=args.alpha, basisfunc=basisFunc)

    beta = args.beta
    max_eval_loss = 1e6
    # training epoch
    for epoch in range(args.epoch):
        # random sample 1 task, and its corresponding 2 demos (1 for meta-train, 1 for meta-test)
        batch = args.batch
        batch_tasks, batch_demos = train_demos(batch)

        meta_train_loss = 0
        for train_batch in range(batch):
            task, demo = batch_tasks[train_batch], batch_demos[train_batch][0]
            loss, saved_data, _ = train(task, demo, goal_model, logits_model, tau_model, integrator, beta=beta)
            meta_train_traj = torch.tensor(
                get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
            ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
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
        optimizer_tau.zero_grad()
        optimizer_logits.zero_grad()

        meta_train_loss.backward()

        clip_grad_value_(tau_model.parameters(), 10)
        clip_grad_value_(logits_model.parameters(), 10)
        clip_grad_value_(goal_model.parameters(), 10)

        # store_weights keep the origin NN weights \theta. \theta' = \theta - lr * \theta.grad
        stored_goal_weights, stored_logits_weights, stored_tau_weights = {}, {}, {}
        stored_goal_weights = inner_update(stored_goal_weights, goal_model, lr=0.00005)
        stored_logits_weights = inner_update(stored_logits_weights, logits_model, lr=0.00001)
        stored_tau_weights = inner_update(stored_tau_weights, tau_model, lr=0.0001)

        # after execuate one inner update, the wights in NN model become \theta'
        # use \theta' to calculate the outer loop loss
        optimizer_goal.zero_grad()
        optimizer_tau.zero_grad()
        optimizer_logits.zero_grad()

        meta_test_loss = 0
        for test_batch in range(batch):
            task, demo = batch_tasks[test_batch], batch_demos[test_batch][1]
            loss, saved_data, predict_pos = train(task, demo, goal_model, logits_model, tau_model, integrator, beta=beta)
            meta_test_traj = torch.tensor(
                get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
            ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
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
        load_back(stored_logits_weights, logits_model)
        load_back(stored_tau_weights, tau_model)

        optimizer_goal.zero_grad()
        optimizer_tau.zero_grad()
        optimizer_logits.zero_grad()

        # save model here
        state_dict = {"goal_model": goal_model.state_dict(), "goal_opt": optimizer_goal.state_dict(),
                      "tau_model": tau_model.state_dict(), "tau_opt": optimizer_tau.state_dict(),
                      "logits_model": logits_model.state_dict(), "logits_opt": optimizer_logits.state_dict()}
        torch.save(state_dict, os.path.join(saved_folder_name, "model.pth"))
        torch.save(meta_train_traj, os.path.join(saved_folder_name, "meta_train_traj"))
        torch.save(saved_data["pos"], os.path.join(saved_folder_name, "predict_traj"))
        torch.save(meta_test_traj, os.path.join(saved_folder_name, "meta_test_traj"))

        meta_test_loss.backward()

        clip_grad_value_(tau_model.parameters(), 10)
        clip_grad_value_(logits_model.parameters(), 10)
        clip_grad_value_(goal_model.parameters(), 10)

        optimizer_goal.step()
        optimizer_tau.step()
        optimizer_logits.step()

        eval_batch = 50
        if (epoch + 1) % eval_batch == 0:
            # update 50 times
            eval_loss = 0
            for _ in range(eval_batch):
                batch = 1
                batch_tasks, batch_demos = test_demos(batch)

                meta_train_loss = 0
                for train_batch in range(batch):
                    task, demo = batch_tasks[train_batch], batch_demos[train_batch][0]
                    loss, saved_data, _ = train(task, demo, goal_model, logits_model, tau_model, integrator, beta=beta)
                    meta_train_traj = torch.tensor(
                        get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
                    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
                    meta_train_loss += loss

                meta_train_loss /= batch

                optimizer_goal.zero_grad()
                optimizer_tau.zero_grad()
                optimizer_logits.zero_grad()

                meta_train_loss.backward()

                clip_grad_value_(tau_model.parameters(), 10)
                clip_grad_value_(logits_model.parameters(), 10)
                clip_grad_value_(goal_model.parameters(), 10)

                # store_weights keep the origin NN weights \theta. \theta' = \theta - lr * \theta.grad
                stored_goal_weights, stored_logits_weights, stored_tau_weights = {}, {}, {}
                stored_goal_weights = inner_update(stored_goal_weights, goal_model, lr=0.00005)
                stored_logits_weights = inner_update(stored_logits_weights, logits_model, lr=0.00001)
                stored_tau_weights = inner_update(stored_tau_weights, tau_model, lr=0.0001)

                # after execuate one inner update, the wights in NN model become \theta'
                # use \theta' to calculate the outer loop loss
                optimizer_goal.zero_grad()
                optimizer_tau.zero_grad()
                optimizer_logits.zero_grad()

                meta_test_loss = 0
                for test_batch in range(batch):
                    task, demo = batch_tasks[test_batch], batch_demos[test_batch][1]
                    loss, saved_data, predict_pos = train(task, demo, goal_model, logits_model, tau_model, integrator, beta=beta)
                    meta_test_traj = torch.tensor(
                        get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
                    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
                    meta_test_loss += loss

                meta_test_loss /= batch

                eval_loss += meta_test_loss.item()

                load_back(stored_goal_weights, goal_model)
                load_back(stored_logits_weights, logits_model)
                load_back(stored_tau_weights, tau_model)

                optimizer_goal.zero_grad()
                optimizer_tau.zero_grad()
                optimizer_logits.zero_grad()

            eval_loss /= eval_batch
            logs = "############# Evaluation #############\n"
            print(logs, end="")
            file.write(logs)
            logs = "evaluation loss {:.2f}\n\n".format(eval_loss)
            print(logs, end="")
            file.write(logs)
            file.flush()
            # save model if the evaluation loss is the lowest
            if max_eval_loss > eval_loss:
                max_eval_loss = eval_loss
                state_dict = {"goal_model": goal_model.state_dict(), "goal_opt": optimizer_goal.state_dict(),
                              "tau_model": tau_model.state_dict(), "tau_opt": optimizer_tau.state_dict(),
                              "logits_model": logits_model.state_dict(), "logits_opt": optimizer_logits.state_dict()}
                torch.save(state_dict, os.path.join(saved_folder_name, "model_eval.pth"))
                torch.save(meta_train_traj, os.path.join(saved_folder_name, "meta_train_traj_eval"))
                torch.save(saved_data["pos"], os.path.join(saved_folder_name, "predict_traj_eval"))
                torch.save(meta_test_traj, os.path.join(saved_folder_name, "meta_test_traj_eval"))

setRandomSeed(args.seed)
main()