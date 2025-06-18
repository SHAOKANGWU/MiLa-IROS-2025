import os
import copy
import random
from PIL import Image
import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from Model import CNN, DMPIntegrator, MyCNN, MyCNN_2
from util import setRandomSeed, GaussianBasisFunction, sample_demos, train_demos, test_demos
from torch.nn.utils import clip_grad_value_
from util.run_aguments import get_args, save_args

args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)

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


def inner_update(stored_weights, model, lr):
    for name, param in model.named_parameters():
        stored_weights[name] = copy.deepcopy(param.data)  # store param data by deep copy
        if not param.grad is None:
            param.data.sub_(param.grad.data * lr)
    return stored_weights


def load_back(stored_weights, model):
    for name, param in model.named_parameters():
        param.data = stored_weights[name]  # restore weights


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



def train(task, demo, goal_model, saved_data, start_index, end_index):
    meta_train_traj = torch.tensor(
        get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    target = meta_train_traj[end_index]

    img_path = "meta-train/task{}/demo{}/img/{}.png".format(task, demo, start_index)
    img_tensor = transform(Image.open(img_path)).to(device)
    imgs_tensor = img_tensor[:, 50:330, 250:570].unsqueeze(0)
    # down sample the image by 2
    imgs_tensor = imgs_tensor[:, :, ::2, ::2]
    goal = goal_model(imgs_tensor)[0]

    loss = 1000 * F.mse_loss(goal, target)

    saved_data["predict_goal"].append(goal.cpu().data.numpy())
    saved_data["target_goal"].append(target.cpu().data.numpy())
    saved_data["loss"].append(loss.item())
    return loss, saved_data


def main():

    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)

    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))

    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    # trainable models
    goal_model = MyCNN(output_dim=3).to(device)
    # optimizer
    optimizer_goal = Adam(goal_model.parameters(), lr=args.outer_goal)

    max_eval_loss = 1e6

    # training epoch
    for epoch in range(args.epoch):
        # random sample 1 task, and its corresponding 2 demos (1 for meta-train, 1 for meta-test)
        batch = args.batch
        # sample task and demo
        batch_tasks, batch_demos = train_demos(batch)
        meta_train_loss = 0
        # used for training log
        saved_data = {"predict_goal": [], "target_goal": [], "loss": []}
        for train_batch in range(batch):
            task, demo = batch_tasks[train_batch], batch_demos[train_batch][0]
            # segment data
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            meta_train_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            tau1, tau2, tau3 = segment_tau(meta_train_gripper)
            loss1, saved_data = train(task, demo, goal_model, saved_data, start_index=0, end_index=int(tau1/0.04))
            # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1/0.04), end_index=int((tau1+tau2)/0.04))
            # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1+tau2)/0.04), end_index=int( (tau1+tau2+tau3) / 0.04)-1 )
            meta_train_loss = meta_train_loss + loss1 #+ loss2 + loss3

        meta_train_loss /= batch

        logs = "epoch: {}, meta train BC loss {:.2f}\n".format(epoch, meta_train_loss.item())
        print(logs, end="")
        file.write(logs)
        # logs = "predict goal {}\n".format(saved_data["predict_goal"])
        # print(logs, end="")
        # file.write(logs)
        # logs = "target goal: {}\n".format(saved_data["target_goal"])
        # print(logs, end="")
        # file.write(logs)
        # logs = "loss: {}\n\n".format(saved_data["loss"])
        # print(logs, end="")
        # file.write(logs)
        file.flush()

        optimizer_goal.zero_grad()
        meta_train_loss.backward()
        # clip_grad_value_(goal_model.parameters(), 10)

        # store_weights keep the origin NN weights \theta. \theta' = \theta - lr * \theta.grad
        stored_goal_weights = {}
        stored_goal_weights = inner_update(stored_goal_weights, goal_model, lr=args.inner_goal)

        optimizer_goal.zero_grad()

        meta_test_loss = 0
        saved_data = {"predict_goal": [], "target_goal": [], "loss": []}
        for test_batch in range(batch):
            task, demo = batch_tasks[test_batch], batch_demos[test_batch][1]
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            meta_test_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            tau1, tau2, tau3 = segment_tau(meta_test_gripper)
            loss1, saved_data = train(task, demo, goal_model, saved_data, start_index=0, end_index=int(tau1/0.04))
            # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1/0.04), end_index=int((tau1+tau2)/0.04))
            # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1+tau2)/0.04), end_index=int( (tau1+tau2+tau3) / 0.04) -1 )
            meta_test_loss = meta_test_loss + loss1 #+ loss2 + loss3

        meta_test_loss /= batch

        logs = "epoch: {}, meta test BC loss {:.2f}\n\n".format(epoch, meta_test_loss.item())
        print(logs, end="")
        file.write(logs)
        # logs = "predict goal {}\n".format(saved_data["predict_goal"])
        # print(logs, end="")
        # file.write(logs)
        # logs = "target goal: {}\n".format(saved_data["target_goal"])
        # print(logs, end="")
        # file.write(logs)
        # logs = "loss: {}\n\n".format(saved_data["loss"])
        # print(logs, end="")
        # file.write(logs)
        file.flush()

        load_back(stored_goal_weights, goal_model)

        optimizer_goal.zero_grad()

        # save model here
        state_dict = {"goal_model": goal_model.state_dict(), "goal_opt": optimizer_goal.state_dict()}
        torch.save(state_dict, os.path.join(saved_folder_name, "model.pth"))

        meta_test_loss.backward()
        # clip_grad_value_(goal_model.parameters(), 10)
        optimizer_goal.step()

        eval_batch = 50
        if (epoch + 1) % eval_batch == 0:
            # update 50 times
            eval_loss = 0
            for _ in range(eval_batch):
                batch = 1
                batch_tasks, batch_demos = test_demos(batch)
                meta_train_loss = 0
                # used for training log
                saved_data = {"predict_goal": [], "target_goal": [], "loss": []}
                for train_batch in range(batch):
                    task, demo = batch_tasks[train_batch], batch_demos[train_batch][0]
                    # segment data
                    gripper_data = np.array(
                        get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
                    meta_train_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
                    tau1, tau2, tau3 = segment_tau(meta_train_gripper)
                    loss1, saved_data = train(task, demo, goal_model, saved_data, start_index=0,
                                              end_index=int(tau1 / 0.04))
                    # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1 / 0.04),
                    #                           end_index=int((tau1 + tau2) / 0.04))
                    # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1 + tau2) / 0.04),
                    #                           end_index=int((tau1 + tau2 + tau3) / 0.04) - 1)
                    meta_train_loss = meta_train_loss + loss1 #+ loss2 + loss3
                meta_train_loss /= batch

                optimizer_goal.zero_grad()
                meta_train_loss.backward()
                # clip_grad_value_(goal_model.parameters(), 10)

                # store_weights keep the origin NN weights \theta. \theta' = \theta - lr * \theta.grad
                stored_goal_weights = {}
                stored_goal_weights = inner_update(stored_goal_weights, goal_model, lr=args.inner_goal)

                optimizer_goal.zero_grad()

                meta_test_loss = 0
                saved_data = {"predict_goal": [], "target_goal": [], "loss": []}
                for test_batch in range(batch):
                    task, demo = batch_tasks[test_batch], batch_demos[test_batch][1]
                    gripper_data = np.array(
                        get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
                    meta_test_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
                    tau1, tau2, tau3 = segment_tau(meta_test_gripper)
                    loss1, saved_data = train(task, demo, goal_model, saved_data, start_index=0,
                                              end_index=int(tau1 / 0.04))
                    # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1 / 0.04),
                    #                           end_index=int((tau1 + tau2) / 0.04))
                    # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1 + tau2) / 0.04),
                    #                           end_index=int((tau1 + tau2 + tau3) / 0.04) -1 )
                    meta_test_loss = meta_test_loss + loss1 #+ loss2 + loss3
                meta_test_loss /= batch
                eval_loss += meta_test_loss.item()
                load_back(stored_goal_weights, goal_model)
                optimizer_goal.zero_grad()

            eval_loss /= eval_batch
            logs = "############# Evaluation #############\n"
            print(logs, end="")
            file.write(logs)
            logs = "evaluation loss {:.2f}\n\n".format(eval_loss)
            file.write(logs)
            print(logs, end="")
            file.flush()
            # save model if the evaluation loss is the lowest
            if max_eval_loss > eval_loss:
                max_eval_loss = eval_loss
                state_dict = {"goal_model": goal_model.state_dict(), "goal_opt": optimizer_goal.state_dict()}
                torch.save(state_dict, os.path.join(saved_folder_name, "model_eval.pth"))

setRandomSeed(args.seed)
main()