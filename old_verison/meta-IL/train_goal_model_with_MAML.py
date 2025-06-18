import os
import copy
import random
from PIL import Image
import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import learn2learn as l2l
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import transforms
from Model import CNN, DMPIntegrator, MyCNN, MyCNN_2, MyCNN_3
from util import setRandomSeed, GaussianBasisFunction, sample_demos, train_demos, valid_demos, test_demos
from torch.nn.utils import clip_grad_value_
from util.run_aguments import get_args, save_args
from torchvision.transforms import v2

args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)

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
    img_tensor = post_transforms(img_tensor)
    imgs_tensor = img_tensor[:, 50:330, 250:570].unsqueeze(0)
    # down sample the image by 2
    imgs_tensor = imgs_tensor[:, :, ::2, ::2]

    goal = goal_model(imgs_tensor)[0]

    # loss = 1000 * F.mse_loss(goal, target)
    loss = 100 * F.l1_loss(100*goal, 100*target)    # cm

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
    # goal_model = MyCNN_3(output_dim=3).to(device)

    # optimizer
    maml = l2l.algorithms.MAML(goal_model, lr=args.inner_goal, first_order=False)   # inner update learning rate
    optimizer_goal = Adam(maml.parameters(), lr=args.outer_goal)                    # outer update learning rate
    max_eval_loss = 1e6
    adaptation_steps = 1    # TODO: remember to tune the hyperparameters here (we can update the model for many times)
    # training epoch
    for epoch in range(args.epoch):

        optimizer_goal.zero_grad()

        # random sample 1 task, and its corresponding 2 demos (1 for meta-train, 1 for meta-test)
        meta_batch_size = args.batch

        # sample task and demo for meta train and meta validation
        train_batch_tasks, train_batch_demos = train_demos(meta_batch_size)
        valid_batch_tasks, valid_batch_demos = valid_demos(meta_batch_size)

        meta_train_loss, meta_valid_loss = 0, 0
        # inner_loss, outer_loss = 0, 0
        # used for training log
        saved_data = {"predict_goal": [], "target_goal": [], "loss": []}

        for task_idx in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            # training (inner update)
            task, demo = train_batch_tasks[task_idx], train_batch_demos[task_idx][0]
            # segment data
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            inner_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            tau1, tau2, tau3 = segment_tau(inner_gripper)
            for step in range(adaptation_steps):
                loss1, saved_data = train(task, demo, learner, saved_data, start_index=0, end_index=int(tau1 / 0.04))
                # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1/0.04), end_index=int((tau1+tau2)/0.04))
                # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1+tau2)/0.04), end_index=int( (tau1+tau2+tau3) / 0.04)-1 )
                # perform meta inner update
                learner.adapt(loss1)

            # evaluation (used for outer update)
            task, demo = train_batch_tasks[task_idx], train_batch_demos[task_idx][1]
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            outer_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            tau1, tau2, tau3 = segment_tau(outer_gripper)
            loss, saved_data = train(task, demo, learner, saved_data, start_index=0, end_index=int(tau1 / 0.04))
            # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1/0.04), end_index=int((tau1+tau2)/0.04))
            # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1+tau2)/0.04), end_index=int( (tau1+tau2+tau3) / 0.04)-1 )
            # perform meta outer update
            loss.backward()
            meta_train_loss += loss.item()


            # Compute meta-validation loss, where the data is from meta-test dataset
            learner = maml.clone()
            # training (inner update)
            task, demo = valid_batch_tasks[task_idx], valid_batch_demos[task_idx][0]
            # segment data
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            inner_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            tau1, tau2, tau3 = segment_tau(inner_gripper)
            for step in range(adaptation_steps):
                loss1, saved_data = train(task, demo, learner, saved_data, start_index=0, end_index=int(tau1 / 0.04))
                # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1/0.04), end_index=int((tau1+tau2)/0.04))
                # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1+tau2)/0.04), end_index=int( (tau1+tau2+tau3) / 0.04)-1 )
                # perform meta inner update
                learner.adapt(loss1)

            # evaluation (used for outer update)
            task, demo = valid_batch_tasks[task_idx], valid_batch_demos[task_idx][1]
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            outer_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            tau1, tau2, tau3 = segment_tau(outer_gripper)
            loss, saved_data = train(task, demo, learner, saved_data, start_index=0, end_index=int(tau1 / 0.04))
            # loss2, saved_data = train(task, demo, goal_model, saved_data, start_index=int(tau1/0.04), end_index=int((tau1+tau2)/0.04))
            # loss3, saved_data = train(task, demo, goal_model, saved_data, start_index=int((tau1+tau2)/0.04), end_index=int( (tau1+tau2+tau3) / 0.04)-1 )
            meta_valid_loss += loss.item()

        state_dict = {"goal_model": goal_model.state_dict(), "goal_opt": optimizer_goal.state_dict()}
        torch.save(state_dict, os.path.join(saved_folder_name, "model.pth"))

        if max_eval_loss > (meta_valid_loss / meta_batch_size):
            max_eval_loss = (meta_valid_loss / meta_batch_size)
            state_dict = {"goal_model": goal_model.state_dict(), "goal_opt": optimizer_goal.state_dict()}
            torch.save(state_dict, os.path.join(saved_folder_name, "model_eval.pth"))

        logs = '\n'
        print(logs, end="")
        file.write(logs)
        logs = 'epoch {}\n'.format(epoch)
        print(logs, end="")
        file.write(logs)
        logs = 'Meta Train Loss {:.2f}\n'.format(meta_train_loss / meta_batch_size)
        print(logs, end="")
        file.write(logs)
        logs = 'Meta Valid Loss {:.2f}\n'.format(meta_valid_loss / meta_batch_size)
        print(logs, end="")
        file.write(logs)
        file.flush()

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer_goal.step()

setRandomSeed(args.seed)
main()