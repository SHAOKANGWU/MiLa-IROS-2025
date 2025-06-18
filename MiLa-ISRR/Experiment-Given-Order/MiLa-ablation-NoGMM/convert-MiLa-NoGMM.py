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
    parser.add_argument("--inner-goal", type=float, default=0.0001, help="inner update learning rate for goal")
    #########################################################################

    #################### meta outer update learnign rate ####################
    parser.add_argument("--outer-goal", type=float, default=0.001, help="outer update learning rate for goal")


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
device = torch.device("cpu")

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
            0: {"mp": MP1},
            1: {"mp": MP2},
            2: {"mp": MP3}
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
        "/home/robot-learn/Downloads/task{}/demo{}/data/gripper.txt".format(task, demo)))
    inner_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
    tau1, tau2, tau3 = segment_tau(inner_gripper)
    meta_train_traj = torch.tensor(
        get_traj_data(
            "/home/robot-learn/Downloads/task{}/demo{}/data/gripper.txt".format(task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    skill1_traj = meta_train_traj[0:int(tau1/dt)].cpu().data.numpy()
    skill2_traj = meta_train_traj[int(tau1/dt): int((tau1 + tau2)/dt)].cpu().data.numpy()
    skill3_traj = meta_train_traj[int((tau1 + tau2)/dt):].cpu().data.numpy()
    return skill1_traj, skill2_traj, skill3_traj



def predict_goal(task, demo, goal_model, current_frame, down_sample_factor=2, mode="wrong", meta_test=None, goal_img_index=None):
    img_path = "/home/robot-learn/Downloads/task{}/demo{}/img/color/{}.png".format(task, demo, goal_img_index)
    img_tensor = transform(Image.open(img_path)).to(device)
    img_tensor = post_transforms(img_tensor)
    imgs_tensor = img_tensor[:, 55:335, 220:560].unsqueeze(0)
    # down sample the image by 2
    goal_img = imgs_tensor[:, :, ::down_sample_factor, ::down_sample_factor]

    img_path = "/home/robot-learn/Downloads/task{}/demo{}/img/color/{}.png".format(task, demo, current_frame)
    img_tensor = transform(Image.open(img_path)).to(device)
    img_tensor = post_transforms(img_tensor)
    imgs_tensor = img_tensor[:, 55:335, 220:560].unsqueeze(0)
    # down sample the image by 2
    Img = imgs_tensor[:, :, ::down_sample_factor, ::down_sample_factor]
    goal = goal_model(Img, goal_img, meta_test)[0]
    return goal, Img


def get_dmp_traj(initPos, initVel, goal, k, mp, tau):
    DMP_pos, _, _ = integrator(initPos, initVel, goal=goal, k=k, dt=DOWN_SAMPLEING * dt, weights=mp, tau=tau, device=device)
    return DMP_pos


def predict_order(task, demo, goal_model, mp_classifier, mode, meta_test, sample_size):
    # TODO: here we do not have access to the skill order
    meta_train_traj = torch.tensor(
        get_traj_data("/home/robot-learn/Downloads/task{}/demo{}/data/traj.txt".format(task, demo))
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
        get_traj_data("/home/robot-learn/Downloads/task{}/demo{}/data/traj.txt".format(task, demo))
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
    DMP_pos_1 = get_dmp_traj(initPos, initVel, g1, k1, SKILLS[skill_index_1]["mp"], tau1)

    initPos = DMP_pos_1[-1]#.detach() # DMP_pos_1[-1].detach()# target_traj[k1]
    initVel = torch.zeros_like(initPos, device=device)
    DMP_pos_2 = get_dmp_traj(initPos, initVel, g2, k2, SKILLS[skill_index_2]["mp"], tau2)

    initPos = DMP_pos_2[-1]#.detach()  # DMP_pos_2[-1].detach() # target_traj[k1+k2]
    initVel = torch.zeros_like(initPos, device=device)
    DMP_pos_3 = get_dmp_traj(initPos, initVel, g3, k3, SKILLS[skill_index_3]["mp"], tau3)

    # calculate loss
    predict_length = len(DMP_pos_1) + len(DMP_pos_2) + len(DMP_pos_3)

    meta_train_traj = target_traj[0::DOWN_SAMPLEING]
    target_length = len(meta_train_traj)
    predict_pos = torch.concat([DMP_pos_1, DMP_pos_2, DMP_pos_3])

    real_targ_pos = torch.zeros([predict_length, 3], device=device)
    real_targ_pos[0:target_length] = copy.deepcopy(meta_train_traj)
    for i in range(predict_length - target_length):
        real_targ_pos[target_length + i] = copy.deepcopy(meta_train_traj[-1])

    subtask1_loss = F.mse_loss(predict_pos[0:len(DMP_pos_1)],  real_targ_pos[0:len(DMP_pos_1)], reduction="sum")
    subtask2_loss = F.mse_loss(predict_pos[len(DMP_pos_1):len(DMP_pos_1)+len(DMP_pos_2)],  real_targ_pos[len(DMP_pos_1):len(DMP_pos_1)+len(DMP_pos_2)], reduction="sum")
    subtask3_loss = F.mse_loss(predict_pos[-len(DMP_pos_3):],  real_targ_pos[-len(DMP_pos_3):], reduction="sum")

    if skill_index_1 == 0:
        subtask1_loss *= 5

    if skill_index_2 == 0:
        subtask2_loss *= 5

    if skill_index_3 == 0:
        subtask3_loss *= 5

    total_loss = 1000 * (subtask1_loss + subtask2_loss + subtask3_loss) / predict_length

    return total_loss, torch.vstack([img1, img2, img3]), [DMP_pos_1.cpu().data.numpy(), DMP_pos_2.cpu().data.numpy(), DMP_pos_3.cpu().data.numpy()]


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

    # load model
    state_dict = torch.load("saved/MiLa-nonNovel/model_eval.pth")
    goal_model.load_state_dict(state_dict["goal_model"])
    optimizer_goal.load_state_dict(state_dict["goal_opt"])

    adaptation_steps = 1
    optimizer_goal.zero_grad()

    # get paired data (d^tr and d^val) from test set
    # task, demo = -28, -1
    # task, demo = -28, -2
    # task, demo = -28, -3
    # task, demo = -28, -4
    task, demo = -28, -5

    skill_order = [0, 1, 2]

    learner = maml.clone()

    for step in range(adaptation_steps):
        loss1, _, _ = train(task, demo, learner, mode="test", meta_test=False, skill_order=skill_order)
        learner.adapt(loss1)
        state_dict = {"goal_model": learner.module.state_dict()}
        torch.save(state_dict, "meta-update-once-goal-model")

    # load the model and evaluation it
    update_once_model = MetaGoalCondiCNN(output_dim=3).to(device)
    state_dict = torch.load("meta-update-once-goal-model")
    update_once_model.load_state_dict(state_dict["goal_model"])

    traced_script_module = torch.jit.script(update_once_model)
    traced_script_module.save("meta-goal-model.pt")

    loss, _, _ = train(task, demo, learner, mode="test", meta_test=True, skill_order=skill_order)
    print(loss.item())



setRandomSeed(args.seed)
main()