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
from Model import BC_Model
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
    parser.add_argument("--lr", type=float, default=0.001, help="inner update learning rate")
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
device = torch.device("cpu")
DOWN_SAMPLE_FACTOR = args.down_sample_factor
dt = 0.0333


def main():
    saved_folder_name = "saved/{}".format(args.save_name)
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)
    save_args(args, os.path.join(saved_folder_name, "run_args.txt"))
    file = open(os.path.join(saved_folder_name, "training_results.log"), 'w')

    model = BC_Model().to(device)

    state_dict = torch.load("pretrained-model/model_eval.pth")
    model.load_state_dict(state_dict["model"])

    model.train()

    traced_script_module = torch.jit.script(model)
    traced_script_module.save("bc-task3.pt")



setRandomSeed(args.seed)
main()