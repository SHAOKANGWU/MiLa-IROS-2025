import sys
sys.path.append("../../")
import os
import copy
from PIL import Image
import torch
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from util import setRandomSeed, GaussianBasisFunction
from torch.nn.utils import clip_grad_value_
from util.run_aguments import save_args
from torchvision.transforms import v2
import random
from util.gmr import *
import learn2learn as l2l
import argparse
from Model import GCSL_Model


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

    #################### meta outer update learnign rate ####################
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for the BC policy")

    args = parser.parse_args()
    return args

args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(3)
device = torch.device("cpu")


def main():
    model = GCSL_Model(output_dim=3).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    state_dict = torch.load("saved/GCSL/model_eval.pth")
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["opt"])

    # data1 = torch.randn([1, 3, 140, 170])
    # data2 = torch.randn([1, 3, 140, 170])
    # data3 = torch.randn([1, 3])
    # prediction = model(data1, data2, data3)

    traced_script_module = torch.jit.script(model)
    traced_script_module.save("GCSL-model.pt")

main()
