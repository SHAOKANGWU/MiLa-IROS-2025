
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def get_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data


def plot_training_data_trajectory():

    ax = plt.axes(projection='3d')
    x_scale = MultipleLocator(0.1)
    y_scale = MultipleLocator(0.1)
    z_scale = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_scale)
    ax.yaxis.set_major_locator(y_scale)
    ax.zaxis.set_major_locator(z_scale)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(None, 0.4)

    traj_data = np.array(get_data("meta-train/task{}/demo{}/data/traj.txt".format(1, 1))).reshape(-1, 21)[0:-90, 14:17]
    target = traj_data[0]
    count = 0

    tasks = [i for i in range(1, 61, 1)]

    for task in tasks:
        for demo in [1, 2, 3, 4, 5]:
            traj_data = np.array(get_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))).reshape(-1, 21)
            end_index = -90
            traj_data = traj_data[:end_index, 14:17]
            if np.abs(traj_data[0] - target)[0] > 0.01 or np.abs(traj_data[0] - target)[1] > 0.01 or np.abs(traj_data[0] - target)[2] > 0.01:
                print("not validate trajectory found in task {} demo {}".format(task, demo))
                count += 1
    print("total not validate trajectory count {}".format(count))

plot_training_data_trajectory()
