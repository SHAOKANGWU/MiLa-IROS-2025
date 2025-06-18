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

def plot_training_data_gripper_width():

    tasks = [i for i in range(1, 61, 1)]

    for task in tasks:
        for demo in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            gripper_data = np.array(get_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))[0:-90]
            length = gripper_data.shape[0]

            # plot gripper data
            tt = np.linspace(0, length * 0.04, length)
            plt.plot(tt, gripper_data)
            plt.savefig("saved/gripper_width_figure/task-{}-demo-{}.png".format(task, demo))
            plt.close()
            print("save figure task-{}-demo-{}.png".format(task, demo))


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

    tasks = [1, 2, 3, 4, 5]

    for task in [53]:
        for demo in [6, 7, 8, 9, 10]:
            traj_data = np.array(get_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))).reshape(-1, 21)
            # traj_data = np.array(get_data("/home/robot-learn/Downloads/cov_skill1/traj{}.txt".format(demo))).reshape(-1, 21)
            end_index = -90
            traj_data = traj_data[:end_index, 14:17]
            # plot 3D trajectory data
            ax.plot3D(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], label="demonstration")
            print("target traj end point {}".format(traj_data[-1]))
            # # plot time-trajectory data
            # length = traj_data.shape[0]
            # tt = np.linspace(0, 0.04 * length, length)
            # plt.plot(tt, traj_data[:, 0], label="t-px")
            # plt.plot(tt, traj_data[:, 1], label="t-py")
            # plt.plot(tt, traj_data[:, 2], label="t-pz")
            # plt.legend()
            # plt.show()
        # plt.legend()

    # traj_data = np.array(get_data("meta-train/task1/demo1/data/traj.txt".format(demo))).reshape(-1, 21)[:, 14:17]
    # ax.plot3D(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], label="demonstration")

    plt.legend()
    plt.show()

def get_training_data_trajectory_end_point():

    tasks = [i for i in range(1, 61, 1)]

    end_point_list = []
    for task in tasks:
        for demo in [1, 2, 3, 4, 5]:
            traj_data = np.array(get_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))).reshape(-1, 21)
            end_index = -90
            traj_data = traj_data[:end_index, 14:17]
            end_point = traj_data[-1]
            end_point_list.append(end_point)

    end_point_array = np.array(end_point_list)
    x_mean, x_std = np.mean(end_point_array[:, 0]), np.std(end_point_array[:, 0])
    y_mean, y_std = np.mean(end_point_array[:, 1]), np.std(end_point_array[:, 1])
    z_mean, z_std = np.mean(end_point_array[:, 2]), np.std(end_point_array[:, 2])
    print(1)


# plot_training_data_gripper_width()
# plot_training_data_trajectory()
plot_training_data_trajectory()