import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from mpl_toolkits import mplot3d
from matplotlib.pyplot import MultipleLocator
from gmr import *
from sklearn import mixture

def get_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data


# TODO: remember to tune the number of k. The number of k for different skills should be the same?
k = 6   # number of Gaussian Component
saved_mp_path = "mp2_GMM.npy"
cut_off_t = 7.85 / 0.04 * 0.0333
save_model = True
dt = 0.0333    # time interval
dim = 3     # 2D trajectory
kp = 150.0
kv = 25.0
alpha = 5

mydict = {"s": [], "cov": []}

# # TODO:  skill 2, k = 6
traj_data1 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj1.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data3 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj3.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data4 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj4.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data5 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj5.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data6 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj6.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data7 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj7.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data8 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj8.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data9 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj9.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_data10 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj10.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
traj_list = [
                traj_data1,
                traj_data3,
                traj_data4,
                # traj_data5,
                # traj_data6,
                # traj_data7,
                traj_data8,
                # traj_data9,
                traj_data10
]

# DELETE 3, 5, 6

# DELETE 1, 5, 6

def extract_force_term_by_GMM(data):
    # data shape: 3, trajectory length
    pos = data  # only use the first demonstration for learning
    tau = dt * pos.shape[1]  # motion duration
    vel = np.gradient(pos, axis=-1) / dt  # velocity
    acc = np.gradient(vel, axis=-1) / dt  # acceleration
    goal = pos[:, -1]  # the end-point of the demonstration

    iterNum = int(np.round(tau / dt))

    # TODO: should share the same initPos and goal
    scale_factor = np.zeros_like(goal)
    scale_factor[0:2] = (pos[:, -1] - pos[:, 0])[0:2]
    scale_factor[-1] = 1

    s_list = []
    fs_list = []
    s = 1  # phase variable
    for i in range(iterNum):
        s = -alpha * s * dt / tau + s  # iteration in the canonical system
        force = tau * tau * acc[:, i] - kp * (goal - pos[:, i]) + tau * kv * vel[:, i]
        force = force / scale_factor / s

        s_list.append(s)
        fs_list.append(force)

    return s_list, fs_list


def extract_s_t_xyz_by_GMM(data):
    # data shape: 3, trajectory length
    pos = data  # only use the first demonstration for learning
    tau = dt * pos.shape[1]  # motion duration

    iterNum = int(np.round(tau / dt))

    s_list = []
    t_list = []
    s = 1  # phase variable
    for i in range(iterNum):
        s = -alpha * s * dt / tau + s  # iteration in the canonical system
        t = (i+1) * dt

        s_list.append(s)
        t_list.append(t)

    return s_list, t_list, copy.deepcopy( np.transpose(pos) )


def GMM_DMP_covariance(traj_data_list):

    trajectory_length = 0

    for traj_data in traj_data_list:
        trajectory_length += len(traj_data)

    mydataset = []  # 1 (dim of s) + 3 (dim of fs)

    for traj_data in traj_data_list:

        s_list, t_list, xyz_array = extract_s_t_xyz_by_GMM( np.transpose(traj_data) )

        s_array = np.array(s_list) # [np.array(s_list) > 0]
        t_array = np.array(t_list) # [np.array(s_list) > 0, :]

        print("tau {}".format(t_list[-1]))
        res = np.hstack([t_array.reshape(-1, 1), xyz_array])
        mydataset.append(res)

    mydataset = np.vstack(mydataset)

    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', reg_covar=1E-5, init_params='random',
                                  random_state=1)
    gmm.fit(mydataset)

    if save_model:
        torch.save(gmm, saved_mp_path)

    d_in = range(0, 1)
    d_out = range(1, 4)

    target_traj = np.transpose(traj_data_list[1])

    # TODO: compress the newTau
    newTau = 1 * target_traj.shape[1] * dt
    scale = cut_off_t / newTau
    newLen = int(np.round(newTau / dt))
    new_dt = scale * dt

    pos_mean = []
    pos_cov = []
    s = 1  # phase variable
    for i in range(newLen - 1):
        s = -alpha * s * new_dt / newTau + s  # iteration in the canonical system
        t = (i+1) * new_dt
        mu, sigma = gmr(gmm, t, d_in, d_out, 1)
        pos_mean.append(mu[0])
        pos_cov.append(np.array(sigma[0]))

    pos_mean_array = np.array(pos_mean)
    pos_cov_array = np.array(pos_cov)
    pos_std_array = np.sqrt(pos_cov_array) * 1000
    ax = plt.axes(projection='3d')

    # 3D trajectory
    x_scale = MultipleLocator(0.1)
    y_scale = MultipleLocator(0.1)
    z_scale = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_scale)
    ax.yaxis.set_major_locator(y_scale)
    ax.zaxis.set_major_locator(z_scale)
    ax.set_xlim(-0.2, 0.8)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.3, 0.7)

    covariance = np.hstack(
                        [pos_cov_array[:, 0, 0].reshape(-1, 1), pos_cov_array[:, 1, 1].reshape(-1, 1), pos_cov_array[:, 2, 2].reshape(-1, 1)]
                    )

    upper_gmr = pos_mean_array + 1 * np.sqrt(covariance)
    lower_gmr = pos_mean_array - 1 * np.sqrt(covariance)

    ax.plot3D(pos_mean_array[:, 0], pos_mean_array[:, 1], pos_mean_array[:, 2], label="GMM+DMP px-py-pz", linewidth=5)
    ax.plot3D(upper_gmr[:, 0], upper_gmr[:, 1], upper_gmr[:, 2], label="upper GMM+DMP px-py-pz", linewidth=3)
    ax.plot3D(lower_gmr[:, 0], lower_gmr[:, 1], lower_gmr[:, 2], label="lower GMM+DMP px-py-pz", linewidth=3)

    for traj_data in traj_data_list:
        ax.plot3D(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], label="Demo px-py-pz")

    ax.plot3D(pos_mean_array[0, 0], pos_mean_array[0, 1], pos_mean_array[0, 2], 'o', ms=10, label="start-point")
    ax.plot3D(pos_mean_array[-1, 0], pos_mean_array[-1, 1], pos_mean_array[-1, 2], 'o', ms=10, label="end-point")
    plt.legend()
    plt.show()

    for traj_data in traj_data_list:
        tt = np.linspace(0, len(traj_data) * dt, len(traj_data))
        plt.plot(tt, traj_data[:, 0], label="x component")
    tt = np.linspace(0, newLen * dt, newLen-1)
    plt.plot(tt, pos_mean_array[:, 0], label="mean x component", linewidth=3)
    plt.legend()
    plt.show()

    for traj_data in traj_data_list:
        tt = np.linspace(0, len(traj_data) * dt, len(traj_data))
        plt.plot(tt, traj_data[:, 1], label="y component")
    tt = np.linspace(0, newLen * dt, newLen-1)
    plt.plot(tt, pos_mean_array[:, 1], label="mean y component", linewidth=3)
    plt.legend()
    plt.show()

    for traj_data in traj_data_list:
        tt = np.linspace(0, len(traj_data) * dt, len(traj_data))
        plt.plot(tt, traj_data[:, 2], label="z component")
    tt = np.linspace(0, newLen * dt, newLen-1)
    plt.plot(tt, pos_mean_array[:, 2], label="mean z component", linewidth=3)
    plt.legend()
    plt.show()


# GMM_GMR_DMP(traj_list)
GMM_DMP_covariance(traj_list)

