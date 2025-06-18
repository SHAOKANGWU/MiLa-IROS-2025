import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from mpl_toolkits import mplot3d
from matplotlib.pyplot import MultipleLocator
from GMM import GMM

def get_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data


traj_data1 = np.array(get_data("MP.txt")).reshape(-1, 21)[:, 14:17]
traj_data2 = np.array(get_data("MP.txt")).reshape(-1, 21)[:, 14:17]
traj_data3 = np.array(get_data("MP.txt")).reshape(-1, 21)[:, 14:17]

k = 7   # number of Gaussian Component

dt = 0.04    # time interval
dim = 3     # 2D trajectory
kp = 150.0
kv = 25.0
alpha = 5


def extract_force_term_by_GMM(data):
    # data is a trajectory: traj_length, 2
    # s and fs is extracted by DMP equation, s (traj_length), fs (traj_length, 2)

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


def GMM_DMP(traj_data_list):

    trajectory_length = 0

    for traj_data in traj_data_list:
        trajectory_length += len(traj_data)

    mydataset = []  # 1 (dim of s) + 3 (dim of fs)
    for traj_data in traj_data_list:

        s_list, fs_list = extract_force_term_by_GMM( np.transpose(traj_data) )

        # TODO: if s in s_list is too small, then we do not save it
        s_array = np.array(s_list)[np.array(s_list) > 0.05]
        fs_array = np.array(fs_list)[np.array(s_list) > 0.05, :]

        res = np.hstack([s_array.reshape(-1, 1), fs_array])
        mydataset.append(res)

    mydataset = np.vstack(mydataset)

    gmm = GMM(k, random_state=1)
    gmm.from_samples(mydataset)

    # TODO: generate your trajectory
    target_traj = np.transpose(traj_data_list[0])
    iniPos = np.array([target_traj[0, 0], target_traj[1, 0], target_traj[2, 0]])  # set a new start position
    iniVel = np.array([0, 0, 0])  # set a new start velocity
    newGoal = np.array([target_traj[0, -1], target_traj[1, -1], target_traj[2, -1]])  # set a new motion target
    matA = np.diag(newGoal - iniPos)
    matA[-1][-1] = 1

    newTau = 1 * target_traj.shape[1] * dt

    scale_factor = np.zeros_like(newGoal)
    scale_factor[0:2] = (newGoal - iniPos)[0:2]
    scale_factor[-1] = 1

    newLen = int(np.round(newTau / dt))
    DMP_pos = np.zeros((dim, newLen))
    DMP_vel = np.zeros((dim, newLen))
    DMP_acc = np.zeros((dim, newLen))

    fs_predict_mean = []
    fs_predict_cov = []
    pos_cov = []
    s = 1  # phase variable
    for i in range(newLen - 1):
        s = -alpha * s * dt / newTau + s  # iteration in the canonical system
        if i == 0:
            DMP_pos[:, 0] = iniPos
            DMP_vel[:, 0] = iniVel

        indices = np.array([0])
        fs_mean, fs_covariances = gmm.gmr(indices, s)

        fs_predict_mean.append(fs_mean)
        # TODO: the covariance for position is to large, especvia
        fs_predict_cov.append(np.array(fs_covariances))
        pos_cov.append( s ** 2 * matA @ np.array(fs_covariances) @ matA)

        newForce = fs_mean * s * scale_factor
        # Iteration in the 2nd-order transformation system
        DMP_acc[:, i] = (kp * (newGoal - DMP_pos[:, i]) - newTau * kv * DMP_vel[:, i] + newForce) / newTau ** 2
        DMP_vel[:, i + 1] = DMP_vel[:, i] + DMP_acc[:, i] * dt  # update velocity using acceleration
        DMP_pos[:, i + 1] = DMP_pos[:, i] + DMP_vel[:, i] * dt  # update position using velocity

    ax = plt.axes(projection='3d')
    ax.plot3D(DMP_pos[0, :], DMP_pos[1, :], DMP_pos[2, :], label="DMP px-py-pz")
    ax.plot3D(target_traj[0, :], target_traj[1, :], target_traj[2, :], label="Demo px-py-pz")
    ax.plot3D(iniPos[0], iniPos[1], iniPos[2], 'o', ms=10, label="start-point")
    ax.plot3D(newGoal[0], newGoal[1], newGoal[2], 'o', ms=10, label="end-point")
    plt.legend()
    plt.show()


GMM_DMP([traj_data1, traj_data2, traj_data3])

