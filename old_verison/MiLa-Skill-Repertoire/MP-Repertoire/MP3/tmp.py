import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits import mplot3d
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


dt = 0.0333    # time interval
dim = 3     # 2D trajectory

# skill 3
traj_data = np.array(get_data("traj.txt")).reshape(-1, 21)
traj_data = traj_data[:-90]
traj_data = traj_data[int(17.24/0.04):]

joint_pos = traj_data[:, 0:7]
joint_vel = traj_data[:, 7:14]
ee_pos = traj_data[:, 14:17]
quat = traj_data[:, 17:]

results = []
for i in range(len(traj_data)):
    for j in range(7):  # joint angles
        results.append(joint_pos[i, j])

    for j in range(7):  # joint velocities
        results.append(joint_vel[i, j])

    for j in range(3):  # 3 * N	DMP position
        results.append(ee_pos[i, j])

    for j in range(4):  # 4 * N 	Quat
        results.append(quat[i, j])


with open("mp3_traj.txt", "w") as file:
    for tmp in results:
        file.write("{} ".format(tmp))


traj1 = np.array(get_data("traj.txt")).reshape(-1, 21)
traj2 = np.array(get_data("mp3_traj.txt")).reshape(-1, 21)
print(1)