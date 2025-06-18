import copy
import numpy as np
import matplotlib.pyplot as plt
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

# SKILL 1 DEMONSTRATIONS
gmm1_data1 = np.array(get_data("/home/robot-learn/Downloads/cov_skill1/traj2.txt")).reshape(-1, 21)[int(1/0.04):int(18/0.04), 14:17]
gmm1_data2 = np.array(get_data("/home/robot-learn/Downloads/history/cov_skill1/traj1.txt")).reshape(-1, 21)[int(0/0.04):int(17/0.04), 14:17]
gmm1_data3 = np.array(get_data("/home/robot-learn/Downloads/history/cov_skill1/traj2.txt")).reshape(-1, 21)[int(0/0.04):int(17/0.04), 14:17]
gmm1_data4 = np.array(get_data("/home/robot-learn/Downloads/history/cov_skill1/traj3.txt")).reshape(-1, 21)[int(0/0.04):int(17/0.04), 14:17]
gmm1_data5 = np.array(get_data("/home/robot-learn/Downloads/history/cov_skill1/traj5.txt")).reshape(-1, 21)[int(0/0.04):int(17/0.04), 14:17]
gmm1_list = [gmm1_data1, gmm1_data2, gmm1_data3, gmm1_data4, gmm1_data5]

# SKILL 2 DEMONSTRATIONS
gmm2_data1 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj1.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
gmm2_data3 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj3.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
gmm2_data4 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj4.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
gmm2_data8 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj8.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
gmm2_data10 = np.array(get_data("/home/robot-learn/Downloads/cov_skill2-Jan8/traj10.txt")).reshape(-1, 21)[int(0.7/0.04):int(8.55/0.04), 14:17]
gmm2_list = [gmm2_data1, gmm2_data3, gmm2_data4, gmm2_data8, gmm2_data10]

# SKILL 3 DEMONSTRATIONS
gmm3_data1 = np.array(get_data("/home/robot-learn/Downloads/cov_skill3-Jan8/traj1.txt")).reshape(-1, 21)[int(1/0.04):int(10.5/0.04), 14:17]
gmm3_data2 = np.array(get_data("/home/robot-learn/Downloads/cov_skill3-Jan8/traj2.txt")).reshape(-1, 21)[int(1/0.04):int(10.5/0.04), 14:17]
gmm3_data3 = np.array(get_data("/home/robot-learn/Downloads/cov_skill3-Jan8/traj3.txt")).reshape(-1, 21)[int(1/0.04):int(10.5/0.04), 14:17]
gmm3_data5 = np.array(get_data("/home/robot-learn/Downloads/cov_skill3-Jan8/traj5.txt")).reshape(-1, 21)[int(1/0.04):int(10.5/0.04), 14:17]
gmm3_data6 = np.array(get_data("/home/robot-learn/Downloads/cov_skill3-Jan8/traj6.txt")).reshape(-1, 21)[int(1/0.04):int(10.5/0.04), 14:17]
gmm3_list = [gmm3_data1, gmm3_data2, gmm3_data3, gmm3_data5, gmm3_data6]

def plot_gmm_figure(traj_list, azim=-7, evel=10, color=[252 / 255, 85 / 255, 84 / 255]):
    ax = plt.axes(projection='3d')
    x_scale = MultipleLocator(0.2)
    y_scale = MultipleLocator(0.2)
    z_scale = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_scale)
    ax.yaxis.set_major_locator(y_scale)
    ax.zaxis.set_major_locator(z_scale)

    ax.tick_params(width=5, labelsize=20)
    ax.view_init(azim=azim, elev=evel)  # adjust viewpoint
    ax.set_xlim([0.1, 0.85])
    ax.set_ylim([-0.4, 0.5])
    ax.set_zlim(0.1, 0.6)
    for traj in traj_list:
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=3, linestyle='dotted')
    plt.show()

def plot_legend(traj1, traj2, traj3, color1, color2, color3):
    ax = plt.axes(projection='3d')
    x_scale = MultipleLocator(0.2)
    y_scale = MultipleLocator(0.2)
    z_scale = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_scale)
    ax.yaxis.set_major_locator(y_scale)
    ax.zaxis.set_major_locator(z_scale)

    ax.tick_params(width=5, labelsize=20)
    ax.view_init(azim=1, elev=1)  # adjust viewpoint
    ax.set_xlim([0.1, 0.85])
    ax.set_ylim([-0.4, 0.5])
    ax.set_zlim(0.1, 0.6)
    ax.plot3D(traj1[:, 0], traj1[:, 1], traj1[:, 2], color=color1, linewidth=3, linestyle='dotted', label='Reach Demonstrations')
    ax.plot3D(traj2[:, 0], traj2[:, 1], traj2[:, 2], color=color2, linewidth=3, linestyle='dotted', label='Place Demonstrations')
    ax.plot3D(traj3[:, 0], traj3[:, 1], traj3[:, 2], color=color3, linewidth=3, linestyle='dotted', label='Push Demonstrations')
    plt.legend(ncol=3, fontsize=15, loc="upper center")
    plt.show()


plot_gmm_figure(gmm1_list, azim=-37, evel=20, color=[252 / 255, 85 / 255, 84 / 255])
plot_gmm_figure(gmm2_list, azim=32, evel=17, color="#89ABE3")
plot_gmm_figure(gmm3_list, azim=-35, evel=21, color=[255 / 255, 177 / 255, 50 / 255])

plot_legend(gmm1_list[0], gmm2_list[0], gmm3_list[0], [252 / 255, 85 / 255, 84 / 255], "#89ABE3", [255 / 255, 177 / 255, 50 / 255])