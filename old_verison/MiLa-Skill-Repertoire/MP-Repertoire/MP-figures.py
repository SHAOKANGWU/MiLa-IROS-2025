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

# 3D trajectory
def get_3D_axis(azim=-7, evel=20):
    ax = plt.axes(projection='3d')
    x_scale = MultipleLocator(0.2)
    y_scale = MultipleLocator(0.2)
    z_scale = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_scale)
    ax.yaxis.set_major_locator(y_scale)
    ax.zaxis.set_major_locator(z_scale)

    ax.tick_params(width=5, labelsize=20)
    ax.view_init(azim=azim, elev=evel)     # adjust viewpoint

    ax.set_xlim([0.1, 0.7])
    ax.set_ylim([-0.4, 0.5])
    ax.set_zlim(0.1, 0.6)
    return ax

dt = 0.0333    # time interval
dim = 3     # 2D trajectory


# skill 1
reach_data = np.array(get_data("MP1/traj.txt")).reshape(-1, 21)
reach_data = reach_data[:-90, 14:17]
reach_data = reach_data[0:int(8 / 0.04)]
reach_data = np.transpose(reach_data)

reach_pos = reach_data # only use the first demonstration for learning
reach_iniPos = reach_pos[:, 0]
reach_goal = reach_pos[:, -1] # the end-point of the demonstration


# skill 2
place_data = np.array(get_data("MP2/traj.txt")).reshape(-1, 21)
place_data = place_data[:, 14:17]
place_data = np.transpose(place_data)

place_pos = place_data # only use the first demonstration for learning
place_iniPos = place_pos[:, 0]
place_goal = place_pos[:, -1] # the end-point of the demonstration


# skill 3
push_data = np.array(get_data("MP3/traj.txt")).reshape(-1, 21)
push_data = push_data[:-90, 14:17]
push_data = push_data[int(17.24/0.04):]
push_data = np.transpose(push_data)

push_pos = push_data # only use the first demonstration for learning
push_iniPos = push_pos[:, 0]
push_goal = push_pos[:, -1] # the end-point of the demonstration


# plot reach task
ax = get_3D_axis(-37, 20)
ax.plot3D(reach_pos[0, :], reach_pos[1, :], reach_pos[2, :], label="Reach Demonstration", linewidth=7,
          color=[252 / 255, 85 / 255, 84 / 255]
            # color=[0.6, 0.6, 0.6]
          )
ax.plot3D(reach_iniPos[0], reach_iniPos[1], reach_iniPos[2], 'o', ms=15, label="Start-point", color=[126 / 255, 47 / 255, 142 / 255])
ax.plot3D(reach_goal[0], reach_goal[1], reach_goal[2], 'o', ms=15, label="End-point", color="g")
plt.legend(ncol=2, fontsize=15, loc="upper center")
plt.show()


# plot place task
ax = get_3D_axis(32, 17)
ax.plot3D(place_pos[0, :], place_pos[1, :], place_pos[2, :], label="Place Demonstration", linewidth=7, color="#89ABE3")
ax.plot3D(place_iniPos[0], place_iniPos[1], place_iniPos[2], 'o', ms=15, color=[126 / 255, 47 / 255, 142 / 255])
ax.plot3D(place_goal[0], place_goal[1], place_goal[2], 'o', ms=15, color="g")
plt.legend(ncol=2, fontsize=15, loc="upper center")
plt.show()


# plot push task
ax = get_3D_axis(-35, 21)
ax.plot3D(push_pos[0, :], push_pos[1, :], push_pos[2, :], label="Push Demonstration", linewidth=7, color=[255 / 255, 177 / 255, 50 / 255])
ax.plot3D(push_iniPos[0], push_iniPos[1], push_iniPos[2], 'o', ms=15, color=[126 / 255, 47 / 255, 142 / 255])
ax.plot3D(push_goal[0], push_goal[1], push_goal[2], 'o', ms=15, color="g")
plt.legend(ncol=2, fontsize=15, loc="upper center")
plt.show()


# plot legend
ax = get_3D_axis(32, 17)
# plot reach task
ax.plot3D(reach_pos[0, :], reach_pos[1, :], reach_pos[2, :], label="Reach Demonstration", linewidth=7,
          color=[252 / 255, 85 / 255, 84 / 255]
            # color=[0.6, 0.6, 0.6]
          )
ax.plot3D(place_pos[0, :], place_pos[1, :], place_pos[2, :], label="Place Demonstration", linewidth=7, color="#89ABE3")

# plot push task
ax.plot3D(push_pos[0, :], push_pos[1, :], push_pos[2, :], label="Push Demonstration", linewidth=7, color=[255 / 255, 177 / 255, 50 / 255])

ax.plot3D(reach_iniPos[0], reach_iniPos[1], reach_iniPos[2], 'o', ms=15, label="Start-point", color=[126 / 255, 47 / 255, 142 / 255])
ax.plot3D(reach_goal[0], reach_goal[1], reach_goal[2], 'o', ms=15, label="End-point", color="g")
plt.legend(ncol=5, fontsize=15, loc="upper center")
plt.show()