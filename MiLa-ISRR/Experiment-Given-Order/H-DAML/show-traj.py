import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0, help="trajectory index")
args = parser.parse_args()


# plot 3D trajectory data
ax = plt.axes(projection='3d')

x_scale = MultipleLocator(0.1)
y_scale = MultipleLocator(0.1)
z_scale = MultipleLocator(0.1)
ax.xaxis.set_major_locator(x_scale)
ax.yaxis.set_major_locator(y_scale)
ax.zaxis.set_major_locator(z_scale)
ax.set_xlim(-0.2, 0.8)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.3, 0.7)



def _3skill_traj(path, index):

    # compute success rate here
    data = torch.load(path)
    predict_traj = data["predict_traj"]
    target_traj = data["target_traj"]

    traj1 = predict_traj[index].cpu().data.numpy()
    traj2 = target_traj[index].cpu().data.numpy()

    ax.plot3D(traj1[:, 0], traj1[:, 1], traj1[:, 2], label="predict traj")
    ax.plot3D(traj2[:, 0], traj2[:, 1], traj2[:, 2], label="target traj")

    plt.legend()
    plt.show()

    tt = np.linspace(0, len(traj1) * 0.0333, len(traj1))
    plt.plot(tt, traj1[:, 0], label="predict px")
    plt.plot(tt, traj1[:, 1], label="predict py")
    plt.plot(tt, traj1[:, 2], label="predict pz")

    plt.plot(tt, traj2[:, 0], label="target px")
    plt.plot(tt, traj2[:, 1], label="target py")
    plt.plot(tt, traj2[:, 2], label="target pz")
    plt.legend()
    plt.show()


_3skill_traj("test-iter", args.index)


