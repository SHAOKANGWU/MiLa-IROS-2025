import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


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

target_data = torch.load("saved/eval/meta_train_traj").cpu().data.numpy()
ax.plot3D(target_data[:, 0], target_data[:, 1], target_data[:, 2], label="meta train traj")

predict_data = torch.load("saved/eval/predict_traj")
for pos in predict_data:
    pos = pos.cpu().data.numpy()
    ax.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], label="prediction")
print("predict traj end point {}".format(pos[-1]))

target_data = torch.load("saved/eval/meta_test_traj").cpu().data.numpy()
ax.plot3D(target_data[:, 0], target_data[:, 1], target_data[:, 2], label="meta test traj")
print("target traj end point {}".format(target_data[-1]))


# target_data = torch.load("demonstration").cpu().data.numpy()
# target_data = target_data[0]
# print("target traj end point {}".format(target_data[-1]))
# ax.plot3D(target_data[:, 0], target_data[:, 1], target_data[:, 2], label="demonstration")
#
# target_data = torch.load("reproduction").cpu().data.numpy()
# ax.plot3D(target_data[:, 0], target_data[:, 1], target_data[:, 2], label="reproduction")


plt.legend()
plt.show()