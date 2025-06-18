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


test_task = [i for i in range(1, 17, 1)]        # task 1 - 16
test_dataset = {}
for task in test_task:
    # each task has 10 demos, we use 8 for training, 1 for validation and 1 for testing
    if 1 <= task <= 4:
        demos = [i for i in range(1, 7, 1)]
    else:
        demos = [i for i in range(1, 9, 1)]
    test_dataset[task] = demos

def single_skill_traj(path, index):

    # compute success rate here
    data = torch.load(path)
    predict_traj = data["predict_traj"]
    target_traj = data["target_traj"]
    count = 0
    for i in range(len(predict_traj)):
        res = np.max(np.abs(predict_traj[i][-1] - target_traj[i][-1]))
        if res <= 0.08:
            count += 1
    print("success rate {}".format(count/len(predict_traj) * 100))


    # display trajectory here
    data = torch.load(path)
    predict_traj = data["predict_traj"][index]
    target_traj = data["target_traj"][index]

    ax.plot3D(predict_traj[:, 0], predict_traj[:, 1], predict_traj[:, 2], label="predict traj")
    ax.plot3D(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], label="target traj")

    plt.legend()
    plt.show()


def _3skill_traj(path, index):

    # compute success rate here
    data = torch.load(path)
    predict_traj = data["predict_traj"]
    target_traj = data["target_traj"]
    loss = data["loss"]
    count = 0
    satisfied_traj_index = []
    for i in range(len(predict_traj) // 3):
        skill_1 = np.max(np.abs(predict_traj[i*3][-1] - target_traj[i*3][-1]))
        skill_2 = np.max(np.abs(predict_traj[i*3+1][-1] - target_traj[i*3+1][-1]))
        skill_3 = np.max(np.abs(predict_traj[i*3+2][-1] - target_traj[i*3+2][-1]))

        # TODO: filter the satisfied trajectories and show it
        if skill_1 <= 0.03 and skill_2 <= 0.05 and skill_3 <= 0.08:
            count += 1
            satisfied_traj_index.append(i)
    print("success rate {}".format(count/(len(predict_traj) // 3) * 100))

    # index = satisfied_traj_index[index] # 0 to 59
    # for _ in satisfied_traj_index:
    #     print("task {}, demo {}".format(data["task"][_], data["demo"][_]))

    # display trajectory here
    data = torch.load(path)
    skill1, skill2, skill3 = data["predict_traj"][::3], data["predict_traj"][1::3], data["predict_traj"][2::3]
    skill1_target, skill2_target, skill3_target = data["target_traj"][::3], data["target_traj"][1::3], data["target_traj"][2::3]
    target_traj, predict_traj = [], []
    for i in range(len(skill1)):
        res = np.vstack([skill1[i], skill2[i], skill3[i]])
        predict_traj.append(res)

        res = np.vstack([skill1_target[i], skill2_target[i], skill3_target[i]])
        target_traj.append(res)

    # ax.plot3D(predict_traj[index][:, 0], predict_traj[index][:, 1], predict_traj[index][:, 2], label="predict traj")
    ax.plot3D(skill1[index][:, 0], skill1[index][:, 1], skill1[index][:, 2], label="predict skill 1")
    ax.plot3D(skill2[index][:, 0], skill2[index][:, 1], skill2[index][:, 2], label="predict skill 2")
    ax.plot3D(skill3[index][:, 0], skill3[index][:, 1], skill3[index][:, 2], label="predict skill 3")

    print("task {}, demo {}".format(data["task"][index], data["demo"][index]))

    ax.plot3D(target_traj[index][:, 0], target_traj[index][:, 1], target_traj[index][:, 2], label="target traj")

    plt.legend()
    plt.show()



def _3skill_noSge_traj(path, index):

    # compute success rate here
    data = torch.load(path)
    predict_traj = data["predict_traj"]
    target_traj = data["target_traj"]
    loss = data["loss"]

    ax.plot3D(predict_traj[index][0][:, 0], predict_traj[index][0][:, 1], predict_traj[index][0][:, 2], label="skill 1")
    ax.plot3D(predict_traj[index][1][:, 0], predict_traj[index][1][:, 1], predict_traj[index][1][:, 2], label="skill 2")
    ax.plot3D(predict_traj[index][2][:, 0], predict_traj[index][2][:, 1], predict_traj[index][2][:, 2], label="skill 3")

    ax.plot3D(target_traj[index][:, 0], target_traj[index][:, 1], target_traj[index][:, 2], label="target traj")

    plt.legend()
    plt.show()

_3skill_traj("test-iter", args.index)
# _3skill_traj("saved_data/3skills-test", args.index) # 3skills-123 saved_data/3skills-test

