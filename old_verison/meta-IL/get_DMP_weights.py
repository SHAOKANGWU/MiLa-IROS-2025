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

# skill 1
# traj_data = np.array(get_data("/home/robot-learn/mydataset/meta-train-new/taskmp/demo1/data/traj.txt")).reshape(-1, 21)
# traj_data = traj_data[:-90, 14:17]
# traj_data = traj_data[0:int(8/0.04)]

# # skill 2
# traj_data1 = np.array(get_data("/home/robot-learn/Downloads/skill2.txt")).reshape(-1, 21)
# traj_data = traj_data1[:, 14:17]

# # skill 3
traj_data = np.array(get_data("/home/robot-learn/mydataset/meta-train-new/taskmp/demo1/data/traj.txt")).reshape(-1, 21)
traj_data = traj_data[:-90, 14:17]
traj_data = traj_data[int(17.24/0.04):]

saved_mp_path = "saved/mp3.npy"

mp1_xyz = np.transpose(traj_data)

dt = 0.04    # time interval
dim = 3     # 2D trajectory
pos = mp1_xyz # only use the first demonstration for learning
tau = dt * pos.shape[1] # motion duration
vel = np.gradient(pos, axis=-1)/dt # velocity
acc = np.gradient(vel, axis=-1)/dt # acceleration
goal = pos[:, -1] # the end-point of the demonstration

# TODO: remember to change the basis function number to 15 (40 is to large).
kp = 150.0
kv = 25.0
alpha = 5
basis_num = 8  # number of basis functions
u = np.zeros(basis_num)
c = np.zeros(basis_num) # centers of basis functions
h = np.zeros(basis_num) # widths of basis functions
for i in range(basis_num):
    u[i] = 1.0 / basis_num * i
    c[i] = np.exp(-alpha * u[i])
for i in range(basis_num-1):
    h[i] = 0.5 / ((0.65 * (c[i+1] - c[i]))**2)
h[basis_num-1] = h[basis_num-2]

iterNum = int(np.round(tau / dt))
Phi = np.zeros(basis_num)
Phi_total = np.zeros((iterNum, basis_num))
force = np.zeros((iterNum, dim))


scale_factor = np.zeros_like(goal)
scale_factor[0:2] = (pos[:, -1] - pos[:, 0])[0:2]
scale_factor[-1] = 1 # np.max(pos[2, :]) - np.min(pos[2, :])


s = 1  # phase variable
for i in range(iterNum):
    s = -alpha * s * dt / tau + s # iteration in the canonical system
    addsum = 0
    for b in range(basis_num):
        Phi[b] = np.exp(-h[b] * (s - c[b]) ** 2)
        addsum = addsum + Phi[b]
    Phi = Phi / addsum * s
    Phi_total[i, :] = Phi
    force[i, :] = tau * tau * acc[:, i] - kp * (goal - pos[:, i]) + tau * kv * vel[:, i]
    # todo: DMP mirror
    force[i, :] = force[i, :] / scale_factor


trainPattern = np.matmul(np.linalg.pinv(Phi_total), force)  # weights \vec{w}
np.save(saved_mp_path, trainPattern)

iniPos = np.array([pos[0, 0], pos[1, 0], pos[2, 0]]) # set a new start position
iniVel = np.array([0, 0, 0])                             # set a new start velocity
newGoal = np.array([pos[0, -1], pos[1, -1], pos[2, -1]])  # set a new motion target
newTau = 1 * tau                                      # set a new motion duration

scale_factor[0:2] = (newGoal - iniPos)[0:2]

newLen = int(np.round(newTau / dt))
print(newLen)
DMP_pos = np.zeros((dim, newLen))
DMP_vel = np.zeros((dim, newLen))
DMP_acc = np.zeros((dim, newLen))
s = 1  # phase variable
for i in range(newLen - 1):
    s = -alpha * s * dt / newTau + s  # iteration in the canonical system
    if i == 0:
        DMP_pos[:, 0] = iniPos
        DMP_vel[:, 0] = iniVel

    addsum = 0
    for b in range(basis_num):
        Phi[b] = np.exp(-h[b] * (s - c[b]) ** 2)
        addsum = addsum + Phi[b]
    newForce = np.matmul(Phi / addsum, trainPattern) * s * scale_factor  # todo: DMP mirror
    # Iteration in the 2nd-order transformation system
    DMP_acc[:, i] = (kp * (newGoal - DMP_pos[:, i]) - newTau * kv * DMP_vel[:, i] + newForce) / newTau ** 2
    DMP_pos[:, i + 1] = DMP_pos[:, i] + DMP_vel[:, i] * dt  # update position using velocity
    DMP_vel[:, i + 1] = DMP_vel[:, i] + DMP_acc[:, i] * dt  # update velocity using acceleration

t0 = np.linspace(dt, tau, num=int(np.round(tau/dt)))
t = np.linspace(dt, newTau, num=int(np.round(newTau/dt)))

# position
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t, DMP_pos[0, :], label='DMP px')
ax1.plot(t0, pos[0, :], label='demo px')
ax1.plot(newTau, newGoal[0], 'go')
ax1.set_ylabel("px")
plt.legend()
ax1.grid()

ax2 = plt.subplot(3, 1, 2)
ax2.plot(t, DMP_pos[1, :], label='DMP py')
ax2.plot(t0, pos[1, :], label='demo py')
ax2.plot(newTau, newGoal[1], 'go')
ax2.set_xlabel("t")
ax2.set_ylabel("py")
ax2.grid()

ax3 = plt.subplot(3, 1, 3)
ax3.plot(t, DMP_pos[2, :], label='DMP pz')
ax3.plot(t0, pos[2, :], label='demo pz')
ax3.plot(newTau, newGoal[2], 'go')
ax3.set_xlabel("t")
ax3.set_ylabel("pz")
ax3.grid()

plt.legend()
plt.show()


# velocity
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t, DMP_vel[0, :])
ax1.plot(newTau, 0, 'go')
ax1.set_ylabel("vx")
ax1.grid()

ax2 = plt.subplot(3, 1, 2)
ax2.plot(t, DMP_vel[1, :])
ax2.plot(newTau, 0, 'go')
ax2.set_xlabel("t")
ax2.set_ylabel("vy")
ax2.grid()

ax3 = plt.subplot(3, 1, 3)
ax3.plot(t, DMP_vel[2, :])
ax3.plot(newTau, 0, 'go')
ax3.set_xlabel("t")
ax3.set_ylabel("vz")
ax3.grid()

plt.show()

# 3D trajectory
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
ax.plot3D(DMP_pos[0, :], DMP_pos[1, :], DMP_pos[2, :], label="DMP px-py-pz")
ax.plot3D(pos[0, :], pos[1, :], pos[2, :], label="Demo px-py-pz")
ax.plot3D(iniPos[0], iniPos[1], iniPos[2], 'o', ms=10, label="start-point")
ax.plot3D(newGoal[0], newGoal[1], newGoal[2], 'o', ms=10, label="end-point")
plt.legend()
plt.show()

goalErr = DMP_pos[0:3, -1] - newGoal
velErr = DMP_vel[0:3, -1]
print("DMP target error:", goalErr)
print("DMP velocity error:", velErr)