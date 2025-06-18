import torch
from Model import CNN, DMPIntegrator
from util import setRandomSeed, GaussianBasisFunction, sample_demos, train_demos, test_demos
from util.run_aguments import get_args, save_args
from gmr import *
import pathlib

args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)

# load MP weights
MP1 = torch.tensor(np.load("saved/mp1.npy"), device=device)  # shape: (basis_num, 3)
MP2 = torch.tensor(np.load("saved/mp2.npy"), device=device)  # shape: (basis_num, 3)
MP3 = torch.tensor(np.load("saved/mp3.npy"), device=device)  # shape: (basis_num, 3)

def get_traj_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data

def classify_gripper_data(gripper_data_array, close_width=0.04, open_width=0.04):
    # given the close width and open width of the gripper data, classify it into binary data (0 or 1), where 0 means open, 1 means close
    open_index, close_index = -1, -1
    gripper_data_status = np.zeros_like(gripper_data_array)
    for index, data in enumerate(gripper_data_array):
        if data <= close_width:
            close_index = index
            break

    reverse_gripper_data_array = gripper_data_array[::-1]
    for index, data in enumerate(reverse_gripper_data_array):
        if data <= open_width:
            open_index = len(gripper_data_array) - index
            break

    gripper_data_status[close_index:open_index] = 1
    return gripper_data_status


def segment_tau(tau_data):
    # tau_data is a binary data
    dt = args.dt
    first_skill_end_idx = int(torch.argmax(tau_data))
    second_skill_end_idx = int(torch.argmin(tau_data[first_skill_end_idx:]) + first_skill_end_idx)
    third_skill_end_idx = len(tau_data)
    tau1 = dt * first_skill_end_idx
    tau2 = dt * second_skill_end_idx - tau1
    tau3 = dt * third_skill_end_idx - tau2 - tau1
    return torch.tensor(tau1, device=device), torch.tensor(tau2, device=device), torch.tensor(tau3, device=device)


def main():

    # DMP functions
    basisFunc = GaussianBasisFunction(basis_num=args.basis_num, alpha=args.alpha, device=device)
    integrator = DMPIntegrator(kp=args.kp, kv=args.kv, alpha=args.alpha, basisfunc=basisFunc)

    # training epoch
    for task in range(1, 61, 1):
        for demo in range(1, 11, 1):
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            meta_train_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            meta_train_traj = torch.tensor(
                get_traj_data("meta-train/task{}/demo{}/data/traj.txt".format(task, demo))
            ).reshape(-1, 21)[:, 14:17][0:-90].to(device)

            tau1, tau2, tau3 = segment_tau(meta_train_gripper)
            pos1, pos2, pos3 = meta_train_traj[int(0/0.04)], meta_train_traj[int(tau1/0.04)], meta_train_traj[int((tau1+tau2)/0.04)]
            goal1, goal2, goal3 = meta_train_traj[int(tau1/0.04)], meta_train_traj[int((tau1+tau2)/0.04)], meta_train_traj[-1]
            vel1, vel2, vel3 = torch.zeros_like(pos1), torch.zeros_like(pos2), torch.zeros_like(pos3)
            k1, k2, k3 = int(tau1 / 0.04), int(tau2 / 0.04), int(tau3 / 0.04)
            traj1_pos, traj1_vel, traj1_acc = integrator(pos1, vel1, goal=goal1, k=k1, dt=0.04, weights=MP1, tau=tau1, device=device)
            traj2_pos, traj2_vel, traj2_acc = integrator(pos2, vel2, goal=goal2, k=k2, dt=0.04, weights=MP2, tau=tau2, device=device)
            traj3_pos, traj3_vel, traj3_acc = integrator(pos3, vel3, goal=goal3, k=k3, dt=0.04, weights=MP3, tau=tau3, device=device)

            traj = torch.concat([traj1_pos, traj2_pos, traj3_pos])
            traj = traj.cpu().data.numpy()
            tau = np.array([tau1.cpu().data.numpy(), tau2.cpu().data.numpy(), tau3.cpu().data.numpy()])
            traj_path = "traj/task{}/demo{}/data/traj".format(task, demo)
            tau_path = "traj/task{}/demo{}/data/tau".format(task, demo)
            pathlib.Path("traj/task{}/demo{}/data/".format(task, demo)).mkdir(parents=True, exist_ok=True)
            np.save(traj_path, traj)
            np.save(tau_path, tau)
            print("task {} demo {} saved".format(task, demo))

setRandomSeed(args.seed)
main()