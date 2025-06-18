import math
import random
import numpy as np
import torch
import pickle


device = torch.device("cpu")
dt = 0.0333

def get_traj_data(path):
    data = []
    with open(path) as file:
        txt = file.readline()
        # the last one is " " (space). discard it
        str_data_list = txt.split(" ")[0:-1]
        for str_data in str_data_list:
            data.append(float(str_data))
    return data

def classify_gripper_data(gripper_data_array, close_width=0.04, open_width=0.05):
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
    first_skill_end_idx = int(torch.argmax(tau_data))
    second_skill_end_idx = int(torch.argmin(tau_data[first_skill_end_idx:]) + first_skill_end_idx)
    third_skill_end_idx = len(tau_data)
    tau1 = dt * first_skill_end_idx
    tau2 = dt * second_skill_end_idx - tau1
    tau3 = dt * third_skill_end_idx - tau2 - tau1
    return torch.tensor(tau1, device=device), torch.tensor(tau2, device=device), torch.tensor(tau3, device=device)


def get_segment_traj(task, demo, mode):
    gripper_data = np.array(get_traj_data(
        "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/gripper.txt".format(mode, task, demo)))
    inner_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
    tau1, tau2, tau3 = segment_tau(inner_gripper)
    meta_train_traj = torch.tensor(
        get_traj_data(
            "/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(mode, task, demo))
    ).reshape(-1, 21)[:, 14:17][0:-90].to(device)
    skill1_traj = meta_train_traj[0:int(tau1/dt)].cpu().data.numpy()
    skill2_traj = meta_train_traj[int(tau1/dt): int((tau1 + tau2)/dt)].cpu().data.numpy()
    skill3_traj = meta_train_traj[int((tau1 + tau2)/dt):].cpu().data.numpy()
    return skill1_traj, skill2_traj, skill3_traj


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

if __name__ == '__main__':
    # 23 degrees
    train_task = [i for i in range(1, 163, 1)]      # task 1 - 162
    validation_task = [i for i in range(1, 19, 1)]  # task 1 - 18

    train_dataset, validation_dataset = {}, {}

    for task in train_task:
        # each task has 10 demos, we use 8 for training, 1 for validation and 1 for testing
        if 1 <= task <= 18:
            demos = [i for i in range(1, 7, 1)]
        else:
            demos = [i for i in range(1, 9, 1)]
        random.shuffle(demos)
        train_dataset[task] = demos

    for task in validation_task:
        # each task has 10 demos, we use 8 for training, 1 for validation and 1 for testing
        if 1 <= task <= 6:
            demos = [i for i in range(1, 7, 1)]
        else:
            demos = [i for i in range(1, 9, 1)]
        random.shuffle(demos)
        validation_dataset[task] = demos

    # Compute here
    radian2degree = 180.0 / math.pi
    angles = []
    for task in train_task:
        for demo in train_dataset[task]:
            task_angle = []
            traj1, traj2, traj3 = get_segment_traj(task, demo, "train")
            start_index, end_index = 0, len(traj1)

            quaterion = torch.tensor(get_traj_data("/media/robot-learn/Shaokang_Wu/LargeDataset/{}/task{}/demo{}/data/traj.txt".format(
                "train", task, demo))).reshape(-1, 21)[:, 17:][0:-90]  # read the quaterion in the whole trajectory

            quaterion = quaterion[start_index:end_index].tolist()

            for quat in quaterion:
                roll_x, pitch_y, yaw_z = euler_from_quaternion(quat[1], quat[2], quat[3], quat[0])
                task_angle.append([radian2degree * roll_x, radian2degree * pitch_y, radian2degree * yaw_z])

            angles.append(task_angle)

        if task % 10 == 0:
            print("complete {}/{}".format(task, len(train_task)))

    with open("angles", "wb") as fp:  # Pickling
        pickle.dump(angles, fp)
