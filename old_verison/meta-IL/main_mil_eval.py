import copy
import random
import torch
from torch.optim import Adam
import torch.nn.functional as F
from util import setRandomSeed
from torch.nn.utils import clip_grad_value_
import argparse

def save_args(args, path):
    args_dict = args.__dict__
    with open(path, "w") as file:
        for key, value in args_dict.items():
            strings = "{} = {}\n".format(key, value)
            file.write(strings)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50000, help="training epoch")
    parser.add_argument("--seed", type=int, default=1, help="the random seed")
    parser.add_argument("--down-sample-factor", type=int, default=3, choices=[1, 2, 3], help="data down sample factors")
    parser.add_argument("--outer-update-lr", type=float, default=0.0001,
                                                        help="learning rate for meta learning outer loop")
    parser.add_argument("--inner-update-lr", type=float, default=0.00005,
                                                        help="learning rate for meta learning inner loop")
    parser.add_argument("--action-dim", type=int, default=3, help="the model predicted output dim")
    parser.add_argument("--dt", type=float, default=0.04, help="the time interval for collecting the data")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"], help="device")
    parser.add_argument("--clip-grad", type=int, default=10, help="the gradient clip value")

    args = parser.parse_args()
    return args


args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)


def inner_update(stored_weights, model, lr):
    for name, param in model.named_parameters():
        stored_weights[name] = copy.deepcopy(param.data)  # store param data by deep copy
        if not param.grad is None:
            param.data.sub_(param.grad.data * lr)
    return stored_weights


def load_back(stored_weights, model):
    for name, param in model.named_parameters():
        param.data = stored_weights[name]  # restore weights


def down_sample(data, down_sample_factor):
    # data shape: batch * N * M
    down_sample_data = data[::down_sample_factor]
    return down_sample_data


def train(imgs_tensor, xyz_tensor, jntPos_tensor, model, samples, dt, post_update):
    current_image = imgs_tensor[samples].to(device)
    current_jntPos = jntPos_tensor[samples].to(device)
    current_xyz = xyz_tensor[samples].to(device)
    current_configuration = current_xyz # torch.concat([current_jntPos, current_xyz], dim=1)

    next_samples = [idx+1 for idx in samples]
    target_next_xyz = xyz_tensor[next_samples].to(device)

    actions = model(current_image, current_configuration, post_update)
    predicted_next_xyz = actions * dt + current_xyz
    loss = F.l1_loss(predicted_next_xyz, target_next_xyz) * 1e6
    return loss, target_next_xyz



def evaluation(imgs_tensor, xyz_tensor, jntPos_tensor, model, samples, dt, post_update):
    imgs_tensor = imgs_tensor.to(device)
    xyz_tensor = xyz_tensor.to(device)

    robot_config = xyz_tensor[0:1]    # initial state
    predict_traj = []
    predict_traj.append(robot_config)
    for idx in samples:
        image = imgs_tensor[idx:idx+1]
        actions = model(image, robot_config.detach(), post_update)
        next_config = actions * dt + robot_config
        robot_config = next_config
        predict_traj.append(robot_config)

    traj = torch.concat(predict_traj, dim=0)
    target_traj = xyz_tensor
    loss = F.l1_loss(traj, target_traj) * 1e6
    return loss, traj



def main():

    # trainable models
    action_dim = args.action_dim  # end-effector velocity
    configuration_dim = 3     # joint dim + end-effector position
    dt = args.dt * args.down_sample_factor
    model = torch.load("saved/mil/mil_batch1_ds3")

    # optimizer
    optimizer = Adam(model.parameters(), lr=args.outer_update_lr)

    # training epoch
    for epoch in range(args.epoch):
        tasks = [1, 2, 3, 4]   # [1, 2, 3, 4, 5]
        demos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        down_sample_factor = args.down_sample_factor
        # sample tasks and demos
        task = random.sample(tasks, 1)[0]
        demo = random.sample(demos, 2)

        predicted_traj = []
        for skill in range(3):
            meta_train_loss = 0
            # load data
            meta_train_imgs = torch.load("saved/8task3skills/task{}Demo{}Img".format(task, demo[0]))
            meta_train_traj = torch.load("saved/8task3skills/task{}Demo{}Traj".format(task, demo[0]))
            meta_train_jntPos = torch.load("saved/8task3skills/task{}Demo{}JntPos".format(task, demo[0]))
            meta_train_gripper = torch.load("saved/8task3skills/task{}Demo{}Gripper".format(task, demo[0]))

            # down sample the data
            down_sample_imgs = down_sample(meta_train_imgs, down_sample_factor=down_sample_factor)
            down_sample_traj = down_sample(meta_train_traj, down_sample_factor=down_sample_factor)
            down_sample_jntPos = down_sample(meta_train_jntPos, down_sample_factor=down_sample_factor)
            down_sample_gripper = down_sample(meta_train_gripper, down_sample_factor=down_sample_factor)
            assert len(down_sample_gripper) == len(down_sample_traj) == len(down_sample_jntPos) == len(down_sample_imgs)

            # split long horizon skills into 3 basic skills
            first_skill_end_idx = int(torch.argmax(down_sample_gripper))
            second_skill_end_idx = int(torch.argmin(down_sample_gripper[first_skill_end_idx:]) + first_skill_end_idx)
            third_skill_end_idx = len(down_sample_gripper)
            skills_idx = [0, first_skill_end_idx, second_skill_end_idx, third_skill_end_idx]

            skill_start_index, skill_end_index = skills_idx[skill], skills_idx[skill+1]
            imgs = down_sample_imgs[skill_start_index:skill_end_index]
            traj = down_sample_traj[skill_start_index:skill_end_index]
            jntPos = down_sample_jntPos[skill_start_index:skill_end_index]

            traj_length = traj.shape[0]
            # because we do not have the next state of the last frame, so here we need to sample from 0 to trajectory_length - 1
            samples = [i for i in range(traj_length - 1)]
            loss, _ = evaluation(imgs, traj, jntPos, model, samples, dt, post_update=False)
            meta_train_loss += loss

            print("epoch: {}, skill {}, task {}, demo {} meta train BC loss {:.2f}\n".format(epoch, skill, task,
                                                  demo[0], meta_train_loss.item() ))

            optimizer.zero_grad()

            meta_train_loss.backward()

            clip_grad_value_(model.parameters(), args.clip_grad)

            # store_weights keep the origin NN weights \theta. \theta' = \theta - lr * \theta.grad
            stored_weights = {}
            stored_weights = inner_update(stored_weights, model, lr=args.inner_update_lr)

            # after execuate one inner update, the wights in NN model become \theta'
            # use \theta' to calculate the outer loop loss
            optimizer.zero_grad()

            meta_test_loss = 0
            meta_test_imgs = torch.load("saved/8task3skills/task{}Demo{}Img".format(task, demo[1]))
            meta_test_traj = torch.load("saved/8task3skills/task{}Demo{}Traj".format(task, demo[1]))
            meta_test_jntPos = torch.load("saved/8task3skills/task{}Demo{}JntPos".format(task, demo[1]))
            meta_test_gripper = torch.load("saved/8task3skills/task{}Demo{}Gripper".format(task, demo[1]))

            # down sample the data
            down_sample_imgs = down_sample(meta_test_imgs, down_sample_factor=down_sample_factor)
            down_sample_traj = down_sample(meta_test_traj, down_sample_factor=down_sample_factor)
            down_sample_jntPos = down_sample(meta_test_jntPos, down_sample_factor=down_sample_factor)
            down_sample_gripper = down_sample(meta_test_gripper, down_sample_factor=down_sample_factor)
            assert len(down_sample_gripper) == len(down_sample_traj) == len(down_sample_jntPos) == len(down_sample_imgs)

            # split long horizon skills into 3 basic skills
            first_skill_end_idx = int(torch.argmax(down_sample_gripper))
            second_skill_end_idx = int(torch.argmin(down_sample_gripper[first_skill_end_idx:]) + first_skill_end_idx)
            third_skill_end_idx = len(down_sample_gripper)
            skills_idx = [0, first_skill_end_idx, second_skill_end_idx, third_skill_end_idx]

            skill_start_index, skill_end_index = skills_idx[skill], skills_idx[skill+1]
            imgs = down_sample_imgs[skill_start_index:skill_end_index]
            traj = down_sample_traj[skill_start_index:skill_end_index]
            jntPos = down_sample_jntPos[skill_start_index:skill_end_index]

            traj_length = traj.shape[0]
            # because we do not have the next state of the last frame, so here we need to sample from 0 to trajectory_length - 1
            samples = [i for i in range(traj_length - 1)]
            loss, basic_skill_traj = evaluation(imgs, traj, jntPos, model, samples, dt, post_update=True)
            predicted_traj.append(basic_skill_traj)
            meta_test_loss += loss

            print("epoch: {}, skill {}, task {}, demo {} meta test BC loss {:.2f}\n".format(epoch, skill, task,
                                                  demo[1], meta_test_loss.item() ))

            load_back(stored_weights, model)

            optimizer.zero_grad()


        torch.save(meta_train_traj, "saved/meta_train_traj_eval")
        torch.save(meta_test_traj, "saved/meta_test_traj_eval")
        torch.save(torch.concat(predicted_traj, dim=0), "saved/predict_traj_eval")


setRandomSeed(1)
main()