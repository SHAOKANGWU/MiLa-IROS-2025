# import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from util import setRandomSeed, GaussianBasisFunction
from Dataloader import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(path):
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


def main():

    # load dataset
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.double),
    ])

    transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
    ])


    for task in range(1, 61, 1):
        for demo in range(9, 10, 1):
            dataset = Dataset(root_dir="meta-train/task{}".format(task), transform=transform)
            image, traj, jntPos = dataset[demo-1]
            gripper_data = np.array(get_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            # tt = np.linspace(0, len(gripper_data) * 0.04, len(gripper_data))
            # plt.plot(tt, gripper_data)
            # plt.savefig('saved/gripper/task-{}-demo-{}.png'.format(task, demo))
            # plt.close()
            gripper_data_status = torch.tensor(classify_gripper_data(gripper_data))
            torch.save(image[0:-91], "/media/robot-learn/Elements/dataset/task{}Demo{}Img".format(task, demo))
            torch.save(traj[0:-90], "/media/robot-learn/Elements/dataset/task{}Demo{}Traj".format(task, demo))
            torch.save(jntPos[0:-90], "/media/robot-learn/Elements/dataset/task{}Demo{}JntPos".format(task, demo))
            torch.save(gripper_data_status[0:-90], "/media/robot-learn/Elements/dataset/task{}Demo{}Gripper".format(task, demo))
            print("saved task {} demo {} data".format(task, demo))

    raise NotImplementedError("Stop running your code here")



    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    image = transform2(image)

    prediction = model(image[0:1]).softmax(1)
    classification = prediction.argmax().item()
    print("classification results {}".format(classification))

    display_image = image[0].data.numpy()
    display_image = np.transpose(display_image, [1, 2, 0])
    # cv2.imshow("image", display_image)
    # cv2.waitKey()
    plt.imshow(display_image)
    plt.show()


def calculate_skills_average_time():

    def segment_tau(tau_data):
        # tau_data is a binary data
        dt = 0.04
        first_skill_end_idx = int(torch.argmax(tau_data))
        second_skill_end_idx = int(torch.argmin(tau_data[first_skill_end_idx:]) + first_skill_end_idx)
        third_skill_end_idx = len(tau_data)
        tau1 = dt * first_skill_end_idx
        tau2 = dt * second_skill_end_idx - tau1
        tau3 = dt * third_skill_end_idx - tau2 - tau1
        return tau1, tau2, tau3

    tau1_list, tau2_list, tau3_list = [], [], []
    total_tau_list = []
    for task in range(1, 61, 1):
        for demo in range(1, 11, 1):
            try:
                gripper_data = np.array(get_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))[0:-90]
                gripper_data_status = torch.tensor(classify_gripper_data(gripper_data))
                tau1, tau2, tau3 = segment_tau(gripper_data_status)
                tau1_list.append(tau1)
                tau2_list.append(tau2)
                tau3_list.append(tau3)
                total_tau_list.append(tau1 + tau2 + tau3)
            except:
                print("error found in task {}, demo {}".format(task, demo))

    tau1_mean, tau1_std = np.mean(np.array(tau1_list)), np.std(np.array(tau1_list))
    tau2_mean, tau2_std = np.mean(np.array(tau2_list)), np.std(np.array(tau2_list))
    tau3_mean, tau3_std = np.mean(np.array(tau3_list)), np.std(np.array(tau3_list))
    total_tau_mean, total_tau_std = np.mean(np.array(total_tau_list)), np.std(np.array(total_tau_list))

    print("skill 1, mean {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}".format(tau1_mean, tau1_std, min(tau1_list), max(tau1_list)))
    print("skill 2, mean {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}".format(tau2_mean, tau2_std, min(tau2_list), max(tau2_list)))
    print("skill 3, mean {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}".format(tau3_mean, tau3_std, min(tau2_list), max(tau2_list)))
    print("total trajectory mean {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}".format(total_tau_mean, total_tau_std, min(total_tau_list), max(total_tau_list)))



setRandomSeed(1)
calculate_skills_average_time()
# main()