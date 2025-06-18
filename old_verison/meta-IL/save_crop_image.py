from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from torchvision import transforms
from torchvision.transforms import v2
from util.run_aguments import get_args, save_args


args = get_args()
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device(args.device)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.double),
])

post_transforms = v2.RandomChoice([
    v2.ColorJitter(brightness=.4, hue=0.0),     # adjust brightness is a good choice
    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),    # gaussian blur is a good choice
    AddGaussianNoise(0, 0.03),   # gaussian noise is also a good choice,
    v2.ColorJitter(brightness=0.0, hue=0.0),    # do nothing
], p=[0.25, 0.25, 0.25, 0.25])




def display_image(image_tensor):
    image_numpy = image_tensor.cpu().data.numpy()
    image = np.transpose(image_numpy, [1, 2, 0])
    plt.imshow(image)
    plt.show()


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

    for task in range(1, 61, 1):
        for demo in range(1, 11, 1):
            # segment data
            gripper_data = np.array(get_traj_data("meta-train/task{}/demo{}/data/gripper.txt".format(task, demo)))
            inner_gripper = torch.tensor(classify_gripper_data(gripper_data))[0:-90]
            tau1, tau2, tau3 = segment_tau(inner_gripper)

            reach_img_path = "meta-train/task{}/demo{}/img/{}.png".format(task, demo, int(0/0.04))
            img_tensor = transform(Image.open(reach_img_path)).to(device)
            img_tensor = post_transforms(img_tensor)
            img_tensor = img_tensor[:, 50:330, 250:570]
            img_tensor = img_tensor[:, ::2, ::2]
            image_numpy = img_tensor.cpu().data.numpy()
            image = np.transpose(image_numpy, [1, 2, 0])
            image = np.clip(image, 0.0, 1.0)
            matplotlib.image.imsave("key_image/reach/task-{}-demo-{}.png".format(task, demo), image)

            pickplace_img_path = "meta-train/task{}/demo{}/img/{}.png".format(task, demo, int(tau1/0.04))
            img_tensor = transform(Image.open(pickplace_img_path)).to(device)
            img_tensor = post_transforms(img_tensor)
            img_tensor = img_tensor[:, 50:330, 250:570]
            img_tensor = img_tensor[:, ::2, ::2]
            image_numpy = img_tensor.cpu().data.numpy()
            image = np.transpose(image_numpy, [1, 2, 0])
            image = np.clip(image, 0.0, 1.0)
            matplotlib.image.imsave("key_image/pickplace/task-{}-demo-{}.png".format(task, demo), image)

            push_img_path = "meta-train/task{}/demo{}/img/{}.png".format(task, demo, int( (tau1+tau2+tau3) /0.04 - 1))
            img_tensor = transform(Image.open(push_img_path)).to(device)
            img_tensor = post_transforms(img_tensor)
            img_tensor = img_tensor[:, 50:330, 250:570]
            img_tensor = img_tensor[:, ::2, ::2]
            image_numpy = img_tensor.cpu().data.numpy()
            image = np.transpose(image_numpy, [1, 2, 0])
            image = np.clip(image, 0.0, 1.0)
            matplotlib.image.imsave("key_image/push/task-{}-demo-{}.png".format(task, demo), image)

        print("save task {}".format(task))

main()