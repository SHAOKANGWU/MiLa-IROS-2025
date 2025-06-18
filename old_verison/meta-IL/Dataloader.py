from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PIL import Image


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


class Dataset(Dataset):
    """
    Load just one dataset.
    However, meta learning required paired data.
    It means we need to load meta-train and meta-test dataset simultaneously
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        if transform is None:
            self.transform = transforms.Compose([
                transforms.PILToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        """
        return the number of demo
        """
        folder_list = os.listdir(self.root_dir)
        return len(folder_list)

    def __getitem__(self, idx):
        """
        return the imgs and traj in the demo_idx folder
        """
        demo_name = "demo{}".format(idx+1)
        demo_path = os.path.join(self.root_dir, demo_name)

        # each demo include two folders (img and traj)
        folder_name = os.listdir(demo_path)

        # folder_name[0] = "img", folder_name[1] = "traj"
        img_folder_path = os.path.join(demo_path, folder_name[0])
        traj_folder_path = os.path.join(demo_path, folder_name[1])

        # get corresponding img data
        imgs = []
        all_imgs_list = os.listdir(img_folder_path)
        # sort the image according to time
        all_imgs_list.sort(key=lambda x: int(x.split(".")[0]))
        for img_name in all_imgs_list:
            img_path = os.path.join(img_folder_path, img_name)
            image = Image.open(img_path)
            img_tensor = self.transform(image)
            # img_tensor = img_tensor[:, 10:306, 225:437]  # 5 tasks 6 demos per task
            # img_tensor = img_tensor[:, 30:330, 100:500]    # 20 tasks 10 demos per task
            img_tensor = img_tensor[:, 50:330, 250:570]    # 5 tasks 5 demos per task
            imgs.append(img_tensor)
        # imgs_tensor shape: frame_length, 3, height, width
        imgs_tensor = torch.stack((imgs))

        # get corresponding trajectory data
        traj_data_path = os.path.join(traj_folder_path, "traj.txt")
        traj = get_traj_data(traj_data_path)
        assert len(traj) % 21 == 0
        traj_tensor = torch.tensor(traj).reshape(-1, 21)
        # xyz_tensor shape: frame_length, 3
        xyz_tensor = traj_tensor[:, 14:17]
        jntPos_tensor = traj_tensor[:, 0:7]

        return imgs_tensor, xyz_tensor, jntPos_tensor


class MetaDataset(Dataset):
    """
    Load just one dataset.
    However, meta learning required paired data.
    It means we need to load meta-train and meta-test dataset simultaneously
    """

    def __init__(self, meta_train_dir, meta_test_dir, transform=None):
        """
        Args:
            meta_train_dir (string): Directory with all the meta train images.
            meta_test_dir (string): Directory with all the meta test images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_train_dir = meta_train_dir
        self.meta_test_dir = meta_test_dir

        if transform is None:
            self.transform = transforms.Compose([
                transforms.PILToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        """
        return the number of paired demo
        """
        meta_train_folder_list = os.listdir(self.meta_train_dir)
        meta_test_folder_list = os.listdir(self.meta_test_dir)
        assert len(meta_train_folder_list) == len(meta_test_folder_list)
        return len(meta_train_folder_list)

    def __getitem__(self, idx):
        """
        return the paired imgs and traj (meta train and meta test) in the demo_idx folder
        """
        demo_name = "demo{}".format(idx)
        dir_list = [self.meta_train_dir, self.meta_test_dir]
        index_list = [0, 1]  # 0 denote for meta_train, 1 denote for meta_test
        for dir, index in zip(dir_list, index_list):
            demo_path = os.path.join(dir, demo_name)

            # each demo include two folders (img and traj)
            folder_name = os.listdir(demo_path)

            # folder_name[0] = "img", folder_name[1] = "traj"
            img_folder_path = os.path.join(demo_path, folder_name[0])
            traj_folder_path = os.path.join(demo_path, folder_name[1])

            # get corresponding img data
            imgs = []
            all_imgs_list = os.listdir(img_folder_path)
            # sort the image according to time
            all_imgs_list.sort(key=lambda x: int(x.split("-")[0]))
            for img_name in all_imgs_list:
                img_path = os.path.join(img_folder_path, img_name)
                image = Image.open(img_path)
                img_tensor = self.transform(image)
                imgs.append(img_tensor)
            # imgs_tensor shape: frame_length, 3, height, width
            imgs_tensor = torch.stack((imgs))

            # get corresponding trajectory data
            traj_data_path = os.path.join(traj_folder_path, "data.txt")
            traj = get_traj_data(traj_data_path)
            assert len(traj) % 21 == 0
            traj_tensor = torch.tensor(traj).reshape(-1, 21)
            # xyz_tensor shape: frame_length, 3
            xyz_tensor = traj_tensor[:, 14:17]
            if index == 0:
                meta_train_imgs_tensor, meta_train_xyz_tensor = imgs_tensor, xyz_tensor
            elif index == 1:
                meta_test_imgs_tensor, meta_test_xyz_tensor = imgs_tensor, xyz_tensor
            else:
                raise NotImplementedError("only support meta train and meta test dataset")

        return meta_train_imgs_tensor, meta_train_xyz_tensor, meta_test_imgs_tensor, meta_test_xyz_tensor


def plot_3D_traj(xyz_tensor):
    # plot 3D trajectory
    # xyz_tensor shape: [trajectory_length, 3]
    xyz_numpy = xyz_tensor.numpy()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyz_numpy[:, 0], xyz_numpy[:, 1], xyz_numpy[:, 2])
    plt.show()


transform = transforms.Compose([
    transforms.PILToTensor()
])


def load_single_dataset():
    dataset = SingleDataset(root_dir="video_dataset/meta-train/task1", transform=transform)
    res = len(dataset)
    imgs_tensor, xyz_tensor = dataset[1]
    assert imgs_tensor.shape[0] == xyz_tensor.shape[0]
    dataloader = DataLoader(dataset, 3, shuffle=True)
    for imgs_tensor, xyz_tensor in dataloader:
        print(1)


def load_meta_dataset():
    dataset = MetaDataset(meta_train_dir="data/MetaDataset/meta-train",
                          meta_test_dir="data/MetaDataset/meta-test",
                          transform=transform)
    res = len(dataset)
    meta_train_imgs_tensor, meta_train_xyz_tensor, meta_test_imgs_tensor, meta_test_xyz_tensor = dataset[1]
    assert meta_train_imgs_tensor.shape[0] == meta_train_xyz_tensor.shape[0]
    assert meta_test_imgs_tensor.shape[0] == meta_test_xyz_tensor.shape[0]
    dataloader = DataLoader(dataset, 2, shuffle=True)

    for meta_train_imgs_tensor, meta_train_xyz_tensor, meta_test_imgs_tensor, meta_test_xyz_tensor in dataloader:
        print(1)
