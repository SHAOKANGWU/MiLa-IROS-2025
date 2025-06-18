import numpy as np
import torch

old_mp1 = np.load("mp1.npy")
old_mp2 = np.load("mp2.npy")
old_mp3 = np.load("mp3.npy")

new_mp1 = np.load("Skill-Repertoire/mp1.npy")
new_mp2 = np.load("Skill-Repertoire/mp2.npy")
new_mp3 = np.load("Skill-Repertoire/mp3.npy")

abs = np.abs(old_mp1 - new_mp1)
print(np.max(abs))

abs = np.abs(old_mp2 - new_mp2)
print(np.max(abs))

abs = np.abs(old_mp3 - new_mp3)
print(np.max(abs))


old_pinv1 = torch.load("Skill-Repertoire/gmm1_pinv")
old_pinv2 = torch.load("Skill-Repertoire/gmm2_pinv")
old_pinv3 = torch.load("Skill-Repertoire/gmm3_pinv")

new_pinv1 = torch.load("gmm1_pinv")
new_pinv2 = torch.load("gmm2_pinv")
new_pinv3 = torch.load("gmm3_pinv")


def compare_two_dict(dict1, dict2):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    print("key is equal", keys1 == keys2)

    for key in keys1:
        value1 = dict1[key]
        value2 = dict2[key]
        if not torch.equal(value1, value2):
            print("value not equal")
    print("value is equal")

compare_two_dict(old_pinv1, new_pinv1)
compare_two_dict(old_pinv2, new_pinv2)
compare_two_dict(old_pinv3, new_pinv3)