import torch
import numpy as np

mp1 = np.load("Skill-Repertoire/mp1.npy")
mp2 = np.load("Skill-Repertoire/mp2.npy")
mp3 = np.load("Skill-Repertoire/mp3.npy")

print("Reach primitive weights")
print(mp1)

print("Place primitive weights")
print(mp2)

print("Push primitive weights")
print(mp3)