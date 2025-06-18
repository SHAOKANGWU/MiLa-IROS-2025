import numpy as np
from gmr import *
import torch

cut_off_mp1 = 17 / 0.04 * 0.0333
cut_off_mp2 = 7.85 / 0.04 * 0.0333
cut_off_mp3 = 9.5 / 0.04 * 0.0333

dt = 0.0333

d_in = range(0, 1)
d_out = range(1, 4)

new_mp = torch.load("mp3_GMM.npy")

cut_off_t = cut_off_mp3

tau = cut_off_t

k = int(tau/dt)

def get_gmm_cov(gmm_model, _cut_off_t, _tau, _k):
    mp_t_list = []
    mp_scale = _cut_off_t / _tau
    for i in range(_k-1):
        mp_t = (i + 1) * mp_scale * dt
        mp_t_list.append(mp_t)
    _, skill_cov_array = gmr(gmm_model, mp_t_list, d_in, d_out, len(mp_t_list))

    skill_cov_std = np.sqrt(skill_cov_array) * 1000
    cov_pinv = {}

    for index, t in enumerate(mp_t_list):
        cov_pinv[t] = torch.tensor(np.linalg.pinv(skill_cov_array[index]), device="cuda:0")

    return cov_pinv


cov_pinv = get_gmm_cov(new_mp, cut_off_t, tau, k)

gmm_pinv = torch.save(cov_pinv, "gmm3_pinv")

print(1)

