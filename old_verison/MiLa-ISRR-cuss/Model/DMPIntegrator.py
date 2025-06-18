import torch
import torch.nn as nn
import torch.nn.functional as F


class DMPIntegrator():
    def __init__(self, kp, kv, alpha, basisfunc):
        self.kp = kp
        self.kv = kv
        self.alpha = alpha
        self.basisfunc = basisfunc

    def __call__(self, initPos, initVel, k, goal, weights, tau, dt, device):
        """
        initPos: batch * dim
        initVel: batch * dim
        k: forward calculation step
        goal: batch * dim
        weights: batch * N (number of basis function) * dim
        """
        assert initPos.shape == initVel.shape == goal.shape
        dim = len(initPos)
        dt = dt
        DMP_pos = torch.zeros([k, dim], device=device)
        DMP_vel = torch.zeros([k, dim], device=device)
        DMP_acc = torch.zeros([k, dim], device=device)
        scale_factor = torch.zeros_like(goal)
        scale_factor[-1] = 1 # 0.20268028
        scale_factor[0:2] = goal[0:2] - initPos[0:2]

        for i in range(k - 1):
            s = torch.exp(-self.alpha / tau * (i + 1) * dt)
            # s = -alpha * s * dt / tau + s # iteration in the canonical system
            if i == 0:
                DMP_pos[0] = initPos
                DMP_vel[0] = initVel
            # here we used gaussian basis function, you can try other one like multi quadratic or linear .etc
            Phi = self.basisfunc(s)
            addsum = torch.sum(Phi, dim=-1)
            newForce = torch.matmul(Phi / addsum, weights) * s * scale_factor.detach()
            # Iteration in the 2nd-order transformation system
            DMP_acc[i] = (self.kp * (goal - DMP_pos[i]) -
                             tau * self.kv * DMP_vel[i].clone() + newForce) / tau**2
            DMP_vel[i + 1] = DMP_vel[i] + DMP_acc[i] * dt  # update velocity using acceleration
            DMP_pos[i + 1] = DMP_pos[i] + DMP_vel[i] * dt  # update position using velocity
        return DMP_pos, DMP_vel, DMP_acc