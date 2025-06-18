import torch


class GaussianBasisFunction():
    def __init__(self, basis_num, alpha, device):
        # u = i/N
        self.device = device
        self.u = torch.zeros(basis_num, device=self.device)
        # c = e^{-alpha * u}
        self.c = torch.zeros(basis_num, device=self.device)  # centers of basis functions
        self.h = torch.zeros(basis_num, device=self.device)  # widths of basis functions
        for i in range(basis_num):
            self.u[i] = 1.0 / basis_num * i
            self.c[i] = torch.exp(-alpha * self.u[i])
        for i in range(basis_num - 1):
            self.h[i] = 0.5 / ((0.65 * (self.c[i + 1] - self.c[i])) ** 2)
        self.h[basis_num - 1] = self.h[basis_num - 2]
        self.Phi = torch.zeros(basis_num, device=self.device)

    def __call__(self, s):
        # res = [\phi_0(s), \phi_1(s), ..., \phi_N(s)]
        res = torch.exp(-self.h * (s - self.c) ** 2)
        return res