import torch
import random
import numpy as np

def setRandomSeed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    disable_cuda = False
    cuda_aval = torch.cuda.is_available()
    cuda_deterministic = True

    if not disable_cuda and cuda_aval and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("********* cudnn.deterministic is set *********")