import argparse


def save_args(args, path):
    args_dict = args.__dict__
    with open(path, "w") as file:
        for key, value in args_dict.items():
            strings = "{} = {}\n".format(key, value)
            file.write(strings)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10000, help="training epoch")
    parser.add_argument("--seed", type=int, default=1, help="the random seed")
    parser.add_argument("--batch", type=int, help="batch size (tasks numbers)")
    parser.add_argument("--save-name", type=str, help="the path to save your model")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"], help="device")

    #################### meta inner update learnign rate ####################
    parser.add_argument("--inner-tau", type=float, default=0.0001, help="inner update learning rate for tau")
    parser.add_argument("--inner-goal", type=float, default=0.00005, help="inner update learning rate for goal")
    parser.add_argument("--inner-logits", type=float, default=0.00001, help="inner update learning rate for logits")
    #########################################################################

    #################### meta outer update learnign rate ####################
    parser.add_argument("--outer-tau", type=float, default=0.0001, help="outer update learning rate for tau")
    parser.add_argument("--outer-goal", type=float, default=0.00005, help="outer update learning rate for goal")
    parser.add_argument("--outer-logits", type=float, default=0.0001, help="outer update learning rate for logits")


    parser.add_argument("--dt", type=float, default=0.04, help="collect data interval, do not change it")
    parser.add_argument("--beta", type=float, default=1.0, help="soft max coefficient, do not change it")

    # ------------------------------ DMP hyperparameters ------------------------------ #
    parser.add_argument("--kp", type=float, default=150.0, help="DMP kp, do not change it")
    parser.add_argument("--kv", type=float, default=25.0, help="DMP kv, do not change it")
    parser.add_argument("--alpha", type=float, default=5, help="DMP alpha, do not change it")
    parser.add_argument("--basis-num", type=float, default=8, help="DMP alpha, do not change it")
    # --------------------------------------------------------------------------------- #

    args = parser.parse_args()
    return args