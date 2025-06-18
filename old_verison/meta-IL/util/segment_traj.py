import torch

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cpu")

def segment_tau(tau_data):
    # tau_data is a binary data
    dt = 0.04
    first_skill_end_idx = int(torch.argmax(tau_data))
    second_skill_end_idx = int(torch.argmin(tau_data[first_skill_end_idx:]) + first_skill_end_idx)
    third_skill_end_idx = len(tau_data)
    tau1 = dt * first_skill_end_idx
    tau2 = dt * second_skill_end_idx - tau1
    tau3 = dt * third_skill_end_idx - tau2 - tau1
    return torch.tensor(tau1, device=device), torch.tensor(tau2, device=device), torch.tensor(tau3, device=device)

tasks = [5]
demos = [1]

meta_train_imgs = torch.load("saved/5task3skill/task{}Demo{}Img".format(tasks[0], demos[0]))
meta_train_traj = torch.load("saved/5task3skill/task{}Demo{}Traj".format(tasks[0], demos[0]))
meta_train_gripper = torch.load("saved/5task3skill/task{}Demo{}Gripper".format(tasks[0], demos[0]))

tau1, tau2, tau3 = segment_tau(meta_train_gripper)

goal1 = meta_train_traj[int(tau1/0.04)]
goal2 = meta_train_traj[int((tau1 + tau2) / 0.04)]
goal3 = meta_train_traj[-1]

print("tau 1 {}".format(tau1.item()))
print("tau 2 {}".format(tau2.item()))
print("tau 3 {}".format(tau3.item()))

print("goal 1 {}".format(goal1))
print("goal 2 {}".format(goal2))
print("goal 3 {}".format(goal3))



