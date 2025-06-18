import os
import sys

dataset_path = "/home/robot-learn/meta-train-part2"

tasks_name = os.listdir(dataset_path)
tasks_name.sort(key=lambda x: int(x.split("task")[-1]))

for task in tasks_name:
    task_path = os.path.join(dataset_path, task)
    demos_path = os.listdir(task_path)
    demos_path.sort(key=lambda x: int(x.split("demo")[-1]))
    for demo in demos_path:
        path = os.path.join(task_path, demo)
        desired_path = "/home/robot-learn/Desktop/meta-IL/meta-train/{}/{}".format(task, demo)
        # print(desired_path)
        os.rename(path, desired_path)




# move part-2 dataset to part-1 dataset
