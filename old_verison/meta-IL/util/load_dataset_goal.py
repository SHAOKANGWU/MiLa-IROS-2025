import random
datasets = {}

random.seed(3)
tasks = [i for i in range(1, 61, 1)]
tasks.pop(5)    # task 6, demo 9, is not a good training data (see its video)
random.shuffle(tasks)

for task in tasks:
    demos = [i for i in range(1, 11, 1)]
    random.shuffle(demos)
    datasets[task] = demos

train_datasets, valid_datasets, test_dataset = {}, {}, {}

index = 0
for key, value in datasets.items():
    # training dataset (0 to 50), validation dataset (50, 55), test dataset (55, 60)
    if 0 <= index < 50:             # 17 refers to 30% data, 25 refers to half data, 50 refers to full data
        train_datasets[key] = value
    elif 50 <= index < 55:
        valid_datasets[key] = value
    elif 55 <= index < 60:
        test_dataset[key] = value
    else:
        pass
    index += 1


def sample_demos(batch_size):

    tasks = list(datasets.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(datasets[task], 2)
        batch_demos.append(demo)

    return batch_tasks, batch_demos


def train_demos(batch_size):

    tasks = list(train_datasets.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(datasets[task], 2)
        batch_demos.append(demo)

    return batch_tasks, batch_demos


def valid_demos(batch_size):

    tasks = list(valid_datasets.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(datasets[task], 2)
        batch_demos.append(demo)

    return batch_tasks, batch_demos


def test_demos(batch_size):

    tasks = list(test_dataset.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(datasets[task], 2)
        batch_demos.append(demo)

    return batch_tasks, batch_demos