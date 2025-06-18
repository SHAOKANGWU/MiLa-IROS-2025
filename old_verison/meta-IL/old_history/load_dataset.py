import random
datasets = {}

random.seed(0)
tasks = [i for i in range(1, 61, 1)]
random.shuffle(tasks)

for task in tasks:
    demos = [i for i in range(1, 6, 1)]
    random.shuffle(demos)
    datasets[task] = demos

train_datasets, test_datasets = {}, {}

index = 0
for key, value in datasets.items():
    if index <= 56:
        train_datasets[key] = value
    else:
        test_datasets[key] = value
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


def test_demos(batch_size):

    tasks = list(test_datasets.keys())
    batch_tasks = []
    batch_demos = []

    for _ in range(batch_size):
        task = random.sample(tasks, 1)[0]
        batch_tasks.append(int(task))
        demo = random.sample(datasets[task], 2)
        batch_demos.append(demo)

    return batch_tasks, batch_demos