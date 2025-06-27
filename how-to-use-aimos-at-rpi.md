_Updated 04/18/2024_



## Introduction
We will use the supercomputer, called AIMOS, at RPI for most of the big jobs. Here I provide a step-by-step tutorial for guiding you on how to use one. **Here, I assume you use Ubuntu**. 

## What is AIMOS?
- AIMOS is a **gateway of supercomputing clusters** open to any members at CSML. It is basically a bunch of computers stored somewhere in the university and you can access them through a terminal to execute codes. 
- For more information about AIMOS, see [here](https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/).
- For our purpose, we will be mainly using
   - dcs (IBM Power 9, difficult to configure python environment, so for CFD mostly)
   - npl (Intel machine, for deep learning training)

## How to log in?
1. make sure you have an account. if not, ask me. 
2. make sure you are either using RPI_WPA or VPN so you are connected to the intranet of RPI.
3. you need to setup DUO at https://secure.cci.rpi.edu/client/login  DUO is just a second security verification.
4. you will have an account name. For example, **my account name is _AGRMpnsh_**, so what I will do to log in is
```
ssh AGRMpnsh@blp04.ccni.rpi.edu
```
then it will ask for password. Remember, do not use `-X` here. 

- Once you log in, you will see 
```

               ** CCI SSH Gateway (Landing pad) **
**                                                             **
**     Please report all support and operation issues to       **
**     support@ccni.rpi.edu                                    **
**                                                             **
**     On-line documentation for the systems can be found at:  **
**     https://secure.cci.rpi.edu/wiki                         **
**                                                             **
**     CCI does not provide any data backup services. Users    **
**     are responsible for their own data management and       **
**     backup.                                                 **
**                                                             **
**     Use is subject to the terms of the policy for           **
**     Acceptable Use of CCI Resources.                        **
**                                                             **
[AGRMpnsh@blp01 ~]$ 

```

Warning: you are not logging into any cluster of AIMOS yet. You can see a list of available clusters at here: https://docs.cci.rpi.edu/clusters/
- for example, to login DCS, you type
```
ssh dcsfen01
```
Now you see
```
[AGRMpnsh@dcsfen01 ~]$ 
```
- Now you are in AIMOS. Specifically, you are now interfacing with the login node of AIMOS, i.e., you are talking with the gatekeeper of AIMOS.  The language you use to talk with him is **Slurm**, it is very simple to use, just remember no more than 4 commands. See [here](https://kb.swarthmore.edu/display/ACADTECH/Slurm+Commands).

**WARNING: You must NOT directly running jobs in the Login node. Otherwise your account will be deleted** 

## How to setup a Conda environment for Python in AIMOS for the very first time?

First, you need to read [here](https://docs.cci.rpi.edu/landingpads/Proxy/) to setup the proxy correct so that it enables download. 

> Q: We will use conda, but where to install conda first?

First, you need to know the resources on the clusters are limited. Everyone has three folders for personal use.
- `home` 10G (entire project, summing all users... try to not put files here )
- `barn` 25G (entire project, summing all users... try to put small amount of files here)
- `scratch` unlimited
and **another two folders across for project**, **which means anything put here can be shared across different users within the same project**
- `barn-shared`
- `scratch-shared`: unlimited again
you can use the command `df -h .` to check the maximal allow disk quota of the current directory. 
for more details, please see [here](https://docs.cci.rpi.edu/File_System/)

**For some reason, 25GB is not enough if you are using pytorch. So you can use `scratch` instead of `barn`**

We will install conda and the corresponding package in `barn`

> Q: how to install conda? 

A: official answer is here https://docs.cci.rpi.edu/software/Conda/#installing-conda-on-ppc

Follow the instruction and type `conda init`
Now you have activated conda, you should see
``` 
(base) [AGRMpnsh@dcsfen01 barn]$ 
```
which means you are in the `base` Conda environment. 

> Q: how to set up the conda environment?

Easy answer. I have prepared a configuration in a `tf2py37.txt` file, so you can just ask conda to create following my configuration. 

```
conda create -n tf2py37 --file tf2py37.txt
```
The file is right here:
[tf2py37.txt](https://github.com/csml-rpi/wiki/files/10204092/tf2py37.txt)

> Q: how to use?

You type `conda activate tf2py37`, you will activate that conda system, which contains the basic packages you need for research. 
``` 
(tf2py37) [AGRMpnsh@dcsfen01 barn]$ 
```

## How to run a job on supercomputer?
The way to use supercomputer is a bit different from what you might expect on your personal laptop: you need to submit a job through their job submitting system (called **Slurm**), and your job will wait in a line. When it is your turn, your job will be executed in the supercomputer. 

Here I show you a quick demo of how to submit your first job. 

### Your first submission for running a Python code on supercomputer
When you log into aimos, and you have activated your conda environment, **tf2py37**, say you are here
``` 
(tf2py37) [AGRMpnsh@dcsfen01 barn]$ 
```

Now you have a python code, called `test.py`, which contains
```
import tensorflow as tf
print(tf.__version__)
```
ok. Done! Now you need to write a submission script for submitting the job. For example, `test.sh`, which contains
```
#!/bin/sh
#SBATCH --job-name=GPUtest
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pswpeterpab@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks=1               
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:2

echo "Executing on the node:" $(hostname)

python test.py

nvidia-smi > check_how_many_gpu
```
This job is asking 2 gpus and run for 5 minutes. 

To submit the job, type
```bash
sbatch test.sh
```

For more information on the commands to use on slurm. Please read it [here](https://hpc.nmsu.edu/discovery/slurm/commands/). The internet has way more resources on how to use slurm.

Done. Type `squeue` to monitor how is your job!

## How to perform hyperparameter tunnig of PyTorch on npl? 

Answer: I use Ray Tune. 
1. use conda to install, following instructions at https://docs.ray.io/en/latest/index.html
2. once you finished installation (ray + tune), download this script and save as `ray-train.py` 

```python
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

ray.init(num_cpus=40,num_gpus=4)

def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


def load_test_data():
    # Load fake data for running a quick smoke-test.
    trainset = torchvision.datasets.FakeData(
        128, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.FakeData(
        16, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    return trainset, testset


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_cifar(config):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    if config["smoke_test"]:
        trainset, _ = load_test_data()
    else:
        trainset, _ = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0 if config["smoke_test"] else 8,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0 if config["smoke_test"] else 8,
    )

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")


def test_best_model(best_result, smoke_test=False):
    best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    if smoke_test:
        _, testset = load_test_data()
    else:
        _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))


config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16]),
}
# Set this to True for a smoke test that runs with a small synthetic dataset.
SMOKE_TEST = False


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2, smoke_test=False):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "smoke_test": smoke_test,
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_cifar),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
	    max_concurrent_trials=128
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result, smoke_test=smoke_test)

main(num_samples=20, max_num_epochs=20, gpus_per_trial=0.25, smoke_test=SMOKE_TEST)
```

Then use my script, save as `script.sh`
```bash
#!/bin/bash
#SBATCH --job-name=dino
#SBATCH --output=output-%A_%a.out
#SBATCH --error=error-%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pans2@rpi.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:4
##SBATCH --mem=40GB
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-gpu=40G

# Assuming '-g' flag is for GPU selection, it should match array task id for unique GPU per job
# python train.py -d navier_stokes -g 0 -r $r -t $mlp_type

cd /gpfs/u/home/AGRM/AGRMpnsh/scratch/hpopt-ray

python ray-train.py
```
3. You should go somewhere and type 
```bash
sbatch script.sh
```
Then your job will be submitted to `npl` with a request of searching for 1 node with 4 GPU, 40 cpus allocated for you. Here in this code, I am just using Ray to fully take advantage of a single node x 4 GPUs. 

### How can I manually change the number of resource for each trial? 

Answer: see here: https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html



## Misc
### How to make my prompt cool

```bash
vim ~/.bashrc
```
add this line somewhere
```bash
export PS1="\[\e[32m\][\[\e[m\]\[\e[31m\]\u\[\e[m\]\[\e[33m\]@\[\e[m\]\[\e[32m\]\h\[\e[m\]:\[\e[36m\]\w\[\e[m\]\[\e[32m\]]\[\e[m\]\[\e[32;47m\]\\$\[\e[m\] "
```

### Is the cluster currently full of jobs?

Go https://docs.cci.rpi.edu/status/ to take a look. 

### I am not satisfied with the package provided in your script. Where can I find more?

Unfortunately, the cluster uses IBM Power machine, so the arch is named `ppc64le`. Most of our personal computer either uses `linux64` or simply `win64`. Because such arch is so weird. There are very very few support from anaconda organization. Instead, the most stable package resource is maintained by IBM employee. The best resource I can find is the `ibm-ai/conda-early-access`. You can type the following command to make sure when you install with conda, you can get the latest one **supported by IBM**.

```bash
conda config --add channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/
```

You can read more about it [here](https://www.ibm.com/docs/en/wmlce/1.6.2?topic=installing-mldl-frameworks). 

