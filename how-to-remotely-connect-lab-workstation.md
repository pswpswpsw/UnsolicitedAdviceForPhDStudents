# Preface
We might need to run the jobs and want to have access to the workstation sometimes. In this post, I will guide you step by step on how to set things up.

# VPN
First, make sure you have connected to RPI inner network using VPN. Check out this at [here](https://itssc.rpi.edu/hc/en-us/articles/360008783172-VPN-Installation-and-Connection) if you don't know what to do.

# Choose a terminal
- If you are using ubuntu, you just use your standard terminal and use ssh.
- If you just want to use windows, you just need to use ssh (if installed)
- If you are using windows, you can also choose **wsl2** at [here](https://learn.microsoft.com/en-us/windows/wsl/install).

# Login via `ssh`
Once you are connected to the RPI's VPN, you can login to the lab's workstation via `ssh` (the `username` is your rpi username)
```
ssh username@pan.win.rpi.edu
```
If you don't have `ssh`, google and search for the package called `OpenSSH`. Now you should be in your `/home/username` directory. If you want to go to the `storage` disk, you should type
```
cd /mnt/storage
```

# Passwordless entry
I don't like to type password when entering supercomputer. So I follow [this](https://www.strongdm.com/blog/ssh-passwordless-login).
Basically, you just use `ssh-keygen -t rsa` to create a key on your computer, then copy that to the remote labmachine under `~/.ssh/id_rsa.pub`. 

# Runing jobs
Really? Isn't that just 
```
python some.py
```
Well, not so simple. What if you close the `ssh` connection because you need shutdown your laptop at home? Then, once the connection is lost, the session is lost. The program you just submit is no longer active.

So our solution is to use `screen`. You can use `screen` to flexibly run program in the background. It will not be lost even if you log out remotely. For a quick introduction, click [this](https://www.youtube.com/watch?v=auWiTGGB6T8). For cheatsheet, here is [it](https://quickref.me/screen).

# Running jobs economically

Currently, we only have 1 workstation which has 3 powerful GPU cards and we have at least 2 people using them. But the issue with `tensorflow` is that if you don't explicitly say you only need 1 GPU, `tensorflow` will ask for ALL GPUs with all of the memory. This is a bad practice because no one else can use this workstation at the same time even though you just need 1 GPU card. Therefore, you  should run jobs with precaution.


## Select which GPU you want to use
This works for both **Pytorch** and **Tensorflow**

```
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = “0” # here I use the first GPU, "1" means second, "2" means third
```

## Use as much GPU memory as you need

### Pytorch
Pytorch didn't pre-occupy GPU memory (wow! that's great). So it only uses what it needs. 

### Tensorflow
Tensorflow pre-occupy all GPU memory in default. We can change this by using the following commands.
```
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True) 
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
```

# Data transfer
What if you want to get your code uploaded to the server and download the trained model to your own laptop? What's the best way to do this? Here I will tell you two ways to do so.

## Easy way: mount remote folder in your local file explorer
In this way, you basically can transfer data just like you would transfer data inside your own local laptop. I like this approach more except when I have to transfer more than 100 files then this approach can be slow in terms of response.

### Windows
1. You need to install `winFSP` and `sshfs` from [here](https://winfsp.dev/rel/). Video for installation is [here](https://www.youtube.com/watch?app=desktop&v=JUDUkfEH6TA).
2. Go to your file explorer and right click `This PC` to find `Map Network Drive`
3. Set up the first one: type `\\sshfs\username@128.113.18.123` and then enter your password
4. Set up the second one type `\\sshfs\username@128.113.18.123/../../../mnt/storage/` and then enter password
You should see something like this image. There are two of them: mount point to home directory and shared disk space.
<img width="253" alt="image" src="https://user-images.githubusercontent.com/7966776/221339236-540eadad-21c6-4101-a770-bf7272a08d29.png">


### Ubuntu
check out this [video](https://www.addictivetips.com/ubuntu-linux-tips/connect-to-servers-with-gnome-file-manager/).

## Hardcore way: `scp`
If you are not offended when using terminal for all file operations, you are welcome to try this

### transfer files to your home directory
```
scp a.py b.py c.py username@128.113.18.123:/home/username
```

### transfer entire folders: `/A`, `/B`, `/C` to your home directory
```
scp -r A B C username@128.113.18.123:/home/username
```

### the other way around?
Just simply switch the order...




