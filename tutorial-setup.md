*Notice: For those of you don't have a Ubuntu system, you can come to me for a disk that will install `Ubuntu 20.04` for you.*

Here I will show an example of the recommended setup for `Ubuntu system 20.04`. We will mostly working in a `Python 3` environment. We use `numpy`, `tensorflow`, `pytorch`, and `matplotlib` frequently. We will not use Matlab. 

## Conda
Since nearly all of the python codes are open source, it means everyone can write their own packages. So it is not like matlab where one company says what you will be installing. Since we are mostly working with Python, how shall we manage so many python packages? 

The answer is to use conda, which is a environment management software that can help you create a *sandbox* where you can install whatever you want in this sandbox while not affecting your original system. It is not like the standard `pip` install where you might use before. Pip is installing python system-wide so one mistake, you have to suffer from this. But for conda, you just `conda deactivate` then the alarm is over. 

## Pre-req
- `sudo apt-get install gcc`
- `sudo apt-get install make`

## Steps for setup conda
- go to conda website to download and install `miniconda`: https://docs.conda.io/en/latest/miniconda.html
- type `conda activate` to enable your conda environment. note that now you are in `base` environment, which should not install any specific packages
- type `conda create -n myenv` to create your conda environment. **Note: myenv is just a random name - you can change**
- type `conda activate myenv` to get into the conda environment you just created

## Setup conda channels
We will use conda to get our package installed that is customized to the environment of our computer. So, since our computers can vary, who did that customization configuration in the first place? 

The answer is some developers will upload their bug-free configuration online. They will post it on `anaconda.org`. Here is what I believe are the 4 most useful channels
- `conda-forge`
- `anaconda`
- `pytorch`
- `nvidia`

To add the channel in your conda, type
```bash
conda config --add channels conda-forge
conda config --add channels anaconda
conda config --add channels pytorch
conda config --add channels nvidia
```

## Install packages in conda

**Before installation, you need to know if your laptop has NVIDIA GPU or not.**

### Pytorch
- go to pytorch's website to figure out the command for installing pytorch

> https://pytorch.org/get-started/locally/

**DO NOT USE TENSORFLOW. It is abandoned by the community.**

<del>
### Tensorflow (if your computer does not have a qualified GPU)
*Update (04/11/2023): because latest Torch and TF used different cuda versions, it is the best to separate them into two conda environment. Otherwise it will take forever to "solving the environment" in Conda.*
- first type `conda search tensorflow` to get an idea about what available choices of tensorflow that you can use
- simply type `conda install tensorflow`, it will install the CPU version for you. It is good for debugging purpose though a bit slow. 

### Tensorflow-GPU
- first type `conda search tensorflow-gpu` to get an idea about what available choices of tensorflow that you can use
- In principle, you need to install a few things to support Nvidia GPU
  - `cuda 11.x`  (the language to ask GPU do the job)
  - `cudnn`  (the tool to ask GPU to efficiently train deep neural networks)
- But in some cases, conda may already did this for you.
- type `conda install -n myenv tensorflow-gpu`
  - you could specify the version you want to use as well. 
</del>
**Q: What if it takes too long to solve the environment?**
> A: Simply deactivate this environment. Then create a blank environment just for tensorflow gpu
```python
conda create -n tf2 tensorflow-gpu
```

**Q: How to check if the GPU version installed correctly?**
> A: You type `nvidia-smi` to check the usage of your GPU. And when you type `import tensorflow as tf`, you should see something pops up in the terminal that has information about cuda library``


### Misc
- type `conda install -n myenv matplotlib scipy numpy scikit-learn`
- type `conda install notebook` to install jupyter notebook

## Summary
In this page, you just learn how to install the following package and how to use conda to setup your Python programming environment. So what are these packages for?

- **Pytorch**: a tool to build any computational graph and equip that graph with automatic differentiation --- a special graph that it is designed for is deep neural network. 
- **Tensorflow**: the same as above but from Google, the syntax is different but the 2.x version of TF is close to Pytorch.
- **Matplotlib**: a library to plot figures, contours, scatter plot, etc.
- **Scipy**: a library to perform any kind of linear algebra operations. It is fast compared to native python because it is calling pre-compiled lapack in the backend.
- **Numpy**: the basic library for dealing with numeric arrays in python.
- **Scikit-learn**: the machine learning library in Python. But no deep learning.  