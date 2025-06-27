# Introduction

ACCESS is a program established and funded by the National Science Foundation to help researchers and educators, with or without supporting grants, to utilize the nation’s advanced computing systems and services – **at no cost**.

Whether you’re looking for advanced computational resources – and outstanding cyberinfrastructure – to take your research to the next level, to explore a career in advanced CI or just to experience the amazing scientific discoveries enabled by supercomputers, you’re in the right place.

> Q: how many supercomputers are supported?

Answer: take a look at https://allocations.access-ci.org/

> Q: how to access?  

Answer: go to the website of the cluster you want to use, and search for instructions on access via "NSF ACCESS". So the way to entry will be different from those local users. Typically, you will need to use command line via `ssh USERNAME@CLUSTER_IP` in order to remotely access their computers. 

> Q: after login, how to run jobs?

Answer: when you first login, you will be in a "login node". Login node is for compiling code, move files, but not for running any jobs. To run jobs on clusters, you need to submit job scripts to a system called `slurm`. You need to checkout the user guide on the website of each cluster.   


# ACCESS Credits
ACCESS credits are the currency for exchanging hours on supercomputers. Currently our group has 750k available.  We need to spend them on A100 GPUs.

# List of GPU resources

- Indiana Jestream2 GPU
   - gpu nodes, 90 nodes, each node 128 CPU cores, with 512 GB, 4xA100 40 GB
   - by default, they will give you 1TB for free as storage
   - I have to ask exchange ACCESS credits to "Jetstream2 GPU", where 1xA100 40 GB requires 128 SUs per hour. 
   - 7.8 gpu-hours/1k credits 
- NCSA Delta GPU
   - according to the calculator, it asys 1000 credits for 15 SU
   - A100
   - not quite sure how much is that. waiting feedback from access. 
   - https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html
   - 15 gpu-hours /1k
- PSC Bridges-2 GPU
   - accordingly, it says 1000 credits for 19 SU. 
   - one GPU takes 1 SU,
   - all V100
   - 19 gpu hours/1k
- Purdue Anvil GPU
   - 15 SUs for 1000 credits
   - sub-cluster G has 16 nodes of 4 way A100. 
   - 15 gpu hours/1k
- SDSC Expanse GPU 
   - 1000 credits for 19 SU
   - V100 32 GB
   - 19 gpu hours/1k 
- TAMU ACES
   - 1000 credits for 8015 SU
   - A100 | 72 SUs per hour, RTX 6000 | 48
   - so effectively it is 1000 credits for 1000 credits for 111 hours of A100, good deal. 
   - https://hprc.tamu.edu/kb/User-Guides/AMS/Service_Unit/#example-3
- TAMU FASTER
   - same here. 1000 credits for 8015.
   - but here A100 charges for 128 SUs with 64SUs using the node alone. So it is not a good deal.


# Final conclusion
- For GPU jobs, try TAMU ACES. Even if the test-bed phase is out, and it increases prices to the same as FASTER, still effectively it is 1k credits for 41 hours. 
- For CPU jobs, try Purdue Anvil CPU. 