# Resources for My Students (CSML/ER Lab)

A curated (and evolving) list of training materials, datasets, tools, and “reference points” for AI4Science / CFD / dynamical systems / operator learning.

---

# Training Courses / Summer Schools
- DL4Science Summer School: https://dl4sci-school.lbl.gov/home
  - You can find the slides by clicking the title in the schedule.
- Physics-based Deep Learning (book + course material): https://physicsbaseddeeplearning.org/intro.html
- Data-driven modeling for dynamical systems (Brunton & Kutz): https://databookuw.com/
- ALCF AI for Science training series (hands-on): https://www.alcf.anl.gov/alcf-ai-science-training-series
- NVIDIA Deep Learning Institute (hands-on GPU/AI training): https://www.nvidia.com/en-us/training/
- ML for PDE / operator learning (good entry points)
  - Fourier Neural Operator (original repo): https://github.com/zongyi-li/fourier_neural_operator
  - DeepONet (original repo): https://github.com/lululxvi/deeponet
  - NeuralOperator library (FNO + variants, practical): https://github.com/neuraloperator/neuraloperator

## Short “practical ML” refreshers (when you need to fill gaps quickly)
- Stanford CS231n (CNNs + training fundamentals): https://cs231n.stanford.edu/
- Stanford CS224N (NLP, transformers fundamentals): https://web.stanford.edu/class/cs224n/
- MIT 6.S191 (intro deep learning): https://introtodeeplearning.com/
- Stanford CS236 (Deep Generative Models): https://deepgenerativemodels.github.io/
- Geometric Deep Learning (book/notes): https://geometricdeeplearning.com/
- PyTorch in One Hour (Raschka; fast practical ramp): https://sebastianraschka.com/teaching/pytorch-1h/

## Selected ML courses (advanced)
- Stanford CS221 (AI, broad foundations): https://stanford-cs221.github.io/autumn2020/modules/
- Stanford CS224W (Graph ML): http://web.stanford.edu/class/cs224w/
- UC Berkeley CS294/CS285 (Disruption Systems Research with AI): https://ucbsky.github.io/ucbsky-cs294-264-fall2025/course-website.html
- Neural Networks and Deep Learning (Michael Nielsen book): http://neuralnetworksanddeeplearning.com/
- Dive into Deep Learning — Q/K/V (attention): https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html
- CSC2541 (Uncertainty / flexible models of uncertainty): https://csc2541-f17.github.io/
- Full Stack Deep Learning (end-to-end ML/LLM systems course): https://fullstackdeeplearning.com/

## Numerical / optimization foundations (high leverage for SciML)
- Stanford EE364A (Convex Optimization — SEE): https://see.stanford.edu/Course/EE364A/85
- MIT OCW 18.155 — Differential Analysis: https://ocw.mit.edu/courses/mathematics/18-155-differential-analysis-fall-2004/
- MIT OCW 18.156 — Differential Analysis II (PDE + Fourier): https://ocw.mit.edu/courses/mathematics/18-156-differential-analysis-ii-partial-differential-equations-and-fourier-analysis-spring-2016/
- MIT 18.336 — Numerical Methods for PDE: http://math.mit.edu/icg/resources/teaching/18.336/
- MIT 18.085 — Computational Science and Engineering: http://math.mit.edu/icg/resources/teaching/18.085-spring2015/
- CMU convex optimization scribes (Ryan Tibshirani): https://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/
- Introduction to Causal Inference (Brady Neal): https://www.bradyneal.com/causal-inference-course
- Hands-On Mathematical Optimization with Python (companion notebooks): https://mobook.github.io/MO-book/intro.html
- Numerical Linear Algebra notes (Nicolas Venkovic): https://venkovic.github.io/NLA-for-CS-and-IE
- STAT 516 — Stochastic modeling of scientific data: https://faculty.washington.edu/yenchic/18A_stat516.html
- Applied Probability Notes (blog): https://appliedprobability.blog/
- Complexity Explorer (short courses on complex systems + ML fundamentals):
  https://www.complexityexplorer.org/

---

# Lecture Videos
- Steve Brunton (beginner-friendly, dynamical systems / data-driven modeling): https://www.youtube.com/@Eigensteve
- Crunch Group @ Brown (cutting-edge fluids seminars): https://www.youtube.com/@CrunchGroup

---

# Workshops / Programs (often no videos)
- UCLA IPAM workshops: https://www.ipam.ucla.edu/programs/workshops/
- ICERM (Brown) programs & workshops: https://icerm.brown.edu/program
- WMLES workshop series (Wall-Modeled LES; good for turbulence + CFD practitioners):
  https://wmles.umd.edu/workshops/workshop-2024/
- Annual Nepal AI School — Program: https://nepalschool.naamii.com.np/program

---

# Realistic Datasets

## CFD / PDE Simulation
- DrivAerML (Ahmed / external aerodynamics):
  https://www.researchgate.net/publication/383308193_DrivAerML_High-Fidelity_Computational_Fluid_Dynamics_Dataset_for_Road-Car_External_Aerodynamics
- The Well (high-resolution PDE datasets on grids): https://github.com/PolymathicAI/the_well
- CFDBench (ML-for-fluids benchmark): https://github.com/luo-yining/CFDBench
- ERCOFTAC ML for Fluid Dynamics workshop test cases / benchmark datasets:
  https://www.ercoftac.org/events/machine-learning-for-fluid-dynamics/workshop-test-cases/
- exaFOAM benchmarks (OpenFOAM-focused benchmark suite / performance + cases):
  https://exafoam.eu/benchmarks/
- CFD-Wiki: validation and test cases (nice index of canonical CFD verification/validation problems):
  http://www.cfd-online.com/Wiki/Validation_and_test_cases
- CFD Online (forums + wiki; useful for “someone has hit this bug before”):
  http://www.cfd-online.com/
- Johns Hopkins Turbulence Database (JHTDB): https://turbulence.pha.jhu.edu/
  - Cutout service (download subsets in HDF5): https://turbulence.pha.jhu.edu/cutout.aspx
- NASA Turbulence Modeling Resource (RANS/LES verification + best practices):
  https://turbmodels.larc.nasa.gov/
- SU2 test cases (useful for “known-good” CFD baselines): https://github.com/su2code/SU2/tree/master/TestCases

## ML / Code Datasets (general)
- Papers with Code — Tasks & SOTA tracker (useful for quickly mapping a field):
  https://paperswithcode.com/

## LLM / Scientific Coding
- SciCode: https://arxiv.org/pdf/2407.13168

## Benchmarks / challenges (misc)
- Nonlinear Benchmark (system ID / nonlinear dynamics benchmarks): https://sites.google.com/view/nonlinear-benchmark/benchmarks?authuser=0
- AssetOpsBench challenge (agentic/ops benchmark): https://sites.google.com/view/assetopsbench-challenge/assetopsbench?authuser=0


---

# Core References (conceptual)

## Dynamical Systems / ROM / Koopman
- Brunton & Kutz — *Data-Driven Science and Engineering* (also ties to DMD/Koopman):
  https://databookuw.com/
- PyDMD (practical DMD toolbox): https://github.com/mathLab/PyDMD
- PyKoopman (Koopman operator toolbox): https://github.com/dynamicslab/pykoopman
- ChaosBook — Nonlinear Dynamics 1: Geometry of Chaos: http://chaosbook.org/course1/about.html

## Neural Operators / Operator Learning
- FNO repo (historical reference): https://github.com/zongyi-li/fourier_neural_operator
- DeepONet repo (historical reference): https://github.com/lululxvi/deeponet
- NeuralOperator library (recommended starting point for implementations):
  https://github.com/neuraloperator/neuraloperator

## Turbulence / Fluids “sanity checks”
- NASA Turbulence Modeling Resource: https://turbmodels.larc.nasa.gov/
- JHTDB: https://turbulence.pha.jhu.edu/
- ChannelFlow (John Gibson) — canonical channel/Couette/pipe flow resources + code pointers:
  http://channelflow.org/dokuwiki/doku.php?id=gibson
- CTR (Stanford) Tea Seminar — high-quality turbulence seminar series:
  https://ctr.stanford.edu/events/ctr-tea
- SIAM Dynamical Systems (DSWeb) — notes/articles that connect fluids ↔ dynamical systems:
  https://dsweb.siam.org/
- DSWeb article: *Geometry of turbulence in wall-bounded shear flows*:
  https://dsweb.siam.org/Education/geometry-of-turbulence-in-wall-bounded-shear-flows-a-stroll-through-61506-dimensions
- UK Fluids Network — researcher resources: https://fluids.ac.uk/researcher-resources
- pyCALC-LES (LES tools / notes): https://www.tfd.chalmers.se/~lada/pyCALC-LES.html


## SciML / differentiable programming (selected)
- Implicit layers tutorial (implicit functions + AD): http://implicit-layers-tutorial.org/implicit_functions/
- Neural ODEs (clear intro + intuition): https://www.depthfirstlearning.com/2019/NeuralODEs
- Differentiating through SDEs (code primer): https://github.com/EmoryMLIP/2026-DifferentiatingSDEPrimer
- Neural PDE Solver (research code): https://github.com/bitzhangcy/Neural-PDE-Solver
- Reduced-order & surrogate modeling for digital twins (IMSI): https://www.imsi.institute/activities/digital-twins/reduced-order-and-surrogate-modeling-for-digital-twins/
- Optimal Transport tutorial (ICML 2023): https://icml.cc/virtual/2023/tutorial/21559?show_abstract=true
- CoDA — Context-informed dynamics adaptation (ICML 2022): https://arxiv.org/abs/2202.01889


## Control / RL (optional)
- Model Predictive Control (Bemporad course): http://cse.lab.imtlucca.it/~bemporad/mpc_course.html
- MPC course schedule/material (APMonitor): https://apmonitor.com/pdc/index.php/Main/CourseSchedule
- MPC (ECE5590): http://mocha-java.uccs.edu/ECE5590/index.html
- Deep RL course (Berkeley CS 285 / formerly CS 294-112): http://rail.eecs.berkeley.edu/deeprlcourse/
- Crash course on RL (repo): https://github.com/FarnazAdib/Crash_course_on_RL
- DRLinFluids (DRL + OpenFOAM platform; arXiv 2205.12699): https://arxiv.org/abs/2205.12699
- Model-based RL (control perspective talk): https://slideslive.com/38930486/modelbased-methods-in-reinforcement-learning-part-2-modelbased-control?ref=account-folder-55826-folders
- Parallel Q-learning (OpenReview): https://openreview.net/forum?id=vFvw8EzQNLy
- RLHF tutorial (ICML 2023): https://icml.cc/virtual/2023/tutorial/21554?show_abstract=true

---

# Tools (practical)

## Physics-ML / SciML frameworks (batteries-included)
- NVIDIA PhysicsNeMo (Physics-ML framework: neural operators, PINNs, geometry/PDE tooling, distributed training):
  https://github.com/NVIDIA/physicsnemo
  - Example: diffusion model for full-waveform inversion (FWI): https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/geophysics/diffusion_fwi/README.html
- Noether (Emmi AI engineering-AI framework, transformer building blocks + scaling/SLURM support):
  https://github.com/Emmi-AI/noether
- SciML (Julia) showcase: *discover “missing physics” via Universal Differential Equations (UDE)*:
  https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

## Differentiable / accelerated simulation kernels
- NVIDIA Warp (JIT-compiled Python kernels for high-performance + differentiable simulation, runs on CPU/GPU):
  https://github.com/NVIDIA/warp

## CFD / Numerics
- OpenFOAM (open-source CFD): https://www.openfoam.com/
- SU2 (open-source CFD, adjoint / optimization friendly): https://su2code.github.io/
- Palabos (LBM solver framework): http://www.palabos.org/
- Cantera (chemical kinetics / reacting flows): http://www.cantera.org/docs/sphinx/html/index.html

## ML engineering (optional)
- PyTorch C++ frontend (LibTorch) docs: https://pytorch.org/cppdocs/frontend.html
- cppflow (run TensorFlow models from C++ without Bazel): https://github.com/serizba/cppflow

## Scientific Python
- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- JAX (autodiff + XLA compilation; useful for differentiable solvers): https://github.com/jax-ml/jax
- PyTorch: https://pytorch.org/

## Optimization / AutoML
- HEBO (Bayesian optimization library): https://github.com/huawei-noah/HEBO
- AutoGluon (AutoML toolkit): https://github.com/autogluon/autogluon

## Experiment / Reproducibility
- Weights & Biases (experiment tracking): https://wandb.ai/
- MLflow: https://mlflow.org/
- Hydra (config management; helps keep runs reproducible): https://github.com/facebookresearch/hydra

## HPC / GPU performance (when your code becomes the bottleneck)
- NVIDIA Nsight Systems (system-wide GPU/CPU profiling): https://developer.nvidia.com/nsight-systems
- NVIDIA Nsight Compute (kernel-level profiling): https://developer.nvidia.com/nsight-compute
- Slurm docs (job scheduling on clusters): https://slurm.schedmd.com/documentation.html
- Differentiable simulation course (SIGGRAPH Asia 2021): https://dl.acm.org/doi/abs/10.1145/3476117.3483433
- NREL HPC: connecting to systems (good template for cluster onboarding): https://www.nrel.gov/hpc/system-connection.html
- OpenMP 4.0 spec (PDF): http://www.openmp.org/mp-documents/OpenMP4.0.0.pdf

---

# Writing / Presentation
- ABT (And–But–Therefore) scientific storytelling (MIT CEE Comm Lab):
  https://mitcommlab.mit.edu/cee/commkit/scientific-storytelling/
- “How to read a paper” (we also keep a local note in this repo): `how-to-read-a-paper.md`

## Research habits / workflow
- The Turing Way (open, reproducible, collaborative research practices):
  https://the-turing-way.netlify.app/
- What my deep model doesn't know… (Yarin Gal — uncertainty, calibration, and why it matters):
  https://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html
- Lil'Log (Lilian Weng) — high-quality, practical research notes on modern ML topics:
  https://lilianweng.github.io/
- “Everything you need for spectral clustering / diffusion maps” (nice reference page):
  https://jlmelville.github.io/smallvis/spectral.html#Diffusion_Maps
- ResearchTrend.AI (topic trend explorer): https://researchtrend.ai/
- CSPaper (paper acceptance odds / review-focused tooling): https://cspaper.org/
- Agentic Design Patterns (Chinese translation; good for agent-system design patterns):
  https://adp.xindoo.xyz/

---

# Conferences (typical targets)
- ICLR
- NeurIPS
- ICML
- AAAI
- SIAM UQ / CSE

---

# Quick “starter packs” (recommended paths)

## If you are new to AI4PDE / operator learning
1) Read/skim: Physics-based Deep Learning (site above)
2) Implement: a small FNO baseline using NeuralOperator (repo above)
3) Evaluate: on a dataset like The Well (repo above)

## If you are new to CFD but want ML-ready baselines
1) Learn a clean, standard case (e.g., cylinder / channel / flat plate)
2) Reproduce one baseline with SU2/OpenFOAM
3) Only then start adding ML components (surrogate / closure / inverse)
