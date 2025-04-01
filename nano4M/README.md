# COM-304 - nano4M Project  

Welcome to the nano4M project exercises!  

In this project, you will conduct hands-on experiments to implement and train nano4M, a multi-modal foundation model capable of any-to-any generation.  

This project consists of three parts:  

1) In this first part, we start by implementing the necessary building blocks to construct an autoregressive Transformer, like GPT.
2) In part 2, we will build a masked model in the style of MaskGIT.  
3) In part 3, we will build a simple 4M-like multimodal model.

Course materials for each part will be released in stages, see the schedule below.  

### Instructions

The instructions for each of these three parts are provided in the notebooks, which you can find under `./notebooks/`. They will introduce the problem statement to you, explain what parts in the codebase need to be completed, and you will use them to perform inference on the trained models. You will be asked to run the cells in those notebooks, provide answers to questions, etc. 

The total project grade will count 40% of your final grade and consists of the three notebooks, your extensions, as well as the demo, presentation and report of your extensions.
The notebooks count 50% towards that (i.e. 20% of your final grade), while the extension demo, presentation and report count another 50% (i.e. also 20% of your final grade).

The notebooks are to be completed and submitted individually. You may discuss with your group members, but are encouraged to try and solve the exercises on your own.

## **Important Dates**  

Below is the completion and homework submission timeline for each part. Please refer to Moodle for further updates and instructions.  

### **Part 1: nanoGPT**  
- **Release:** Tue 25.3.2025  
- **Due:** By 23:59 on Fri 4.4.2025  

### **Part 2: nanoMaskGIT**  
- **Release:** Tue 1.4.2025  
- **Due:** By 23:59 on Fri 11.4.2025  

### **Part 3: nano4M**  
- **Release:** Tue 8.4.2025  
- **Due:** By 23:59 on Fri 18.4.2025  

### **Progress Report (Updated Extension Proposal)**
- **Due:** By 23:59 on Fri 18.4.2025  

## **Installation**  

To begin the experiments (and submit jobs), we first need to install the required packages and dependencies. For ease of installation and running experiments directly, we provide a pre-installed Anaconda environment that you can activate as follows:  

```bash
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda init
conda activate nanofm
```

Alternatively, you can install the environment yourself by running [setup_env.sh](setup_env.sh)
```bash
bash setup_env.sh
```

## Getting Started

We will implement the building blocks of autoregressive, masked, and multimodal models and train them on language and image modeling tasks.

You will primarily run the following files:
1. Jupyter notebooks: `nano4M/notebooks/` 
   - Usage: Introduction of the week's tasks and inference (post-training result generation and analysis).
2. Main training script: `run_training.py` 
   - Usage: Train your models after implementing the building blocks (refer to the notebooks for more details).

### Jupyter notebooks `nano4M/notebooks/`:
To use the Jupyter notebooks, activate the `nano4m` environment and run the notebooks. Follow the same steps outlined in [4M_Tutorial Environment Setup](4M_Tutorial/Environment.md) to launch the notebook in a browser.

### Main training script `run_training.py`:

You can run the training job in one of two ways:

1. **Interactively using `srun`** – great for debugging.
2. **Using a SLURM batch script** – better for running longer jobs.

> **Before you begin**:  
> Make sure to have your Weights & Biases (W&B) account and obtain your W&B API key.  
> Follow the instructions in **Section 1.3 (Weights & Biases setup)** of the Jupyter Notebook.

---

#### Option 1: Run Interactively via `srun`

Start an interactive session on a compute node (eg, 2 GPUs case):

```bash
srun -t 120 -A com-304 --qos=com-304 --gres=gpu:2 --mem=16G --pty bash
```
Then, on the compute node:

```bash
conda activate nanofm
wandb login
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py --config cfgs/nanoGPT/tinystories_d8w512.yaml
```
> **Note:**  
> To run the job on **one GPU**, make sure to:
> - Adjust the `--gres=gpu:1` option in the `srun` command, and  
> - Set `--nproc_per_node=1` in the `torchrun` command.

#### Option 2: Submit as a Batch Job via SLURM
You can use the provided submit_job.sh script to request GPUs and launch training.

Run:
```bash
sbatch submit_job.sh <config_file> <your_wandb_key> <num_gpus>
```
Replace the placeholders as follows:

- <config_file> — Path to your YAML config file

- <your_wandb_key> — Your W&B API key

- <num_gpus> — Set to 1 or 2 depending on your setup

Example Usage:
```bash
sbatch submit_job.sh cfgs/nanoGPT/tinystories_d8w512.yaml abcdef1234567890 2
```
