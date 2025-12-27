
# gbayesdesign  
**Quickstart & Reproducibility Guide (Narval / Compute Canada)**

---

## Overview
This repository contains the full `gbayesdesign` codebase and reproducibility materials used to generate **Figure 2** in the accompanying journal paper "Bayesian Adaptive Design for Clinical Trials with Potential Subgroup Effects".

The code implements **GPU-accelerated Bayesian power calculations** for Bayesian Adaptive Design for Clinical Trials Designs.  
All simulations and plots referenced in the paper can be reproduced using the scripts under `examples/`.

---

## System Requirements

### Hardware
- CUDA-enabled **NVIDIA GPU**

### Software
- Python **3.10+** (recommended)
- CUDA **11.7+**

### Python Packages
- Core: `cupy`, `numpy`, `scipy`
- For examples/plots: `pandas`, `matplotlib`

---

## Repository Structure
The following paths are relative to the repository root:

```markdown
gbayesdesign/
├── docs/ # html explainary documents for the source files
│ ├──BayesSampler.html
│ ├──Optimizer.html
│ ├──index.html
│ ├──mvn.html
│ ├──powerZ.html
│ └──rndgenerator.html
│ │
├── src/
│ └── gbayesdesign/ # Core Python package (GPU-accelerated Bayesian design methods)
│ │ ├── rndgenerator.py # Gets the state of the random number generator for the current device.
│ │ ├── mvn.py # A GPU-accelerated 2D Multivariate Normal PDF and CDF implementation in CuPy.
│ │ ├── BayesSampler.py # Takes input parameters and calculates new parameters and random samples for: 1) a prior distribution of drug efficacy 𝚫, 2) a posterior distribution based on the prior distribution 𝚫|Xₜ, and 3) a marginal interim distribution at t, Xₜ
│ │ ├── Optimizer.py # Generic Optimizer object, Used to standardize derivative-free constrained nonlinear multivariate solvers from different packages
│ │ └── powerZ.py # Returns a computed power based on the constraint function alpha(Z,Xt)
│ │
├── examples/ # Reproducibility materials for Figure 2
│ ├── input/ # Input design grids (CSV files)
│ │ ├── default_table_none_xall_t.csv
│ │ └── default_table_weak_xall_t.csv
│ │
│ ├── results/
│ │ ├── sub/ # Per-job outputs from GPU / SLURM runs
│ │ └── *.pdf # Final combined plots (e.g., Figure 2)
│ │
│ ├── powerz_example_true.py # Main GPU simulation & power calculation script
│ ├── run_powerz_true.sh # SLURM submission script for Narval GPU jobs
│ ├── combine.py # Merges per-job CSV outputs into combined result tables
│ └── plot_fig2.py # Generates Figure 2 from combined results
│
├── pyproject.toml # Build system configuration
├── setup.cfg # Package metadata and dependencies
└── LICENSE.md # MIT License
```

---

## Citation / Usage

This code is provided to support reproducibility of results reported in the associated manuscript.
Please cite the paper when using or adapting this implementation.
```
Xuekui Zhang, Qianyun Zhao, Cong Chen, Belaid Moa, and Shelley Gao. Bayesian Adaptive Design for Clinical Trials with Potential Subgroup Effects, (submitted in 2026)
```
---

## Software Installation and Quick Start

### Hardware
- CUDA-enabled NVidia GPU

### Packages
- CuPy
- NumPy
- SciPy

### Installation

1. Install requirements: 

    ```
    pip install numpy scipy cupy
    ```

2. Navigate to package root: 

    ```
    cd ~/gbayesdesign_clean
    ```
3. Use setuptools to install: 

    ```
    pip install -e .
    ```

4. (Optional) Install testing interface packages

    ```
    pip install --no-index pandas matplotlib
    ```

---

## Reproducing Results from the Paper 

### Environment Setup (Narval / Compute Canada)

#### Load modules
```bash
module load StdEnv/2020 gcc/11.3.0
module load python/3.10.2 cuda/11.7
````

#### Create and activate virtual environment

```bash
python -m venv ~/environments/gbayesdesign
source ~/environments/gbayesdesign/bin/activate
pip install --upgrade pip
```

#### Install dependencies and package

```bash
pip install numpy scipy cupy pandas matplotlib
pip install -e .
```

---

#### Input Design Grids

Input grids used in the paper are provided under `examples/input/`:

* `default_table_none_xall_t.csv`
  Correctly estimated **non-biomarker** effect parameters

* `default_table_weak_xall_t.csv`
  Correctly estimated **biomarker** effect parameters

Custom design grids may be supplied, provided they follow the same column schema.

---

### Reproducing Figure 2

The following steps reproduce
**`examples/results/grid_power_vs_r_2x2_true_4.pdf`**.

#### Step 1 — Run GPU simulations (SLURM)

From `examples/`, submit batch jobs (update account name in the script before use):

```bash
sbatch run_powerz_true.sh 0
sbatch run_powerz_true.sh 1
sbatch run_powerz_true.sh 2
sbatch run_powerz_true.sh 3
sbatch run_powerz_true.sh 4
```

Each job:

* Runs `powerz_example_true.py` for both **none** and **weak** effect settings
* Writes per-job CSV outputs to `results/sub/` as:

  ```
  t_Test_i<job>_<effect>_combined_results_01.csv
  ```

Batching behavior can be adjusted via:

* `--num_batches (-nb)`
* `--test_iteration (-T)`

---

#### Alternative: Run without SLURM (single batch)

```bash
cd examples
source ~/environments/gbayesdesign/bin/activate

python powerz_example_true.py \
  -i input/default_table_none_xall_t.csv \
  --effect_setting 0 -tt t_ -nb 1 -T 0

python powerz_example_true.py \
  -i input/default_table_weak_xall_t.csv \
  --effect_setting 1 -tt t_ -nb 1 -T 0
```

---

#### Step 2 — Combine per-job outputs

From `examples/`:

```bash
python combine.py
```

Outputs:

* `results/t_Test_iall_none_combined_results.csv`
* `results/t_Test_iall_weak_combined_results.csv`

---

#### Step 3 — Generate Figure 2

```bash
python plot_fig2.py
```

Output:

```
results/grid_power_vs_r_2x2_true_4.pdf
```

---

### Notes

* `run_powerz_true.sh` requests **1 GPU**, ~**1 GB RAM**, and **5:59:00** walltime.
* `powerz_example_true.py` key parameters:

  * `--effect_setting {0,1,2}` → none / weak / strong
  * `--num_batches (-nb)` → split grid across jobs
  * `--test_iteration (-T)` → batch index
* `plot_fig2.py` assumes combined CSVs already exist (run `combine.py` first).



