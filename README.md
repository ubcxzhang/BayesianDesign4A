# gbayesdesign

**Quick Start & Reproducibility Guide (Narval / Compute Canada)**

---

## Overview

This repository contains the complete `gbayesdesign` codebase and all reproducibility materials required to generate **Figure 2** in the accompanying manuscript:

> *Bayesian Adaptive Design for Clinical Trials with Potential Subgroup Effects*.

The package implements **GPU‑accelerated Bayesian power calculations** for adaptive clinical trial designs with potential subgroup effects. All simulations and figures reported in the manuscript can be reproduced using the scripts provided under `examples/`.

---

## System Requirements

### Hardware

* CUDA‑enabled **NVIDIA GPU**

### Software

* Python **3.10+** (recommended)
* CUDA **11.7+**

### Python Dependencies

* Core: `cupy`, `numpy`, `scipy`
* Examples / plotting: `pandas`, `matplotlib`

---

## Repository Structure

Paths are relative to the repository root:

```
gbayesdesign/
├── docs/                         # HTML API documentation (generated via pdoc)
│   ├── BayesSampler.html
│   ├── Optimizer.html
│   ├── index.html
│   ├── mvn.html
│   ├── powerZ.html
│   └── rndgenerator.html
│
├── src/
│   └── gbayesdesign/             # Core Python package
│       ├── rndgenerator.py       # GPU‑aware random number generator utilities
│       ├── mvn.py                # GPU‑accelerated 2D multivariate normal PDF/CDF
│       ├── BayesSampler.py       # Bayesian prior, posterior, and interim sampling
│       ├── Optimizer.py          # Derivative‑free constrained optimization wrappers
│       └── powerZ.py             # Power computation based on α(Z, X_t)
│
├── examples/                     # Reproducibility materials for Figure 2
│   ├── input/                    # Input design grids (CSV)
│   │   ├── default_table_none_xall_t.csv
│   │   └── default_table_weak_xall_t.csv
│   │
│   ├── results/
│   │   ├── sub/                  # Per‑job SLURM/GPU outputs
│   │   └── *.pdf                 # Final combined plots (e.g., Figure 2)
│   │
│   ├── powerz_example_true.py    # Main GPU simulation & power calculation script
│   ├── run_powerz_true.sh        # SLURM submission script (Narval)
│   ├── combine.py                # Merge per‑job CSV outputs
│   └── plot_fig2.py              # Generate Figure 2 from combined results
│
├── tests/
│   ├── test_BayesSampler.py      # Unit tests for BayesSampler
│   └── test_mvn.py               # Unit tests for mvn utilities
│
├── pyproject.toml                # Build system configuration
├── setup.cfg                     # Package metadata and dependencies
└── LICENSE.md                    # MIT License
```

---

## Citation and Usage

This software is provided to support reproducibility of results reported in the associated manuscript. If you use or adapt this code, please cite:

```
Xuekui Zhang, Qianyun Zhao, Cong Chen, Belaid Moa, and Shelley Gao.
Bayesian Adaptive Design for Clinical Trials with Potential Subgroup Effects.
(Submitted, 2026)
```

---

## Installation and Quick Start

### Prerequisites

* CUDA‑enabled NVIDIA GPU

### Installation

1. Install core dependencies:

```bash
pip install numpy scipy cupy
```

2. Navigate to the repository root:

```bash
cd ~/gbayesdesign_clean
```

3. Install the package in editable mode:

```bash
pip install -e .
```

4. (Optional) Install plotting and analysis dependencies:

```bash
pip install pandas matplotlib
```

5. (Optional) Run unit tests:

```bash
python tests/test_BayesSampler.py
python tests/test_mvn.py
```

---

## Reproducing Results from the Paper

### Environment Setup (Narval / Compute Canada)

#### Load required modules

```bash
module load StdEnv/2020 gcc/11.3.0
module load python/3.10.2 cuda/11.7
```

#### Create and activate a virtual environment

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

### Input Design Grids

Design grids used in the manuscript are provided in `examples/input/`:

* `default_table_none_xall_t.csv`
  Correctly specified **non‑biomarker** effect parameters

* `default_table_weak_xall_t.csv`
  Correctly specified **biomarker** effect parameters

Custom design grids may be supplied, provided they follow the same column schema.

#### Input Design Grid Format

Input design grids must be provided as **CSV files** and loaded into a pandas `DataFrame`.
Each row corresponds to a single design point evaluated in the simulation.

##### Required Columns (all effect settings)

The following columns **must be present** in all input design grid files:
| Column name                   | Description                                |
| ----------------------------- | ------------------------------------------ |
| `t`                           | Interim analysis time point                |
| `r`                           | Subgroup proportion                        |
| `Is`                          | Total information units                    |
| `p_1`                         | Prior probability of subgroup effect       |
| `X1_t`                        | Interim statistic for overall population   |
| `X2_t`                        | Interim statistic for subgroup             |
| `delta`                       | Mean treatment effect (overall population) |
| **one of:** `d` **or** `dcof` | Subgroup effect specification              |

Internally, the interim statistics are combined as:

```python
X_t = [X1_t, X2_t]
```
---

### Reproducing Figure 2

The following steps reproduce:

```
examples/results/grid_power_vs_r_2x2_true_4.pdf
```

#### Step 1 — Run GPU simulations (SLURM)

From `examples/`, submit batch jobs (update the account name in the script as needed):

```bash
sbatch run_powerz_true.sh 0
sbatch run_powerz_true.sh 1
sbatch run_powerz_true.sh 2
sbatch run_powerz_true.sh 3
sbatch run_powerz_true.sh 4
```

Each job:

* Runs `powerz_example_true.py` for both **none** and **weak** effect settings
* Writes per‑job CSV outputs to `results/sub/` with filenames of the form:

```
t_Test_i<job>_<effect>_combined_results_01.csv
```

Batch behavior can be controlled via:

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

#### Step 2 — Combine per‑job outputs

```bash
python combine.py
```

Generated files:

* `results/t_Test_iall_none_combined_results.csv`
* `results/t_Test_iall_weak_combined_results.csv`

---

#### Step 3 — Generate Figure 2

```bash
python plot_fig2.py
```

Output:

```
results/grid_power_vs_r_2x2_true_4.pdf
```

---

## Notes

* `run_powerz_true.sh` requests **1 GPU**, approximately **1 GB RAM**, and **5:59:00** walltime.
* Key parameters in `powerz_example_true.py`:

  * `--effect_setting {0,1,2}` → none / weak / strong
  * `--num_batches (-nb)` → split design grid across jobs
  * `--test_iteration (-T)` → batch index
* `plot_fig2.py` assumes combined CSV files already exist (run `combine.py` first).
