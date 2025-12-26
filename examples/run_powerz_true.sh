#!/bin/bash
#SBATCH --account=your-acount         # Account name
#SBATCH --job-name=fig2_true
#SBATCH --tasks-per-node=1            # Number of parallel tasks per node
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --gpus-per-node=1             # One GPU per node
#SBATCH --mem=15000M                  # Request 15GB memory for the node
#SBATCH --time=1:59:00                # Set job time limit

# Load necessary modules
module load StdEnv/2020 gcc/11.3.0
module load python/3.10.2 cuda/11.7

# Set up the virtual environment if not already created
ENV_DIR="$HOME/environments/gbayesdesign"
if [ ! -d "$ENV_DIR" ]; then
    python -m venv "$ENV_DIR"
fi

# Activate the virtual environment
source "$ENV_DIR/bin/activate"

# Navigate to the examples directory
cd ~/bayesian-design4a/examples

# Display job index passed as an argument
echo "Job index: $1"

# Run Python scripts in parallel

python powerz_example_true.py -i input/default_table_none_xall_t.csv --effect_setting 0 -tt t_ -nb 5 -T "$1" &
python powerz_example_true.py -i input/default_table_weak_xall_t.csv --effect_setting 1 -tt t_ -nb 5 -T "$1" &

# Wait for all background jobs to finish
wait
