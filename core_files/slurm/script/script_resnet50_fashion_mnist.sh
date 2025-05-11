#!/bin/bash

#SBATCH -J data_resnet           # Job name
#SBATCH -p gpu                # Partition name
#SBATCH --gres=gpu:a40-48:1  # GPUs
#SBATCH --ntasks=1            # Number of tasks (processes)
#SBATCH -t 5:00:00           # Time limit
#SBATCH --array=1          # Array job with tasks 1 to 20
#SBATCH --mem-per-gpu=20000M

# Enable core dumps for debugging
ulimit -c unlimited

# Load necessary module
module load tensorflow/2.16

# Print SLURM array task ID
echo "ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "GPU Info:"
nvidia-smi

# Run the main TensorFlow script with detailed logging
srun python resnet50_fashion_mnist.py > outfile_resnet50_fashion_mnist.out

