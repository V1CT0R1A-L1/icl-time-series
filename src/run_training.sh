#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p mit_normal
#SBATCH --mem=32G
#SBATCH -J transformer_mixture_large
#SBATCH -o training_large_%j.out
#SBATCH -e training_large_%j.err

echo "Job ID: $SLURM_JOB_ID"

source in-context-learning-env/bin/activate

echo "✅ Virtualenv activated"
echo "Python patcdh: $(which python)"
echo "Python version: $(python --version)"

cd src
python train.py --config conf/multi_context_large.yaml

echo "Training completed"
