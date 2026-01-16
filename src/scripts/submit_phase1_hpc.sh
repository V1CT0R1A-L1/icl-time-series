#!/bin/bash
# HPC Job Submission Script for Phase 1 Experiments
# Usage: ./submit_phase1_hpc.sh [start_k] [end_k]
# Example: ./submit_phase1_hpc.sh 2 10  # Submit jobs for K=2 through K=10

START_K=${1:-1}
END_K=${2:-10}

# Generate configs first
echo "Generating configs..."
python src/scripts/phase1_sweep.py

# Create logs directory
mkdir -p logs

# Submit jobs for each K value
for k in $(seq $START_K $END_K); do
    if [ -f "src/conf/phase1/k${k}.yaml" ]; then
        echo "Submitting job for K=${k}..."
        
        # Create job script
        JOB_SCRIPT="src/scripts/job_k${k}.sh"
        cat > $JOB_SCRIPT << EOF
#!/bin/bash
#SBATCH --job-name=phase1_k${k}
#SBATCH --output=logs/phase1_k${k}_%j.out
#SBATCH --error=logs/phase1_k${k}_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load modules (adjust for your HPC system)
module load python/3.9
source activate icl-time-series  # Your conda environment

# Run experiment
cd \$SLURM_SUBMIT_DIR
python src/train.py --config src/conf/phase1/k${k}.yaml
EOF
        
        chmod +x $JOB_SCRIPT
        sbatch $JOB_SCRIPT
        echo "  Job script: $JOB_SCRIPT"
    else
        echo "  Warning: Config not found for K=${k}"
    fi
done

echo "All jobs submitted. Check status with: squeue -u \$USER"
