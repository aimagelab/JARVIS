#!/bin/bash
#SBATCH --job-name=eval_cambrian
#SBATCH --output=./logs/%x-%j
#SBATCH --error=./logs/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --account=
#SBATCH --time=01:30:00
#SBATCH --array=0-22

set -e

# TODO: change this
conda activate jarvis
cd ~/git/JARVIS
export PYTHONPATH=.

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# TODO: Cache directories
export HF_HUB_CACHE=
export HF_DATASETS_CACHE=

checkpoint="$1"
conv_mode="$2"
eval_output_dir="${3:-./eval_cambrian_results}"
gpu_devices="${4:-0}"

echo "Checkpoint: $checkpoint"
echo "Conversation mode: $conv_mode"
echo "Evaluation output directory: $eval_output_dir"
echo "GPU devices: $gpu_devices"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

mkdir -p "$eval_output_dir"
echo "Created evaluation directory: $eval_output_dir"

export CUDA_VISIBLE_DEVICES="$gpu_devices"

# All Cambrian benchmarks
benchmarks=(
    gqa
    vizwiz
    scienceqa
    textvqa
    pope
    mme
    mmbench_en
    mmbench_cn
    seed
    # mmvet
    mmmu
    mathvista
    ai2d
    chartqa
    # docvqa
    # infovqa
    # stvqa
    ocrbench
    mmstar
    realworldqa
    qbench
    blink
    mmvp
    vstar
    ade
    omni
    coco
    # synthdog
)

benchmark=${benchmarks[$SLURM_ARRAY_TASK_ID]}

echo "Running benchmark: $benchmark (array index: $SLURM_ARRAY_TASK_ID)"
start_time=$(date +%s)
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Starting benchmark $benchmark at $timestamp"

benchmark_output_dir="${eval_output_dir}/${benchmark}"
mkdir -p "$benchmark_output_dir"

if bash scripts/eval/eval_cambrian_single_benchmark.sh \
    --benchmark "$benchmark" \
    --ckpt "$checkpoint" \
    --conv_mode "$conv_mode" \
    --output_dir "$benchmark_output_dir"; then
    echo "Successfully completed benchmark: $benchmark"
    
    # Create a completion marker file
    touch "${benchmark_output_dir}/.${checkpoint}_completed"
    
else
    echo "Error: Failed to complete benchmark: $benchmark"
    exit 1
fi

end_time=$(date +%s)
duration=$(( (end_time - start_time) / 60 ))
cur_timestamp=$(date +"%Y-%m-%d %H:%M:%S")

echo "Benchmark $benchmark completed in $duration minutes"
echo "Finished at: $cur_timestamp"

echo "Benchmark $benchmark evaluation completed successfully!"
echo "Results saved in: $benchmark_output_dir"