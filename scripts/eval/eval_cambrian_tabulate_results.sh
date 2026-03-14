#!/bin/bash

# TODO: change this
cambrian_root=~/git/JARVIS

model_name="$1"
eval_output_dir="$2"
echo "Model name: ${model_name}"

# All Cambrian benchmarks (should match the array in the main script)
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
    mmmu
    mathvista
    ai2d
    chartqa
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
)

echo "Checking status of Cambrian benchmarks for model: $model_name"
echo "=========================================================="

completed=0
failed=0
running=0
pending=0

for i in "${!benchmarks[@]}"; do
    benchmark="${benchmarks[$i]}"
    benchmark_output_dir="${eval_output_dir}/${benchmark}"
    
    # if [[ -f "${benchmark_output_dir}/.${model_name}_completed" ]]; then
    if [[ -f "${benchmark_output_dir}/.${model_name}_completed" ]]; then
        echo "[$i] $benchmark: ✅ COMPLETED"
        ((completed++))
    elif [[ -d "$benchmark_output_dir" ]]; then
        echo "[$i] $benchmark: 🔄 RUNNING/FAILED (check logs)"
        ((running++))
    else
        echo "[$i] $benchmark: ⏳ PENDING"
        ((pending++))
    fi
done

echo ""
echo "Summary:"
echo "========="
echo "Completed: $completed/${#benchmarks[@]}"
echo "Running/Failed: $running"
echo "Pending: $pending"

# Check if all benchmarks are completed
if [[ $completed -eq ${#benchmarks[@]} ]]; then
    echo ""
    echo "🎉 All benchmarks completed!"
    echo ""
    echo "Running tabulation..."
fi

# Run tabulation
cd $cambrian_root
python cambrian/eval/scripts/tabulate.py \
    --eval_dir "$eval_output_dir" \
    --experiment_csv experiments.csv \
    --out_pivot "${eval_output_dir}/cambrian_results.xlsx"
echo "Results tabulated in: ${eval_output_dir}/cambrian_results.xlsx"    