#!/bin/bash
set -e

echo "> eval_cambrian_single_benchmark.sh $@"

################# Parse Arguments #################

conv_mode=""
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --ckpt)
        ckpt="$2"
        shift 2
        ;;
    --conv_mode)
        conv_mode="$2"
        shift 2
        ;;
    --output_dir)
        output_dir="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

# Check if the required arguments are provided
if [[ -z "$benchmark" || -z "$ckpt" || -z "$output_dir" ]]; then
  echo "Error: --benchmark, --ckpt, and --output_dir arguments are required."
  exit 1
fi

# Print the values
echo "Benchmark: $benchmark"
echo "Checkpoint path: $ckpt"
echo "Conversation mode: $conv_mode"
echo "Output directory: $output_dir"

################# Process Arguments #################

# get the dir of this script and set up paths
script_dir=$(dirname $(realpath $0))
echo "Script dir: ${script_dir}"
src_root="$script_dir/../.."
echo "src_root: $(realpath $src_root)"
cambrian_eval_dir="$src_root/cambrian/eval"

# Ensure PYTHONPATH encludes src/*
export PYTHONPATH="$src_root:$src_root/cambrian:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

cd "$cambrian_eval_dir"
echo "Changed to directory: $(pwd)"

benchmark_dir="./eval/${benchmark}"
eval_file="${benchmark_dir}/${benchmark}_eval.py"
test_file="${benchmark_dir}/${benchmark}_test.py"

# verify that the benchmark exists and the eval/test files exist
if [ ! -d $benchmark_dir ]; then
    echo "Error: Benchmark directory $benchmark_dir does not exist."
    exit 1
fi
if [ ! -f $eval_file ]; then
    echo "Error: Eval file $eval_file does not exist."
    exit 1
fi
if [ ! -f $test_file ]; then
    echo "Error: Test file $test_file does not exist."
    exit 1
fi

# extract basename for answers file naming
model_basename=$(basename $ckpt)
answers_file="${output_dir}/answers_${model_basename}.jsonl"

# The evaluation script will create the directory structure based on answers_file path

################# Handle Chunking #################
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo "Running with $CHUNKS chunks"

################# Run Evaluation #################

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo "Starting evaluation at: $current_date_time";

for IDX in $(seq 0 $((CHUNKS-1))); do
    {
        CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python $eval_file \
            --model_path "$ckpt" \
            --num_chunks "$CHUNKS" \
            --chunk_idx "$IDX" \
            --answers_file "$answers_file" \
            --conv_mode "$conv_mode"
    } &
done

wait

################# Combine Results #################

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo "Combining results at: $current_date_time";

# check if the answers_file already exists. if it does, move it to a backup file with the current timestamp
if [ -f "$answers_file" ]; then
    mv "$answers_file" "${answers_file}.bak.$(date +%s)"
    echo "Moved existing answers file to ${answers_file}.bak.$(date +%s)"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    # The eval script creates chunk files in the same directory as the main answers file
    idx_file="${output_dir}/answers_${model_basename}_${IDX}.jsonl"
    if [ -f "$idx_file" ]; then
        cat "$idx_file" >> "$answers_file"
        rm "$idx_file"
    else
        echo "Warning: Expected chunk file $idx_file not found"
    fi
done

################# Run Testing #####################

# Change to benchmark directory for testing (some tests expect to be run from there)
cd $benchmark_dir

# Create a temporary symlink to the answers file for the test script
temp_answers_file="./answers/answers_${model_basename}.jsonl"
mkdir -p "./answers"
ln -sf "$(realpath $answers_file)" "$temp_answers_file"

# Run the test
python "${benchmark}_test.py" --answers_file "$temp_answers_file"

# Copy the generated experiments.csv to the output directory
if [ -f "./experiments.csv" ]; then
    cp "./experiments.csv" "$output_dir/"
    echo "Copied experiments.csv to $output_dir/"
else
    echo "Warning: experiments.csv not found in $benchmark_dir"
fi

# Clean up temporary symlink
rm -f "$temp_answers_file"

echo "Done evaluation and testing for $benchmark on model at $ckpt with conversation mode $conv_mode"
echo "Results saved to: $output_dir"
echo "Answers file: $(realpath $answers_file)"