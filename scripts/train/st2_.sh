#!/bin/bash
#SBATCH --job-name=st2_
#SBATCH --output=./logs/%x-%j
#SBATCH --error=./logs/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=220G
#SBATCH --cpus-per-task=16
#SBATCH --partition=
#SBATCH --account=
#SBATCH --time=07:00:00

# TODO: change this
conda activate jarvis
cd ~/git/JARVIS

export PYTHONPATH=.
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=
export WANDB_PROJECT=
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

# TODO: change to the path of a checkpoint from stage 1, either from LLaVA or Jarvis
model_name="/path/to/your/checkpoint/folder/st1_llava_vicuna"
_model_name=$(basename $model_name)
run_name="${SLURM_JOB_NAME}${_model_name}"
output_dir="/path/to/your/checkpoint/folder/${run_name}"

# TODO: change this
conv_template=vicuna_v1
# conv_template=qwen2
# conv_template=mistral

# TODO: change this
train_data_path="/st2_llava/llava_v1_5_mix665k.json"
train_image_folder="/st2_llava"

((ws = $SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export WORLD_SIZE=$ws

srun -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
torchrun \
--nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --master-port=$MASTER_PORT --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
src/train/train.py \
--deepspeed deepspeed/zero3.json \
--gradient_checkpointing True \
--seed 42 \
--save_strategy steps \
--save_steps 500 \
--save_total_limit 1 \
--output_dir $output_dir \
--run_name $run_name \
--report_to wandb \
--model_name $model_name \
--attn_implementation sdpa \
--conv_template $conv_template \
--train_data_path $train_data_path \
--train_image_folder $train_image_folder \
--remove_unused_columns False \
--bf16 True \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--eval_strategy no \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 5 \
--tf32 True \
--dataloader_num_workers 4 \
--dataloader_pin_memory True \
--group_by_mod_lens True \
--image_aspect_ratio pad