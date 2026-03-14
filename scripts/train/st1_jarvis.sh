#!/bin/bash
#SBATCH --job-name=st1_jarvis
#SBATCH --output=./logs/%x-%j
#SBATCH --error=./logs/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --partition=
#SBATCH --account=
#SBATCH --time=03:30:00

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

# TODO: change this
run_name="${SLURM_JOB_NAME}"
output_dir="/path/to/your/checkpoint/folder/${run_name}"

language_model_name="lmsys/vicuna-7b-v1.5"
# language_model_name="Qwen/Qwen2-7B-Instruct"
# language_model_name="mistralai/Ministral-8B-Instruct-2410"

vision_model_name="openai/clip-vit-large-patch14-336"
# vision_model_name="google/siglip-so400m-patch14-384"

vision_layer_idx=-2
# vision_layer_idx=-1 # for SigLIP2
skip_left_visual_tokens=1
# skip_left_visual_tokens=0 # for SigLIP2, as it does not have any CLS token

# jepa_llm_layer_idx = num_hidden_layers / 4
jepa_llm_layer_idx=8 # Vicuna-7B
# jepa_llm_layer_idx=7 # Qwen2-7B
# jepa_llm_layer_idx=9 # Ministral-8B

# TODO: change this
train_data_path="/st1_llava/blip_laion_cc_sbu_558k.json"
train_image_folder="/st1_llava"

((ws = $SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export WORLD_SIZE=$ws

dataloader_num_workers=4

srun -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
torchrun \
--nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --master-port=$MASTER_PORT --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
src/train/train.py \
--deepspeed deepspeed/zero2.json \
--gradient_checkpointing True \
--seed 42 \
--save_strategy steps \
--save_steps 1000 \
--save_total_limit 1 \
--output_dir $output_dir \
--run_name $run_name \
--report_to wandb \
--language_model_name $language_model_name \
--attn_implementation sdpa \
--vision_model_name $vision_model_name \
--vision_layer_idx $vision_layer_idx \
--projector_type mlp \
--projector_bias True \
--bidir_visual_attn True \
--conv_template plain \
--train_proj_only \
--train_data_path $train_data_path \
--train_image_folder $train_image_folder \
--remove_unused_columns False \
--bf16 True \
--num_train_epochs 1 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--eval_strategy "no" \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 5 \
--tf32 True \
--dataloader_num_workers 4 \
--dataloader_pin_memory True \
--tgt_vision_model_name "facebook/dinov2-large" \
--tgt_skip_left_visual_tokens 1 \
--tgt_vision_layer_idx -1 \
--tgt_proj_type mlp \
--jepa_loss \
--jepa_loss_fn cos_sim \
--jepa_llm_layer_idx $jepa_llm_layer_idx \
--jepa_lambda 0.2