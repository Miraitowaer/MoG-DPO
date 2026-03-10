#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# 显存检查
nvidia-smi

# 路径配置
ROOT="/data/private/ExeCoder"
SFT_MODEL_PATH="/data/private/ExeCoder/cg_results/Deepseek-coder-6.7b-1epoch-code"
DATA_PATH="/data/private/ExeCoder/data/dpo_train_qwen_clean.json"
OUTPUT_DIR="/data/private/ExeCoder/cg_results/dpo_deepseek"

# 创建日志目录
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S-%N" # current time
}

LOG_PATH=$ROOT/cg_loginfo/dpo_$(timestamp)-$RANDOM
mkdir -p "$LOG_PATH"

echo "Starting DPO Training..."
echo "Model: $SFT_MODEL_PATH"

# 启动训练
# 关键修改：
# 1. 使用 zero3 配置文件 (你上传的那个)
# 2. 开启 gradient_checkpointing
# 3. 使用 bf16 (L40 支持)
# 4. 调整学习率 (DPO 通常比 SFT 低)
deepspeed --master_port=29501 src/train_dpo.py \
    --model_name_or_path $SFT_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --max_length 2049 \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --remove_unused_columns False \
    --warmup_ratio 0.1 \
    --logging_steps 5 \
    --save_steps 50 \
    --beta 0.1 \
    --deepspeed src/configs/deepspeed_config_zero3.json \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    > "$LOG_PATH/out.txt" 2>&1

#debug
# deepspeed --master_port=29501 src/train_dpo.py \
#     --model_name_or_path $SFT_MODEL_PATH \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 5 \
#     --max_length 2048 \
#     --max_prompt_length 1024 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 5e-7 \
#     --warmup_ratio 0.1 \
#     --logging_steps 1 \
#     --save_steps 5 \
#     --beta 0.1 \
#     --deepspeed src/configs/deepspeed_config_zero3.json \
#     --bf16 True \
#     --gradient_checkpointing True \
#     --report_to "tensorboard" \
#     > "$LOG_PATH/out.txt" 2>&1

echo "Done. Log saved to $ROOT/loginfo/dpo_train.log"