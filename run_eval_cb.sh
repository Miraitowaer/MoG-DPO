#!/bin/bash

# ================= 配置区域 =================
# 设置使用的 GPU 数量 (例如 8 卡就是 8, 4 卡就是 4)
WORLD_SIZE=8
LOG_DIR=/data/private/ExeCoder/cg_results/Deepseek-coder-6.7b-1epoch-code/codebleu_results
mkdir -p "$LOG_DIR"


# 如果你的 Python 环境需要激活，请取消注释下面这行
# source /path/to/your/venv/bin/activate
# ===========================================

echo "Starting evaluation with $WORLD_SIZE GPUs..."

# 循环启动子进程
for (( rank=0; rank<WORLD_SIZE; rank++ ))
do
    # 计算当前进程应该使用的 GPU ID (通常 rank 0 用 GPU 0, rank 1 用 GPU 1...)
    # 如果你的 GPU 编号不是连续的，可以在这里手动指定映射
    
    echo "Launching Rank $rank on GPU $rank..."
    
    # 核心命令：
    # 1. CUDA_VISIBLE_DEVICES=$rank : 限制该进程只能看到这张卡
    # 2. nohup ... & : 后台运行，防止 SSH 断开导致中断
    # 3. > log... : 保存日志方便查看报错
    
    CUDA_VISIBLE_DEVICES=$rank nohup python eval_codebleu_generation.py \
        --rank $rank \
        --world_size $WORLD_SIZE \
        > ${LOG_DIR}/eval_log_rank_${rank}.log 2>&1 &
        
    # 稍微停顿一下，防止瞬间同时加载模型导致 CPU/IO 飙升
    sleep 10
done

# 等待所有后台任务结束
echo "All processes launched. Waiting for them to finish..."
wait

echo "========================================================"
echo "Evaluation finished on all GPUs."
echo "Running merge script..."
echo "========================================================"

# 自动合并结果
# (请确保你保存了之前的 merge_final_results.py)
python merge_final_results.py