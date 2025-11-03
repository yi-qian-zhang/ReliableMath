# tmux新建会话并后台运行
tmux new -s vllm
tmux attach -t vllm
Ctrl + b，然后松开，再按 d
即可“detach”会话，返回到普通终端。

tmux new -s run_construct
tmux attach -t run_construct
#启动DeepSeek-R1-Distill-Qwen-7B 单卡
CUDA_VISIBLE_DEVICES=7 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 16384 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --port 8715

# DeepSeek-R1-Distill-Qwen-7B 双卡
CUDA_VISIBLE_DEVICES=6,7 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 16384 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8715

# 基础模式（仅启发式，免费）
python code/construct_mip_data/construct_mip_with_deepscaler.py \
    --dataset polaris_normal_5_times_8 \
    --model gpt-4o-mini \
    --verify_model DeepSeek-R1-Distill-Qwen-7B \
    --temperature 0.9 \
    --max_attempts 8 \
    --threads 16 \
    --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-03/polaris_normal_5_times_8 \
    --force

# 完整模式（启发式 + ORM 备份）
python code/construct_mip_data/construct_mip_with_deepscaler.py \
    --dataset polaris_easy_20 \
    --model gpt-4o-mini \
    --verify_model DeepSeek-R1-Distill-Qwen-7B \
    --temperature 0.9 \
    --max_attempts 8 \
    --use_math_orm \
    --threads 16 \
    --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-03/polaris_easy_20 \
    --force