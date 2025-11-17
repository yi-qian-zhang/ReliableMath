# tmux新建会话并后台运行
tmux new -s vllm-8715
tmux attach -t vllm-8715
tmux attach -t vllm-8717
tmux attach -t s1k
Ctrl + b，然后松开，再按 d
即可“detach”会话，返回到普通终端。

tmux new -s run_construct
tmux attach -t run_construct

tmux list-sessions

# A800
#启动DeepSeek-R1-Distill-Qwen-7B 单卡
CUDA_VISIBLE_DEVICES=7 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 16384 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --port 8715

CUDA_VISIBLE_DEVICES=055 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 8192 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8717

# DeepSeek-R1-Distill-Qwen-7B 双卡
CUDA_VISIBLE_DEVICES=6,7 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 8192 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8715

# DeepSeek-R1-Distill-Qwen-7B 4卡
# 1. 设置环境变量，禁用P2P
export NCCL_P2P_DISABLE=1
# 2. 运行你修正后的vLLM命令
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 8192 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --port 8715
# 1. 这会立即删除该变量
unset NCCL_P2P_DISABLE
# 2. （可选）验证它是否已消失
echo $NCCL_P2P_DISABLE
# (此时会输出一个空行)
    
# 基础模式（仅启发式，免费）
python code/construct_mip_data/construct_mip_with_deepscaler.py \
    --dataset polaris_normal_5_times_8 \
    --model gpt-4o-mini \
    --verify_model DeepSeek-R1-Distill-Qwen-7B-8717 \
    --temperature 1.0 \
    --max_attempts 8 \
    --threads 32 \
    --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-05/polaris_normal_5_times_8/deepscaler_extract \
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

# 完整模式（启发式 + ORM 备份），大规模线程 4卡VLLM
python code/construct_mip_data/construct_mip_with_deepscaler.py \
    --dataset polaris_normal_5_times_8 \
    --model gpt-4o-mini \
    --verify_model DeepSeek-R1-Distill-Qwen-7B \
    --temperature 1.0 \
    --max_attempts 8 \
    --threads 32 \
    --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_5_times_8/deepscaler_extract \
    --force

    python code/construct_mip_data/construct_mip_with_deepscaler.py \
    --dataset polaris_normal_600_times_8 \
    --model gpt-4o-mini \
    --verify_model DeepSeek-R1-Distill-Qwen-7B \
    --temperature 1.0 \
    --max_attempts 8 \
    --threads 32 \
    --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract \
    --force

# 测试多选题
python code/construct_mip_data/construct_mip_with_deepscaler.py \
    --dataset polaris_choose_que_demo \
    --model gpt-4o-mini \
    --verify_model DeepSeek-R1-Distill-Qwen-7B \
    --temperature 1.0 \
    --max_attempts 8 \
    --threads 32 \
    --output_dir /home/zhangyiqian/ReliableMath/data/polaris_choose_que_demo \
    --force

python code/construct_mip_data/construct_mip_with_deepscaler.py \
    --dataset polaris_diff_6_150 \
    --model gpt-4o-mini \
    --verify_model DeepSeek-R1-Distill-Qwen-7B \
    --temperature 1.0 \
    --max_attempts 8 \
    --threads 32 \
    --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract \
    --force

# A100
#启动DeepSeek-R1-Distill-Qwen-7B 单卡
CUDA_VISIBLE_DEVICES=7 vllm /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 16384 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --port 8715

#启动DeepSeek-R1-Distill-Qwen-7B 双卡
CUDA_VISIBLE_DEVICES=1,2 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 8192 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8715

CUDA_VISIBLE_DEVICES=1,2 vllm serve \
    --model /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 8192 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8715
    
# DeepSeek-R1-Distill-Qwen-7B 双卡
CUDA_VISIBLE_DEVICES=0,5 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 8192 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8717