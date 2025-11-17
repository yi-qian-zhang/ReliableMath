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



# 基础测试（推荐先运行这个）
# 1. 进入工作目录
conda activate Interactive_R1
cd /data2/yiqianzhang/ReliableMath

# 2. 测试模式（只处理前5个样本，缺省1个条件）
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_easy_20 \
  --num_missing 1 \
  --test_mode \
  --output_dir data/construct_mip_qwen_7B_16384/11-17 \
  --threads 32

# 3. 查看输出
ls -lh data/construct_mip_data/polaris_easy_20_*

# 正式运行
# 缺省 1 个条件（低难度）
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_easy_20 \
  --num_missing 1 \
  --threads 8

# 缺省 2 个条件（中难度）
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_easy_20 \
  --num_missing 2 \
  --threads 8

# 缺省 3 个条件（高难度）
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_easy_20 \
  --num_missing 3 \
  --threads 8

# 批量生成
# 一次生成多个难度级别
for n in 1 2 3; do
  python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
    --dataset polaris_easy_20 \
    --num_missing $n \
    --threads 8
done