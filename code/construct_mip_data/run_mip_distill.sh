#启动DeepSeek-R1-Distill-Qwen-7B和Qwen3-32B的vllm服务
CUDA_VISIBLE_DEVICES=0,1 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 16384 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8715

#QWEN3-8B
CUDA_VISIBLE_DEVICES=2,3 vllm serve /shared/models/Qwen3-8B \
    --served-model-name Qwen3-8B \
    --max-model-len 16384 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8717


#启动Qwen3-32B的vllm服务
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve /shared/models/Qwen3-32B \
    --served-model-name Qwen3-32B \
    --max-model-len 16384 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9 \
    --port 8716
    
# 测试模式
python code/construct_mip_data/construct_mip_distill.py \
  --dataset polaris_easy_20 \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --model gpt-4o \
  --judge_model gpt-4o \
  --test_mode \
  --force

# 正式运行
cd ~/ReliableMath

# DeepSeek-R1-Distill-Qwen-7B 版本
python code/construct_mip_data/construct_mip_distill.py \
  --dataset polaris_easy_40 \
  --model gpt-4o-mini \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --model gpt-4o \
  --judge_model gpt-4o \
  --prompt_dir prompt/construct_mip_distill \
  --output_dir data/construct_mip_qwen_7B_16384/10-28/easy_40 \
  --force

# QWEN3-32B 版本
python code/construct_mip_data/construct_mip_distill.py \
  --dataset polaris_easy_20 \
  --verify_model Qwen3-32B \
  --model gpt-4o \
  --judge_model gpt-4o \
  --prompt_dir prompt/construct_mip_distill \
  --output_dir data/construct_mip_qwen3_32B/ \
  --force