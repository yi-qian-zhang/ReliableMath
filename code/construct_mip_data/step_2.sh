#启动DeepSeek-R1-Distill-Qwen-7B
CUDA_VISIBLE_DEVICES=0,1 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 16384 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8715

#QWEN3-8B 双卡
CUDA_VISIBLE_DEVICES=4,7 vllm serve /shared/models/Qwen3-8B \
    --served-model-name Qwen3-8B \
    --max-model-len 16384 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8717
#QWEN3-8B 单卡
CUDA_VISIBLE_DEVICES=7 vllm serve /shared/models/Qwen3-8B \
    --served-model-name Qwen3-8B \
    --max-model-len 16384 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --port 8717

#运行启动DeepSeek-R1-Distill-Qwen-7B的step_2脚本
python code/construct_mip_data/step_2.py \
  --dataset polaris_easy_20 \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/10-27  \
  --force

#运行启动Qwen3-8B的step_2脚本
python code/construct_mip_data/step_2.py \
  --dataset polaris_easy_20 \
  --verify_model Qwen3-8B \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/Qwen3-8B  \
  --force


python code/construct_mip_data/step_2_new.py \
  --dataset polaris_easy_40 \
  --model gpt-4o-mini \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/10-28/easy_40 \
  --force


python code/construct_mip_data/step_2_new.py \
  --dataset polaris_easy_20 \
  --verify_model Qwen3-8B \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/Qwen3-8B/10-28  \
  --force