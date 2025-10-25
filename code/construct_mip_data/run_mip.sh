# 测试模式（前5个问题）
python code/construct_mip_data/construct_mip.py \
  --dataset polaris_easy_20 \
  --test_mode \
  --force

# 正式运行
python code/construct_mip_data/construct_mip.py \
  --dataset polaris_easy_20 \
  --output_dir data/construct_mip_data/2025_10_20 \
  --force

# 使用自定义参数
python code/construct_mip_data/construct_mip.py \
  --dataset polaris_easy_20 \
  --model gpt-4o \
  --verify_model gpt-4o \
  --force

python code/construct_mip_data/construct_mip.py --force

CUDA_VISIBLE_DEVICES=7 vllm serve /shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 4096 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --port 8715
