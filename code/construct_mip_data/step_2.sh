python code/construct_mip_data/step_2.py \
  --dataset polaris_easy_20 \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --temperature 0.9 \
  --max_attempts 8 \
  --force

python code/construct_mip_data/step_2.py \
  --dataset polaris_easy_20 \
  --verify_model Qwen3-8B \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/Qwen3-8B  \
  --force