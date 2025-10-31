# 正式运行（20 条数据）
python code/construct_mip_data/construct_mip_sampling8.py \
  --dataset polaris_easy_20 \
  --model gpt-4o-mini \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --judge_model gpt-4o-mini \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_sampling8/10-28/polaris_easy_20 \
  --threads 4 \
  --force

# 正式运行（40 条数据）
python code/construct_mip_data/construct_mip_sampling8.py \
  --dataset polaris_hard_40 \
  --model gpt-4o-mini \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --judge_model gpt-4o-mini \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_sampling8/10-29 \
  --threads 4 \
  --force