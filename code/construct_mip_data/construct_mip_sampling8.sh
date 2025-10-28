# 正式运行（40 条数据）
python code/construct_mip_data/construct_mip_sampling8.py \
  --dataset polaris_easy_40 \
  --model gpt-4o-mini \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --judge_model gpt-4o-mini \
  --temperature 0.9 \
  --max_attempts 8 \
  --output_dir /home/zhangyiqian/ReliableMath/data/construct_mip_sampling8/10-28 \
  --threads 4 \
  --force