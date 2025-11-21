🚀 运行命令
1️⃣ 测试模式（推荐先跑这个）

cd /data2/yiqianzhang/ReliableMath
conda activate Interactive_R1

#8715端口 DeepSeek-R1-Distill-Qwen-7B
python code/construct_mip_data/construct_mip_final.py \
  --dataset polaris_easy_50 \
  --num_missing 1 \
  --extract_model gpt-4o-mini \
  --rewrite_model DeepSeek-R1-Distill-Qwen-32B-8716 \
  --verify_model DeepSeek-R1-Distill-Qwen-32B-8716 \
  --judge_model gpt-4o-mini \
  --threads 16 \
  --output_dir data/construct_mip_final/missing_one/11-21 \
  --test_mode \
  --force