# 基本运行
python main.py --dataset your_dataset --model deepseek_v3

# 指定参数
python code/remove_recon.py \
  --dataset polaris_2 \
  --model gpt-4o \
  --data_dir ./data/solve \
  --output_dir ./data/unsol \
  --prompt v4-remove-only \
  --temperature 0.0