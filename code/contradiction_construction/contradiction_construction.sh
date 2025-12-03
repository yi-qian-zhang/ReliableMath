cd ~/ReliableMath

# 基础运行（使用本地模型）
python code/contradiction_construction.py \
  --dataset aime \
  --analysis_model DeepSeek-R1-Distill-Qwen-7B \
  --verify_model DeepSeek-R1-Distill-Qwen-7B \
  --temperature 1.0 \
  --max_attempts 8

# 测试模式
python code/contradiction_construction.py \
  --dataset aime \
  --test_mode

# 启用ORM验证（更准确但更贵）
python code/contradiction_construction.py \
  --dataset aime \
  --use_math_orm

使用方法：
conda activate Interactive_R1
cd /data2/yiqianzhang/ReliableMath

#8715-32B端口 DeepSeek-R1-Distill-Qwen-32B
python code/contradiction_construction/contradiction_construction.py \
  --analysis_model  DeepSeek-R1-Distill-Qwen-32B-8716\
  --verify_model   DeepSeek-R1-Distill-Qwen-32B-8716\
  --judge_model    DeepSeek-R1-Distill-Qwen-32B-8716\
  --dataset polaris_easy_100 \
  --threads 4   

#8716-7B端口 DeepSeek-R1-Distill-Qwen-7B
python code/contradiction_construction/contradiction_construction.py \
  --analysis_model  DeepSeek-R1-Distill-Qwen-7B-8716\
  --verify_model   DeepSeek-R1-Distill-Qwen-7B-8716\
  --judge_model    DeepSeek-R1-Distill-Qwen-7B-8716\
  --dataset polaris_easy_100 \
  --output_dir data/construct_contradiction/con_one/8716/11-22/polaris_easy_100  \
  --threads 16  
  