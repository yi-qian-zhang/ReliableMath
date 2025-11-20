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

python code/contradiction_construction/contradiction_construction.py \
  --dataset polaris_20 \
  --prompt_dir /data2/yiqianzhang/ReliableMath/prompt/contradict_data \
  --test_mode