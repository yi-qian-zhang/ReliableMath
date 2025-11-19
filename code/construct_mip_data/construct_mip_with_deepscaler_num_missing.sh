# tmux新建会话并后台运行
tmux new -s vllm-8715
tmux attach -t vllm-8715
tmux attach -t vllm-8717
tmux attach -t s1k
Ctrl + b，然后松开，再按 d
即可“detach”会话，返回到普通终端。

tmux new -s run_construct
tmux attach -t run_construct

tmux list-sessions

# A100
#启动DeepSeek-R1-Distill-Qwen-7B 单卡
CUDA_VISIBLE_DEVICES=5 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --served-model-name DeepSeek-R1-Distill-Qwen-32B \
    --max-model-len 16384 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --port 8716


#启动DeepSeek-R1-Distill-Qwen-7B 双卡
CUDA_VISIBLE_DEVICES=0,1 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --max-model-len 8192 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8715


#启动DeepSeek-R1-Distill-Qwen-32B 双卡
CUDA_VISIBLE_DEVICES=0,1 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --served-model-name DeepSeek-R1-Distill-Qwen-32B \
    --max-model-len 8192 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.98 \
    --port 8715

CUDA_VISIBLE_DEVICES=2 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --served-model-nam DeepSeek-R1-Distill-Qwen-32B \
    --max-model-len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --port 8717
CUDA_VISIBLE_DEVICES=3 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --served-model-nam DeepSeek-R1-Distill-Qwen-32B \
    --max-model-len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --port 8719

CUDA_VISIBLE_DEVICES=6,7 vllm serve /data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --served-model-nam DeepSeek-R1-Distill-Qwen-32B \
    --max-model-len 16384 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --port 8719

🚀 运行命令
1️⃣ 测试模式（推荐先跑这个）

cd /data2/yiqianzhang/ReliableMath
conda activate Interactive_R1

#8715端口 DeepSeek-R1-Distill-Qwen-7B
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_easy_20 \
  --num_missing 1 \
  --extract_model gpt-4o-mini \
  --rewrite_model DeepSeek-R1-Distill-Qwen-32B-8715 \
  --verify_model DeepSeek-R1-Distill-Qwen-32B-8715 \
  --judge_model gpt-4o-mini \
  --threads 32 \
  --output_dir data/DeepSeek-R1-Distill-Qwen-32B/11-18/test_mode/missing_one/19-19 \
  --test_mode \
  --force

python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_easy_20 \
  --num_missing 2 \
  --extract_model gpt-4o-mini \
  --rewrite_model DeepSeek-R1-Distill-Qwen-32B-8717 \
  --verify_model DeepSeek-R1-Distill-Qwen-32B-8717 \
  --judge_model gpt-4o-mini \
  --threads 32 \
  --output_dir data/DeepSeek-R1-Distill-Qwen-32B/11-18/test_mode/missing_two/19-19 \
  --test_mode \
  --force


python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_easy_20 \
  --num_missing 3 \
  --extract_model gpt-4o-mini \
  --rewrite_model DeepSeek-R1-Distill-Qwen-32B-8719 \
  --verify_model DeepSeek-R1-Distill-Qwen-32B-8719 \
  --judge_model gpt-4o-mini \
  --threads 32 \
  --output_dir data/DeepSeek-R1-Distill-Qwen-32B/11-18/test_mode/missing_three/19-19 \
  --test_mode \
  --force



2️⃣ 完整运行（测试通过后）
去掉 --test_mode 和 --force 运行完整数据集：

#8715端口 DeepSeek-R1-Distill-Qwen-32B
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_normal_10times7 \
  --num_missing 1 \
  --extract_model gpt-4o-mini \
  --rewrite_model DeepSeek-R1-Distill-Qwen-32B-8715 \
  --verify_model DeepSeek-R1-Distill-Qwen-32B-8715 \
  --judge_model gpt-4o-mini \
  --threads 32 \
  --output_dir data/DeepSeek-R1-Distill-Qwen-32B-8715/11-19/official_mode/missing_one/polaris_normal_10times7 \
  --force


#8717端口 DeepSeek-R1-Distill-Qwen-32B
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_normal_20_times_7 \
  --num_missing 2 \
  --extract_model gpt-4o-mini \
  --rewrite_model DeepSeek-R1-Distill-Qwen-32B-8717 \
  --verify_model DeepSeek-R1-Distill-Qwen-32B-8717 \
  --judge_model gpt-4o-mini \
  --threads 16 \
  --output_dir data/DeepSeek-R1-Distill-Qwen-32B/11-19/official_mode/missing_two/data/solve/polaris_normal_20_times_7 \
  --force


#8719端口 DeepSeek-R1-Distill-Qwen-32B
python code/construct_mip_data/construct_mip_with_deepscaler_num_missing.py \
  --dataset polaris_normal_20_times_7 \
  --num_missing 3 \
  --extract_model gpt-4o-mini \
  --rewrite_model DeepSeek-R1-Distill-Qwen-32B-8719 \
  --verify_model DeepSeek-R1-Distill-Qwen-32B-8719 \
  --judge_model gpt-4o-mini \
  --threads 16 \
  --output_dir data/DeepSeek-R1-Distill-Qwen-32B/11-19/official_mode/missing_three/data/solve/polaris_normal_20_times_7 \
  --force