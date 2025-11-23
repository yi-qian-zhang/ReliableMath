conda activate Interactive_R1
# 翻译 sample 数据集为中文版本
python code/construct_mip_data/scripts/translate_sample_data.py \
  --input data/construct_mip/missiong_one/8715/11-22/polaris_easy_100/polaris_easy_100_sample_valid_n1.json \
  --output data/construct_mip/missiong_one/8715/11-22/polaris_easy_100/polaris_easy_100_sample_valid_n1_zh.json \
  --model_url http://localhost:8715/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-7B-8715  \
  --threads 16

python code/construct_mip_data/scripts/translate_sample_data.py \
  --input data/construct_mip/missiong_two/8717/11-22/polaris_easy_100/polaris_easy_100_sample_valid_n2.json \
  --output data/construct_mip/missiong_two/8717/11-22/polaris_easy_100/polaris_easy_100_sample_valid_n2_zh.json \
  --model_url http://localhost:8717/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-7B-8717  \
  --threads 16


python code/construct_mip_data/translate_sample_data.py \
  --input /data2/yiqianzhang/ReliableMath/data/DeepSeek-R1-Distill-Qwen-32B/11-18/official_mode/missing_two/polaris_normal_10times7/polaris_normal_10times7_sample_valid_n2.json \
  --output /data2/yiqianzhang/ReliableMath/data/DeepSeek-R1-Distill-Qwen-32B/11-18/official_mode/missing_two/polaris_normal_10times7/polaris_normal_10times7_sample_valid_n2_zh.json \
  --model_url http://localhost:8716/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-32B  \
  --threads 16

python code/construct_mip_data/translate_sample_data.py \
  --input data/DeepSeek-R1-Distill-Qwen-32B/11-18/official_mode/missing_three/polaris_normal_10times7/polaris_normal_10times7_sample_valid_n3.json \
  --output data/DeepSeek-R1-Distill-Qwen-32B/11-18/official_mode/missing_three/polaris_normal_10times7/polaris_normal_10times7_sample_valid_n3_zh.json \
  --model_url http://localhost:8716/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-32B  \
  --threads 16

# 统计Token
python code/construct_mip_data/scripts/translate_sample_data.py \
  --input data/construct_mip/missiong_one/8715/11-22/polaris_easy_100/token_count/polaris_easy_100_sample_valid_n1.json \
  --output data/construct_mip/missiong_one/8715/11-22/polaris_easy_100/token_count/polaris_easy_100_sample_valid_n1_zh.json \
  --model_url http://localhost:8715/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-7B-8715  \
  --threads 16
