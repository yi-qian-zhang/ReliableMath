conda activate Interactive_R1
# 翻译 sample 数据集为中文版本
python code/construct_mip_data/scripts/translate_sample_data.py \
  --input data/DeepSeek-R1-Distill-Qwen-32B-8715/11-21/official_mode/missing_one/polaris_easy_100/validity_v6/polaris_normal_20_times_7/polaris_normal_20_times_7_sample_valid_n1.json \
  --output data/DeepSeek-R1-Distill-Qwen-32B-8715/11-21/official_mode/missing_one/polaris_easy_100/validity_v6/polaris_normal_20_times_7/polaris_normal_20_times_7_sample_valid_n1_zh.json \
  --model_url http://localhost:8717/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-32B-8717  \
  --threads 32


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

