conda activate Interactive_R1
# 翻译 sample 数据集为中文版本
python code/construct_mip_data/translate_sample_data.py \
  --input data/DeepSeek-R1-Distill-Qwen-32B/11-19/official_mode/missing_one/19-19/polaris_easy_20_sample_valid_n1.json \
  --output data/DeepSeek-R1-Distill-Qwen-32B/11-19/official_mode/missing_one/19-19/polaris_easy_20_sample_valid_n1_zh.json \
  --model_url http://localhost:8716/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-32B  \
  --threads 16


python code/construct_mip_data/translate_sample_data.py \
  --input data/DeepSeek-R1-Distill-Qwen-32B/11-19/official_mode/missing_one/19-19/polaris_easy_20_sample_valid_n1.json \
  --output data/DeepSeek-R1-Distill-Qwen-32B/11-19/official_mode/missing_one/19-19/polaris_easy_20_sample_valid_n1_zh.json \
  --model_url http://localhost:8716/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-32B  \
  --threads 16