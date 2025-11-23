cd /home/user/ReliableMath

# DeepSeek-R1-Distill-Qwen-7B-8715端口
python code/contradiction_construction/scripts/translate_sample_data.py \
  --input data/construct_contradiction/con_one/8716/11-22/polaris_easy_100/polaris_easy_100_sample_valid.json \
  --output data/construct_contradiction/con_one/8716/11-22/polaris_easy_100/polaris_easy_100_sample_valid_zh.json \
  --model_url http://localhost:8715/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-7B-8715 \
  --threads 16

# DeepSeek-R1-Distill-Qwen-7B-8716端口
python code/contradiction_construction/scripts/translate_sample_data.py \
  --input data/construct_contradiction/con_one/8716/11-22/polaris_easy_100/polaris_easy_100_sample_valid.json \
  --output data/construct_contradiction/con_one/8716/11-22/polaris_easy_100/polaris_easy_100_sample_valid_zh.json \
  --model_url http://localhost:8716/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-7B-8716 \
  --threads 16

# DeepSeek-R1-Distill-Qwen-7B-8717端口
python code/contradiction_construction/scripts/translate_sample_data.py \
  --input data/construct_contradiction/con_one/8716/11-22/polaris_easy_100/polaris_easy_100_sample_valid.json \
  --output data/construct_contradiction/con_one/8716/11-22/polaris_easy_100/polaris_easy_100_sample_valid_zh.json \
  --model_url http://localhost:8717/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-7B-8717 \
  --threads 16