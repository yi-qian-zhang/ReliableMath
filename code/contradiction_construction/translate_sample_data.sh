cd /home/user/ReliableMath

python code/contradiction_construction/translate_sample_data.py \
  --input data/construct_contradiction/polaris_easy_100_sample_valid.json \
  --output data/construct_contradiction/polaris_easy_100_sample_valid_zh.json \
  --model_url http://localhost:8715/v1 \
  --model_name DeepSeek-R1-Distill-Qwen-32B-8715 \
  --threads 16