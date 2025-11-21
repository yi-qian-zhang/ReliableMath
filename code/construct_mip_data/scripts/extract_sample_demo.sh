# 基本用法 (默认保存为 CSV)
python extract_sample_demo.py -i 缺省一条.json -o output.csv

# 指定绝对路径
python extract_sample_demo.py -i /Users/yiqianzhang/data/source.json -o /Users/yiqianzhang/data/result.csv

# 保存为 JSON 格式
python code/construct_mip_data/scripts/extract_sample_demo.py  \
--input /data2/yiqianzhang/ReliableMath/data/DeepSeek-R1-Distill-Qwen-32B-8715/11-21/official_mode/missing_one/polaris_easy_100/validity_v6/polaris_easy_100_sample_valid_n1_zh.json  \
--output /data2/yiqianzhang/ReliableMath/data/DeepSeek-R1-Distill-Qwen-32B-8715/11-21/official_mode/missing_one/polaris_easy_100/validity_v6/缺省一条.json
