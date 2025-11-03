# 使用默认配置（每个难度 100 条）
python extract_valid_data.py --input /home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/10-30/normal_80*8/multi/polaris_normal_80*8_valid.json --output valid_extracted_100*7.json

# 自定义每个难度的数量
python extract_valid_data.py \
    --input valid_data.json \
    --output valid_extracted.json \
    --count_1 50 \
    --count_2 80 \
    --count_3 100 \
    --count_4 100 \
    --count_5 120 \
    --count_6 100 \
    --count_7 50

# 只提取某些难度
python extract_valid_data.py \
    --count_1 100 \
    --count_2 100 \
    --count_3 0 \
    --count_4 0 \
    --count_5 0 \
    --count_6 0 \
    --count_7 0