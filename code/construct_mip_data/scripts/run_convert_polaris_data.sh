python code/construct_mip_data/scripts/convert_polaris_data.py \

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --count_7 30 \
  --output data/solve/polaris_easy_30.json

#生成难度平均分布的数据集demo
python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_normal_80*8.json \
  --count_1 80 \
  --count_2 80 \
  --count_3 80 \
  --count_4 80 \
  --count_5 80 \
  --count_6 80 \
  --count_7 80 

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_normal_5*8.json \
  --count_1 5 \
  --count_2 5 \
  --count_3 5 \
  --count_4 5 \
  --count_5 5 \
  --count_6 5 \
  --count_7 5 \

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_normal_10times7.json \
  --count_1 10 \
  --count_2 10 \
  --count_3 10 \
  --count_4 10 \
  --count_5 10 \
  --count_6 10 \
  --count_7 10 \

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_easy_40.json \
  --count_1 0 \
  --count_2 0 \
  --count_3 0 \
  --count_4 0 \
  --count_5 0 \
  --count_6 20 \
  --count_7 20 \

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_hard_40.json \
  --count_1 20 \
  --count_2 20 \
  --count_3 0 \
  --count_4 0 \
  --count_5 0 \
  --count_6 0 \
  --count_7 0 \

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_normal_800*8.json \
  --count_1 800 \
  --count_2 800 \
  --count_3 800 \
  --count_4 800 \
  --count_5 800 \
  --count_6 800 \
  --count_7 800 