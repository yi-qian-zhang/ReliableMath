python code/construct_mip_data/scripts/convert_polaris_data.py \

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --count_7 30 \
  --output data/solve/polaris_easy_30.json

#生成难度平均分布的数据集demo
python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_normal_50*8.json \
  --count_1 80 \
  --count_2 80 \
  --count_3 80 \
  --count_4 80 \
  --count_5 80 \
  --count_6 80 \
  --count_7 80 

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_normal_2*8.json \
  --count_1 2 \
  --count_2 2 \
  --count_3 2 \
  --count_4 2 \
  --count_5 2 \
  --count_6 2 \
  --count_7 2 \

python convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_hard_40.json \
  --count_1 10 \
  --count_2 10 \
  --count_3 10 \
  --count_4 10 \
  --count_5 0 \
  --count_6 0 \
  --count_7 0 \