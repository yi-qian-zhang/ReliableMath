python code/construct_mip_data/scripts/convert_polaris_data.py \

python code/construct_mip_data/scripts/convert_polaris_data.py \
  --count_7 30 \
  --output data/solve/polaris_easy_30.json

#生成难度平均分布的数据集demo
python convert_polaris_data.py \
  --input /home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl \
  --output /home/zhangyiqian/ReliableMath/data/solve/polaris_normal_70.json \
  --count_1 10 \
  --count_2 10 \
  --count_3 10 \
  --count_4 10 \
  --count_5 10 \
  --count_6 10 \
  --count_7 10 \