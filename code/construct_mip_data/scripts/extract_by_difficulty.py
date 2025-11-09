import json
from collections import defaultdict
import random

def extract_by_difficulty(input_file, output_file, difficulty_counts, random_sample=True):
    """
    从JSON文件中按照difficulty字段提取指定数量的数据
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        difficulty_counts: 字典，key为difficulty值（如"1/8"），value为要提取的数量
        random_sample: 是否随机抽取，False则按顺序提取
    """
    # 读取数据
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共读取了 {len(data)} 条数据")
    
    # 按difficulty分组
    grouped_data = defaultdict(list)
    for item in data:
        difficulty = item.get('difficulty')
        if difficulty:
            grouped_data[difficulty].append(item)
    
    # 显示每个难度等级的数据量
    print("\n各难度等级的数据量:")
    for difficulty in sorted(grouped_data.keys()):
        print(f"  {difficulty}: {len(grouped_data[difficulty])} 条")
    
    # 提取数据
    extracted_data = []
    print("\n开始提取数据:")
    for difficulty, count in difficulty_counts.items():
        if difficulty in grouped_data:
            available = len(grouped_data[difficulty])
            actual_count = min(count, available)
            
            # 随机抽取或按顺序取
            if random_sample:
                sampled = random.sample(grouped_data[difficulty], actual_count)
            else:
                sampled = grouped_data[difficulty][:actual_count]
            
            extracted_data.extend(sampled)
            
            print(f"  {difficulty}: 请求 {count} 条, 可用 {available} 条, 实际提取 {actual_count} 条")
        else:
            print(f"  {difficulty}: 未找到该难度等级的数据")
    
    # 保存结果
    print(f"\n总共提取了 {len(extracted_data)} 条数据")
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print("完成!")
    return extracted_data


# 使用示例
if __name__ == "__main__":
    input_file = "/home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/polaris_normal_600_times_8_valid_fixed.json"
    output_file = "/home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/extracted_data.json"
    
    # 指定每个难度等级要提取的数量
    difficulty_counts = {
        "1/8": 100,  # difficulty为1/8提取100条
        "6/8": 100,  # difficulty为6/8提取100条
        # 可以继续添加其他难度等级
        "2/8": 100,
        "3/8": 100,
        "4/8": 100,
        "5/8": 100,
        "7/8": 100
    }
    
    extract_by_difficulty(input_file, output_file, difficulty_counts, random_sample=True)