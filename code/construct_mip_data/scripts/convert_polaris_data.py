"""
将 Polaris JSONL 数据转换为标准格式
可以精确控制每个难度的数量
"""

import json
import argparse
import os

def convert_polaris_data(input_file, output_file, difficulty_counts):
    """
    转换 Polaris 数据格式
    
    Args:
        input_file: 输入的 JSONL 文件路径
        output_file: 输出的 JSON 文件路径
        difficulty_counts: 字典，格式 {'6/8': 6, '7/8': 7, '8/8': 7}
    """
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    print(f"正在读取: {input_file}")
    print(f"目标数量:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count} 条")
    print("-" * 70)
    
    # 按难度分类存储数据
    data_by_difficulty = {diff: [] for diff in difficulty_counts.keys()}
    
    # 读取 JSONL 文件
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    difficulty = item.get('difficulty')
                    
                    # 如果是我们需要的难度，且还没收集够
                    if difficulty in difficulty_counts:
                        if len(data_by_difficulty[difficulty]) < difficulty_counts[difficulty]:
                            data_by_difficulty[difficulty].append(item)
                    
                    # 检查是否已经收集够了所有难度
                    if all(len(data_by_difficulty[d]) >= difficulty_counts[d] for d in difficulty_counts):
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"警告：跳过无效行 {total_lines}: {e}")
                    continue
    
    print(f"✓ 读取总行数: {total_lines}")
    print(f"✓ 收集到的数据:")
    for diff in sorted(difficulty_counts.keys()):
        target = difficulty_counts[diff]
        actual = len(data_by_difficulty[diff])
        status = "✓" if actual >= target else "✗"
        print(f"  {status} {diff}: {actual}/{target} 条")
    
    # 检查是否所有难度都收集够了
    for diff, target_count in difficulty_counts.items():
        actual_count = len(data_by_difficulty[diff])
        if actual_count < target_count:
            print(f"\n警告：难度 {diff} 只找到 {actual_count} 条，不足目标 {target_count} 条")
    
    # 合并所有数据
    all_data = []
    for diff in sorted(difficulty_counts.keys()):
        all_data.extend(data_by_difficulty[diff])
    
    if len(all_data) == 0:
        print("错误：没有找到符合条件的数据！")
        return
    
    # 转换格式
    converted_data = []
    for i, item in enumerate(all_data):
        converted_item = {
            "data_source": "polaris",
            "id": i,
            "question": item.get("problem", ""),
            "ground_truth": item.get("answer", ""),
            "solution": "",
            "difficulty": item.get("difficulty", "")
        }
        converted_data.append(converted_item)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("-" * 70)
    print(f"✓ 已保存到: {output_file}")
    print(f"✓ 共转换 {len(converted_data)} 条数据")
    
    # 统计难度分布
    difficulty_count = {}
    for item in converted_data:
        diff = item['difficulty']
        difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
    
    print("\n最终难度分布:")
    for diff in sorted(difficulty_count.keys()):
        print(f"  {diff}: {difficulty_count[diff]} 条")
    
    # 显示前3条样例
    print("\n前3条数据预览:")
    for i, item in enumerate(converted_data[:3]):
        print(f"\n[{i}] ID: {item['id']}, 难度: {item['difficulty']}")
        question_preview = item['question'][:80] + "..." if len(item['question']) > 80 else item['question']
        print(f"    问题: {question_preview}")
        print(f"    答案: {item['ground_truth']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换 Polaris 数据格式")
    parser.add_argument(
        "--input", 
        default="/home/zhangyiqian/ReliableMath/data/solve/polaris-data-53K.jsonl", 
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output", 
        default="/data/home/zyq/ReliableMath/data/solve/polaris_easy_20.json", 
        help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--count_6", 
        type=int, 
        default=10, 
        help="难度 6/8 的数量"
    )
    parser.add_argument(
        "--count_7", 
        type=int, 
        default=10, 
        help="难度 7/8 的数量"
    )
    parser.add_argument(
        "--count_8", 
        type=int, 
        default=10, 
        help="难度 8/8 的数量"
    )
    
    args = parser.parse_args()
    
    # 构建难度计数字典
    difficulty_counts = {
        '6/8': args.count_6,
        '7/8': args.count_7,
        '8/8': args.count_8
    }
    
    convert_polaris_data(
        input_file=args.input,
        output_file=args.output,
        difficulty_counts=difficulty_counts
    )