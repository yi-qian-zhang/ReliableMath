"""
从 valid 数据集中提取指定数量的数据
按难度分类，同一难度内按 ID 排序
"""

import json
import argparse
import os
import re

def parse_id_for_sorting(item_id):
    """
    解析 ID 用于排序
    ID 格式: "{original_id}_remove_{index}"
    返回: (original_id, index) 用于排序
    """
    match = re.match(r'(\d+)_remove_(\d+)', item_id)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # 如果无法解析，返回一个很大的数，放到最后
    return (float('inf'), float('inf'))

def extract_valid_data(input_file: str, output_file: str, difficulty_counts: dict):
    """
    从 valid 数据集中提取指定数量的数据

    Args:
        input_file: 输入的 JSON 文件路径
        output_file: 输出的 JSON 文件路径
        difficulty_counts: 字典，格式 {'1/8': 100, '2/8': 100, ..., '7/8': 100}
    """

    # 过滤掉目标数量为 0 的难度
    target_difficulties = {diff: count for diff, count in difficulty_counts.items() if count > 0}

    if not os.path.exists(input_file):
        print(f"❌ 错误：输入文件不存在: {input_file}")
        return

    if not target_difficulties:
        print("⚠️ 警告：所有目标难度数量均为 0，没有数据需要提取。")
        return

    print(f"正在读取: {input_file}")
    print(f"目标数量:")
    # 按难度值排序打印目标数量
    sorted_targets = sorted(target_difficulties.items(), key=lambda item: int(item[0].split('/')[0]))
    for diff, count in sorted_targets:
        print(f"  {diff}: {count} 条")
    print("-" * 70)

    # 读取 JSON 文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON 文件格式错误: {e}")
        return
    except Exception as e:
        print(f"❌ 错误：读取文件失败: {e}")
        return

    print(f"✓ 读取总数据量: {len(all_data)} 条")

    # 按难度分类存储数据
    data_by_difficulty = {diff: [] for diff in target_difficulties.keys()}

    # 将数据按难度分类
    for item in all_data:
        difficulty = item.get('difficulty')
        if difficulty in target_difficulties:
            data_by_difficulty[difficulty].append(item)

    # 对每个难度的数据按 ID 排序
    for diff in data_by_difficulty:
        data_by_difficulty[diff].sort(key=lambda x: parse_id_for_sorting(x.get('id', '')))

    print(f"✓ 按难度分类统计:")
    # 按难度值排序打印分类结果
    sorted_diffs = sorted(data_by_difficulty.items(), key=lambda item: int(item[0].split('/')[0]))
    for diff, items in sorted_diffs:
        print(f"  {diff}: {len(items)} 条可用")

    # 提取指定数量的数据
    extracted_data = []
    extraction_summary = {}

    sorted_target_diffs = sorted(target_difficulties.keys(), key=lambda d: int(d.split('/')[0]))
    
    for diff in sorted_target_diffs:
        available = data_by_difficulty[diff]
        target_count = target_difficulties[diff]
        
        # 提取前 N 条（已经按 ID 排序）
        extracted = available[:target_count]
        extracted_data.extend(extracted)
        extraction_summary[diff] = len(extracted)
        
        # 警告：如果数据不足
        if len(extracted) < target_count:
            print(f"\n⚠️ 警告：难度 {diff} 只有 {len(extracted)} 条数据，不足目标 {target_count} 条")

    if len(extracted_data) == 0:
        print("❌ 错误：没有找到符合条件的数据！")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print(f"✅ 已保存到: {output_file}")
    print(f"✅ 共提取 {len(extracted_data)} 条数据")

    # 显示提取结果统计
    print("\n提取结果统计:")
    sorted_summary = sorted(extraction_summary.items(), key=lambda item: int(item[0].split('/')[0]))
    for diff, count in sorted_summary:
        target = target_difficulties[diff]
        status = "✓" if count >= target else "✗"
        print(f"  {status} {diff}: {count}/{target} 条")

    # 显示每个难度的 ID 范围
    print("\n每个难度的 ID 范围:")
    for diff in sorted_target_diffs:
        items = [item for item in extracted_data if item.get('difficulty') == diff]
        if items:
            first_id = items[0].get('id', 'N/A')
            last_id = items[-1].get('id', 'N/A')
            print(f"  {diff}: {first_id} ~ {last_id}")

    # 显示前3条样例
    print("\n前3条数据预览:")
    for i, item in enumerate(extracted_data[:3]):
        print(f"\n[{i}] ID: {item.get('id')}, 难度: {item.get('difficulty')}")
        incomplete_q = item.get('incomplete_question', '')
        question_preview = incomplete_q[:80].replace('\n', ' ') + "..." if len(incomplete_q) > 80 else incomplete_q.replace('\n', ' ')
        print(f"    问题: {question_preview}")
        print(f"    答案: {item.get('ground_truth', 'N/A')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 valid 数据集中提取指定数量的数据，按难度分类并按 ID 排序。",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 核心路径参数
    parser.add_argument(
        "--input",
        default="valid_data.json",
        help="输入 JSON 文件路径 (默认: valid_data.json)"
    )
    parser.add_argument(
        "--output",
        default="valid_extracted.json",
        help="输出 JSON 文件路径 (默认: valid_extracted.json)"
    )

    # 难度控制参数
    difficulty_group = parser.add_argument_group('难度控制参数 (1/8 ~ 7/8)')
    
    default_count = 100
    for i in range(1, 8):
        diff_str = f"{i}/8"
        difficulty_group.add_argument(
            f"--count_{i}",
            type=int,
            default=default_count,
            help=f"难度 {diff_str} 的数量 (默认: {default_count})"
        )
    
    args = parser.parse_args()

    # 构建难度计数字典
    difficulty_counts = {}
    for i in range(1, 8):
        diff_str = f"{i}/8"
        arg_name = f"count_{i}"
        count = getattr(args, arg_name)
        if count > 0:
            difficulty_counts[diff_str] = count

    extract_valid_data(
        input_file=args.input,
        output_file=args.output,
        difficulty_counts=difficulty_counts
    )