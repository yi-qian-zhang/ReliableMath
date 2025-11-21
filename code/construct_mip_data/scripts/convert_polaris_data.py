"""
将 Polaris JSONL 数据转换为标准格式
可以精确控制每个难度的数量
"""

import json
import argparse
import os

def convert_polaris_data(input_file: str, output_file: str, difficulty_counts: dict):
    """
    转换 Polaris 数据格式

    Args:
        input_file: 输入的 JSONL 文件路径
        output_file: 输出的 JSON 文件路径
        difficulty_counts: 字典，格式 {'1/8': 10, '2/8': 10, ..., '7/8': 10}
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

    # 按难度分类存储数据
    data_by_difficulty = {diff: [] for diff in target_difficulties.keys()}

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
                    if difficulty in target_difficulties:
                        if len(data_by_difficulty[difficulty]) < target_difficulties[difficulty]:
                            data_by_difficulty[difficulty].append(item)

                    # 检查是否已经收集够了所有目标难度的数据
                    if all(len(data_by_difficulty[d]) >= target_difficulties[d] for d in target_difficulties):
                        print(f"✅ 已收集齐所有目标数据，提前停止读取（共读取 {total_lines} 行）")
                        break

                except json.JSONDecodeError as e:
                    print(f"⚠️ 警告：跳过无效行 {total_lines}: {e}")
                    continue

    print(f"✓ 读取总行数: {total_lines}")
    print(f"✓ 收集到的数据:")
    # 按难度值排序打印收集结果
    sorted_collected = sorted(target_difficulties.items(), key=lambda item: int(item[0].split('/')[0]))
    for diff, target in sorted_collected:
        actual = len(data_by_difficulty[diff])
        status = "✓" if actual >= target else "✗"
        print(f"  {status} {diff}: {actual}/{target} 条")

    # 检查是否所有难度都收集够了
    for diff, target_count in target_difficulties.items():
        actual_count = len(data_by_difficulty[diff])
        if actual_count < target_count:
            print(f"\n❌ 警告：难度 {diff} 只找到 {actual_count} 条，不足目标 {target_count} 条")

    # 合并所有数据
    all_data = []
    # 按照难度顺序合并，确保输出顺序一致
    sorted_diffs = sorted(target_difficulties.keys(), key=lambda d: int(d.split('/')[0]))
    for diff in sorted_diffs:
        all_data.extend(data_by_difficulty[diff])

    if len(all_data) == 0:
        print("❌ 错误：没有找到符合条件的数据！")
        return

    # 转换格式
    converted_data = []
    for i, item in enumerate(all_data, start=1):
        # 保持原始ID或使用行号作为新ID
        converted_item = {
            "data_source": "polaris",
            # 使用原始 ID，如果不存在则使用序号（纯数字）
            "id": item.get("id", i),
            "original_question": item.get("problem", ""),
            "ground_truth": item.get("answer", ""),
            "solution": "",
            "difficulty": item.get("difficulty", "")
        }
        converted_data.append(converted_item)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print(f"✅ 已保存到: {output_file}")
    print(f"✅ 共转换 {len(converted_data)} 条数据")

    # 统计难度分布
    difficulty_count = {}
    for item in converted_data:
        diff = item['difficulty']
        difficulty_count[diff] = difficulty_count.get(diff, 0) + 1

    print("\n最终难度分布:")
    sorted_final_counts = sorted(difficulty_count.items(), key=lambda item: int(item[0].split('/')[0]))
    for diff, count in sorted_final_counts:
        print(f"  {diff}: {count} 条")

    # 显示前3条样例
    print("\n前3条数据预览:")
    for i, item in enumerate(converted_data[:3]):
        print(f"\n[{i}] ID: {item['id']}, 难度: {item['difficulty']}")
        question_preview = item['original_question'][:80].replace('\n', ' ') + "..." if len(item['original_question']) > 80 else item['original_question'].replace('\n', ' ')
        print(f"    问题: {'original_question'}")
        print(f"    答案: {item['ground_truth']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="转换 Polaris JSONL 数据格式，可精确控制提取的难度和数量。",
        formatter_class=argparse.RawTextHelpFormatter # 保持描述中的换行
    )

    # 核心路径参数
    parser.add_argument(
        "--input",
        default="polaris-data.jsonl",
        help="输入 JSONL 文件路径 (默认: polaris-data.jsonl)"
    )
    parser.add_argument(
        "--output",
        default="polaris_selected.json",
        help="输出 JSON 文件路径 (默认: polaris_selected.json)"
    )

    # 难度控制参数
    # 创建一个参数组来管理难度相关的参数，便于查看
    difficulty_group = parser.add_argument_group('难度控制参数 (1/8 ~ 7/8)')
    
    # 动态创建 1/8 到 7/8 的参数
    default_count = 0
    for i in range(1, 8):
        diff_str = f"{i}/8"
        difficulty_group.add_argument(
            f"--count_{i}",
            type=int,
            default=default_count,
            help=f"难度 {diff_str} 的数量 (默认: {default_count})"
        )
    
    # 增加对 8/8 的支持
    difficulty_group.add_argument(
        "--count_8",
        type=int,
        default=0,
        help="难度 8/8 的数量 (默认: 0)"
    )
    
    args = parser.parse_args()

    # 构建难度计数字典
    difficulty_counts = {}
    for i in range(1, 9):
        diff_str = f"{i}/8"
        arg_name = f"count_{i}"
        # 使用 getattr 动态获取参数值
        count = getattr(args, arg_name)
        if count > 0:
            difficulty_counts[diff_str] = count

    convert_polaris_data(
        input_file=args.input,
        output_file=args.output,
        difficulty_counts=difficulty_counts
    )