"""
combined_analysis.py

功能:
1. 读取原始 Polaris JSON 文件，统计每个问题 (ID) 的总条件数 (即 len(removal_variants))。
2. 读取筛选后的 JSON 文件，统计每个原始问题 (original_id) 剩下的变体数量。
3. 结合两项统计结果，计算保留率，并输出为 Markdown 表格。
"""

import json
import os
from collections import defaultdict
import argparse
import sys

def load_data(file_path: str) -> list:
    """加载 JSON 文件，并检查是否为列表。"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件不存在: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                # 假设单个对象也有效，将其包装成列表
                return [data]
            return data
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON 文件解码失败 ({file_path}): {e}")
        return []
    except Exception as e:
        print(f"❌ 错误：读取文件时发生意外错误 ({file_path}): {e}")
        return []

def get_original_counts(data: list) -> dict:
    """
    从原始数据中统计每个 ID 的总条件数。
    总条件数 = len(item['removal_variants'])
    """
    counts = {}
    for item in data:
        problem_id = item.get("id")
        # 确保 ID 是可哈希的且有效
        if problem_id is not None:
            variants = item.get("removal_variants", [])
            counts[problem_id] = len(variants)
    return counts

def get_filtered_counts(data: list) -> defaultdict:
    """
    从筛选数据中统计每个 original_id 剩下的变体数量。
    """
    counts = defaultdict(int)
    for item in data:
        original_id = item.get("original_id")
        if original_id is not None:
            counts[original_id] += 1
    return counts

def run_analysis(original_file: str, filtered_file: str):
    """执行完整的分析并输出表格结果。"""
    print(f"正在加载原始数据: {original_file}...")
    original_data = load_data(original_file)
    original_counts = get_original_counts(original_data)
    
    print(f"正在加载筛选数据: {filtered_file}...")
    filtered_data = load_data(filtered_file)
    filtered_counts = get_filtered_counts(filtered_data)
    
    if not original_counts and not filtered_counts:
        print("无有效数据可供分析。请检查文件内容。")
        return

    # --- 组合数据并准备输出 ---
    
    # 获取所有涉及的 ID
    all_ids = set(original_counts.keys()) | set(filtered_counts.keys())
    
    results = []
    total_original = 0
    total_filtered = 0
    
    # 转换为整数后排序
    sorted_ids = sorted(list(all_ids), key=lambda x: int(x) if str(x).isdigit() else sys.maxsize)

    for problem_id in sorted_ids:
        # 获取数量
        total = original_counts.get(problem_id, 0)
        filtered = filtered_counts.get(problem_id, 0)
        
        # 计算保留率
        retention_rate = (filtered / total) * 100 if total > 0 else 0.0
        
        # 备注逻辑
        notes = ""
        if total == 0 and filtered > 0:
             notes = "原始记录缺失，但筛选数据中有保留项。"
        elif filtered == 0 and total > 0:
            notes = "所有变体均被筛选掉。"
        elif total != filtered and filtered > 0 and total > 0:
            notes = f"有 {total - filtered} 个变体被筛选掉。"
        elif total == filtered and total > 0:
             notes = "所有变体均保留。"
        
        results.append({
            "id": problem_id,
            "total_conditions": total,
            "filtered_count": filtered,
            "retention_rate": f"{retention_rate:.1f}%",
            "notes": notes
        })
        
        total_original += total
        total_filtered += filtered

    # --- 输出 Markdown 表格 (即可视化结果) ---
    
    print("\n" + "=" * 50)
    print("## 📊 Polaris 条件移除变体统计结果")
    print("-" * 50)
    
    # 表格头部
    print("| Original ID | 原始总条件数 | 筛选后保留变体数 | 保留率 | 备注 |")
    print("|:---:|:---:|:---:|:---:|:---|")
    
    # 表格内容
    for row in results:
        print(f"| {row['id']} | **{row['total_conditions']}** | **{row['filtered_count']}** | {row['retention_rate']} | {row['notes']} |")

    # 总计行
    overall_retention = (total_filtered / total_original) * 100 if total_original > 0 else 0.0
    print("-" * 50)
    print(f"| **总计** | **{total_original}** | **{total_filtered}** | **{overall_retention:.1f}%** | |")
    print("=" * 50)
    
    print("\n**重要说明:**")
    print("1. **原始总条件数**：根据 `len(removal_variants)` 字段计算。")
    print("2. **筛选后保留变体数**：根据筛选文件中的 `original_id` 计数。")
    print("3. 如果某 ID 的'原始总条件数'为 0 但'筛选后保留变体数'大于 0，说明原始 JSON 中该问题没有 `removal_variants` 字段或其值为空，但您的筛选结果中包含了该 ID 的变体。**这可能是原始数据不完整或数据格式不一致导致的。**")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="分析 Polaris 条件移除变体数据，计算每个问题的总条件数和筛选后保留数。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--original_file", 
        type=str, 
        help="原始 Polaris JSON 文件路径 (包含 'removal_variants' 字段)"
    )
    parser.add_argument(
        "--filtered_file", 
        type=str, 
        help="筛选后的 JSON 文件路径 (包含 'original_id' 字段)"
    )
    
    args = parser.parse_args()
    
    run_analysis(args.original_file, args.filtered_file)