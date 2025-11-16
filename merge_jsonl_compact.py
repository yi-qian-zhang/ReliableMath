#!/usr/bin/env python3
"""
从 JSONL 文件直接合并为紧凑的 JSON 文件（节省磁盘空间）
用于磁盘空间不足的情况
"""
import json
import sys
import os

def read_jsonl(filepath):
    """读取 JSONL 文件"""
    data = []
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return data

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"Warning: Error parsing line {i}: {e}")
                    continue
    return data

def write_json_compact(filepath, data):
    """写入紧凑的 JSON（不使用缩进，节省空间）"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_jsonl_compact.py <input.jsonl> <output.json>")
        print("\nThis script merges JSONL data into a compact JSON file (no indentation)")
        print("to save disk space when storage is limited.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading {input_file}...")
    data = read_jsonl(input_file)

    if not data:
        print("Error: No valid data found in JSONL file")
        sys.exit(1)

    print(f"Loaded {len(data)} items")

    # 按 ID 排序
    data.sort(key=lambda x: x.get('id', 0))

    print(f"Writing compact JSON to {output_file}...")
    print("(Using compact format without indentation to save disk space)")

    try:
        write_json_compact(output_file, data)

        # 显示文件大小
        file_size = os.path.getsize(output_file)
        if file_size < 1024*1024:
            size_str = f"{file_size/1024:.1f} KB"
        elif file_size < 1024*1024*1024:
            size_str = f"{file_size/(1024*1024):.1f} MB"
        else:
            size_str = f"{file_size/(1024*1024*1024):.2f} GB"

        print(f"✓ Done! {len(data)} items saved to {output_file}")
        print(f"File size: {size_str}")

    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(f"\n❌ Error: Disk is full!")
            print(f"You need to free up disk space or change output to a different partition.")
            print(f"Suggestion: Use /data2 which has 1.7T available")
            sys.exit(1)
        else:
            raise
