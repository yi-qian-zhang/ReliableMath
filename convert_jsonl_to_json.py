#!/usr/bin/env python3
"""
从 JSONL 文件恢复为 JSON 文件
"""
import json
import sys

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue
    return data

def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_jsonl_to_json.py <input.jsonl> <output.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading {input_file}...")
    data = read_jsonl(input_file)

    print(f"Loaded {len(data)} items")

    # 按 ID 排序
    data.sort(key=lambda x: x.get('id', 0))

    print(f"Writing {output_file}...")
    write_json(output_file, data)

    print(f"✓ Done! {len(data)} items saved to {output_file}")
