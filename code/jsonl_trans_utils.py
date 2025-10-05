import json

input_file = "/data1/HF-Datasets/POLARIS-Project/Polaris-Dataset-53K/polaris-data-53K.jsonl"
output_file = "/home/zhangyiqian/ReliableMath/data/solve/polaris_20.json"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(fin):
        if idx >= 20:  # 只处理前 20 条
            break
        item = json.loads(line)
        new_item = {
            "data_source": "polaris",
            "id": idx,
            "question": item["problem"],
            "ground_truth": item["answer"],
            "solution": "",
            "difficulty": item["difficulty"]
        }
        fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"前 20 条已转换完成，保存到 {output_file}")
