import json
import random
import os  # 导入 os 模块以处理文件路径

def load_data(filepath):
    """从指定路径加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：文件未找到 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 {filepath}")
        return None
    except Exception as e:
        print(f"加载文件时发生未知错误: {e}")
        return None

def process_item(item):
    """
    提取所需字段，并找到 'verification['round_b']['all_attempts']' 中
    所有 'is_correct' 为 true 的 attempt 对象。
    """
    
    # 提取基本字段
    processed = {
        "id": item.get("id"),
        "difficulty": item.get("difficulty"),
        "original_question": item.get("original_question"),
        "all_extracted_conditions": item.get("all_extracted_conditions"),
        "num_conditions_extracted": item.get("num_conditions_extracted"),
        "incomplete_question": item.get("incomplete_question"),
        "removed_condition": item.get("removed_condition"),
        "remaining_conditions": item.get("remaining_conditions"),
        "correct_round_b_attempts": [] 
    }
    
    # 查找 'round_b' 中所有 'is_correct': true 的 attempts
    try:
        if "verification" in item and "round_b" in item["verification"]:
            all_attempts = item["verification"]["round_b"].get("all_attempts", [])
            for attempt in all_attempts:
                if attempt.get("is_correct") is True:
                    processed["correct_round_b_attempts"].append(attempt)
    except Exception as e:
        print(f"处理 item {item.get('id')} 的 'verification' 字段时出错: {e}")

    return processed

def save_data(data, filepath):
    """将数据保存到指定的JSON文件路径"""
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n--- 结果已成功保存到 {filepath} ---")
    except PermissionError:
        print(f"错误：权限不足，无法写入文件 {filepath}")
    except Exception as e:
        print(f"保存文件时发生未知错误: {e}")

def main():
    FILE_PATH = "/home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/extracted_data_fixed.json"
    OUTPUT_FILE = "/home/zhangyiqian/ReliableMath/code/construct_mip_data/scripts/check_extracted_data_output.json"
    SAMPLE_COUNT = 20

    data = load_data(FILE_PATH)
    
    if data is None:
        return

    # 确保采样数量不超过数据总量
    if len(data) < SAMPLE_COUNT:
        print(f"警告：数据总量 ({len(data)}) 少于采样数量 ({SAMPLE_COUNT})。将采样所有数据。")
        SAMPLE_COUNT = len(data)
        
    # 随机采样
    sampled_data = random.sample(data, SAMPLE_COUNT)
    
    # 处理采样的数据
    processed_results = [process_item(item) for item in sampled_data]
    
    # 打印结果以便查看
    print(f"--- 成功采样并处理了 {len(processed_results)} 条数据 ---")
    # 为了避免控制台刷屏，只打印部分结果的ID
    for i, item in enumerate(processed_results[:5]): # 最多打印前5个的ID
        print(f"  样本 {i+1}: id = {item.get('id')}")
    if len(processed_results) > 5:
        print(f"  ... (及其他 {len(processed_results) - 5} 个)")
    
    # --- 保存到文件 ---
    save_data(processed_results, OUTPUT_FILE)

if __name__ == "__main__":
    main()