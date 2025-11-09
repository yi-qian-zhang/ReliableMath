import json
import sys
import os

def check_json_format(file_path):
    """
    检查给定路径的 JSON 文件格式是否有效。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件未找到 - {file_path}", file=sys.stderr)
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试加载文件内容
            json.load(f)
        
        print(f"✅ 成功: JSON 文件格式有效 - {file_path}")
        return True
        
    except json.JSONDecodeError as e:
        # 捕获 JSON 解码错误 (格式错误)
        print(f"❌ 错误: JSON 文件格式无效 - {file_path}")
        print(f"   详细错误: {e}", file=sys.stderr)
        return False
    except IOError as e:
        # 捕获文件读取错误 (如权限问题)
        print(f"❌ 错误: 无法读取文件 - {file_path}", file=sys.stderr)
        print(f"   详细错误: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        # 检查命令行参数数量
        print("用法: python check_json.py </home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/extracted_data.json>")
        sys.exit(1)
        
    json_file = sys.argv[1]
    
    if check_json_format(json_file):
        sys.exit(0) # 格式正确，退出码为 0
    else:
        sys.exit(1) # 格式错误，退出码为 1