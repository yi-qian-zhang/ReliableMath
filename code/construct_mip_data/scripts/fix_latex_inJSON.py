import re
import json

def fix_latex_escaping_in_file(input_file, output_file):
    """修复 JSON 文件中的 LaTeX 转义问题"""
    print(f"正在读取文件: {input_file}")
    
    # 以文本模式读取（不直接用 json.load，因为文件本身就有问题）
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("正在修复 LaTeX 转义...")
    
    # 1. 修复 \u{...} 这种错误的转义（可能是 \frac 被错误处理）
    # 常见的错误模式：\u{AK}、\u{xyz} 等
    content = re.sub(r'\\u\{([^}]+)\}', r'\\\\text{\1}', content)
    
    # 2. 确保所有 LaTeX 命令的反斜杠被正确转义（单个 \ 变成 \\）
    # 但要小心不要重复转义已经是 \\ 的情况
    # 这个正则会匹配单个反斜杠后面跟字母的情况（LaTeX 命令）
    def replace_single_backslash(match):
        # 如果前面没有反斜杠，就添加一个
        return '\\\\' + match.group(1)
    
    # 注意：这个替换需要谨慎，可能需要根据实际情况调整
    # content = re.sub(r'(?<!\\)\\([a-zA-Z])', replace_single_backslash, content)
    
    print("正在验证修复后的 JSON...")
    try:
        data = json.loads(content)
        print(f"✅ 修复成功！共有 {len(data)} 条数据")
        
        # 保存修复后的文件
        print(f"正在保存到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("✅ 保存完成!")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ 修复后仍有问题: {e}")
        # 保存修复尝试的内容以便检查
        debug_file = output_file.replace('.json', '_debug.txt')
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已将修复尝试的内容保存到: {debug_file}")
        return None


# 使用
# input_file = "/home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/polaris_normal_600_times_8_valid.json"
# output_file = "/home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/polaris_normal_600_times_8_valid_fixed.json"
input_file = "/home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/extracted_data.json"
output_file = "/home/zhangyiqian/ReliableMath/data/construct_mip_qwen_7B_16384/11-04/polaris_normal_600_times_8/deepscaler_extract/extracted_data_fixed.json"
fix_latex_escaping_in_file(input_file, output_file)