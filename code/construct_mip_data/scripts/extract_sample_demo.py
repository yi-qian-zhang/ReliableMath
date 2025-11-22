import json
import pandas as pd
import argparse
import os

def process_data(input_path, output_path):
    # 1. 检查输入
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到输入文件: {input_path}")
        return

    try:
        print(f"📖 正在读取: {input_path} ...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 智能解析结构
        if isinstance(data, dict):
            # 尝试解包常见的包裹层
            for key in ['fullContent', 'data', 'items', 'list']:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            if isinstance(data, dict):
                 data = [data]

        if not isinstance(data, list):
            print("❌ 错误: 无法解析出列表数据。")
            return

        # 3. 指定提取字段
        target_columns = [
            'id', 
            'difficulty', 
            'original_question', 
            'incomplete_question', 
            'ground_truth', 
            'removed_conditions'
        ]

        # 4. 使用 Pandas 整理数据 (方便处理缺失列)
        df = pd.DataFrame(data)
        
        # 补全缺失列
        for col in target_columns:
            if col not in df.columns:
                df[col] = ""
        
        # 筛选列
        df_final = df[target_columns]

        # === 关键修复：确保 difficulty 里的反斜杠被移除 ===
        # 这一步是防止源数据里本身就写了 "6\/8" (这是有可能的)
        # 如果源数据是 "6/8"，这行代码不会有负面影响
        if 'difficulty' in df_final.columns:
             df_final['difficulty'] = df_final['difficulty'].astype(str).str.replace(r'\\/', '/', regex=True)

        # 5. 自动创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"💾 正在保存到: {output_path} ...")

        # 6. 保存逻辑 (根据后缀分流)
        if output_path.endswith('.json'):
            # === 核心修改 ===
            # 不使用 df.to_json()，因为它可能会转义斜杠。
            # 改用 Python 原生 json.dump，它默认不转义斜杠。
            
            # 将 DataFrame 转回 Python 字典列表
            records = df_final.to_dict(orient='records')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # ensure_ascii=False 保证中文不乱码
                # indent=4 保证格式美观
                # Python 原生 json.dump 默认保留 "/" 为 "/"
                json.dump(records, f, ensure_ascii=False, indent=4)
                
        elif output_path.endswith('.xlsx'):
            df_final.to_excel(output_path, index=False)
        else:
            # CSV 默认保存
            df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"✅ 完成！成功提取 {len(df_final)} 条数据。")
        print(f"   difficulty 字段已强制修正为无转义格式 (如 6/8)。")

    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON 提取工具 (修复转义问题)")
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    args = parser.parse_args()

    if not args.input or not args.output:
        # IDE 调试用的默认路径
        default_input = "缺省一条.json"
        default_output = "result.json"  # 测试输出为 JSON
        process_data(default_input, default_output)
    else:
        process_data(args.input, args.output)