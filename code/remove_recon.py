import os
import json
import time
import logging
import numpy as np
import argparse
from openai import OpenAI
import random
from tqdm import tqdm
import requests

# 设置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="POLARIS数据集改造")

parser.add_argument("--model", default="deepseek_v3", help="模型名称")
parser.add_argument("--data_dir", default="./data/solve", help="输入文件路径")
parser.add_argument("--output_dir", default="./data/unsol", help="输出文件路径")
parser.add_argument("--prompt_dir", default="./prompt/{}/rewrite", help="prompt模板路径")
parser.add_argument("--dataset", default="your_dataset", help="数据集名称")
parser.add_argument("--prompt", default="v4-remove-only", type=str, help="prompt类型")
parser.add_argument("--temperature", default=0.0, type=float, help="temperature")
parser.add_argument("--split_id", default=0, type=int, help="分片ID")
args = parser.parse_args()

# 只使用remove类型
UNS_TYPE = ["remove"]

# 加载API配置
model_options = json.load(open("./data/api_keys.json", "r"))

# ============= 工具函数 =============

def read_json(filepath):
    """读取JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(filepath, data):
    """写入JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_jsonl(filepath):
    """读取JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def dump_jsonl(data, filepath, append=False):
    """写入JSONL文件"""
    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def jsonl2json(input_path, output_path):
    """将JSONL转换为JSON"""
    data = read_jsonl(input_path)
    write_json(output_path, data)

# ============= API调用函数 =============

def get_response_openai(input_prompt, persona, model=None, temperature=0.0):
    """调用OpenAI API"""
    if model is None:
        model = args.model
    
    model_name, key, url = random.choice(model_options[model])
    client = OpenAI(api_key=key, base_url=url)
    
    message = [
        {"role": "system", "content": persona},
        {"role": "user", "content": input_prompt},
    ]
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.warning(f'API调用失败: {e}\n重试中...')
            time.sleep(20 * (attempt + 1))
    
    return ""

# ============= 核心处理函数 =============

def extract_condition(dataset, extract_path):
    """提取关键条件"""
    if os.path.exists(extract_path):
        try:
            logging.info(f"继续从上次提取，输出文件：{extract_path}")
            total_len = len(dataset)
            data_saved = read_jsonl(extract_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            return False
    else:
        total_len = len(dataset)

    with tqdm(total=len(dataset), desc="提取条件") as t:
        for data in dataset:
            prompt_path = os.path.join(args.prompt_dir.format(args.prompt), "extract.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                extract_prompt = f.read()
            
            input_prompt = extract_prompt.format(
                original_math_question=data["question"]
            )
            
            extracted_condition = get_response_openai(
                input_prompt, 
                persona="你是一个优秀的数学条件提取器。",
                model=args.model
            )
            
            data["extracted_condition"] = extracted_condition
            t.update(1)
            dump_jsonl(data, extract_path, append=True)
    
    if len(read_jsonl(extract_path)) == total_len:
        jsonl2json(extract_path, extract_path)
        return True
    return False

def condition_process(extract_path):
    """处理提取的条件"""
    dataset = read_json(extract_path)
    
    for data in dataset:
        conditions = data["extracted_condition"]
        
        # 清理格式
        if "### 提取的条件 ###" in conditions:
            conditions = conditions.split("### 提取的条件 ###")[-1]
        
        # 分割条件
        conditions = conditions.strip()
        conditions = conditions.replace("\\n\\n", "\n\n")
        sentences = conditions.split('\n\n')
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # 移除编号
            if sentence and sentence[0].isdigit() and len(sentence) > 2:
                if sentence[1] == '.' and sentence[2] == ' ':
                    sentence = sentence[3:]
            
            if sentence:
                cleaned_sentences.append(sentence)
        
        data["extracted_condition"] = cleaned_sentences
    
    write_json(extract_path, dataset)

def remove_analysis(dataset, analysis_path):
    """分析移除条件的影响"""
    if os.path.exists(analysis_path):
        try:
            logging.info(f"继续从上次分析，输出文件：{analysis_path}")
            total_len = len(dataset)
            data_saved = read_jsonl(analysis_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            return False
    else:
        total_len = len(dataset)

    with tqdm(total=len(dataset), desc="分析移除影响") as t:
        for data in dataset:
            prompt_path = os.path.join(args.prompt_dir.format(args.prompt), "remove_analysis.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                analysis_prompt = f.read()
            
            analyses = []
            for condition in data["extracted_condition"]:
                input_prompt = analysis_prompt.format(
                    original_math_question=data["question"],
                    original_answer=data["ground_truth"],
                    extracted_condition=condition
                )
                
                generation = get_response_openai(
                    input_prompt,
                    persona="你是一个优秀的数学问题分析师。",
                    model=args.model
                )
                analyses.append(generation)
            
            data["remove_analysis"] = analyses
            t.update(1)
            dump_jsonl(data, analysis_path, append=True)
    
    if len(read_jsonl(analysis_path)) == total_len:
        jsonl2json(analysis_path, analysis_path)
        return True
    return False

def analysis_process(analysis_path):
    """处理分析结果"""
    dataset = read_json(analysis_path)
    
    for data in dataset:
        cleaned_analysis = []
        for analysis in data["remove_analysis"]:
            # 清理格式
            analysis = analysis.replace("### 分析 ###：", "### 分析 ###")
            if "### 分析 ###" in analysis:
                analysis = analysis.split("### 分析 ###")[-1]
            analysis = analysis.strip()
            cleaned_analysis.append(analysis)
        
        data["remove_analysis"] = cleaned_analysis
    
    write_json(analysis_path, dataset)

def condition_rewrite(dataset, rewrite_path):
    """重写问题（移除条件）"""
    if os.path.exists(rewrite_path):
        try:
            logging.info(f"继续从上次重写，输出文件：{rewrite_path}")
            total_len = len(dataset)
            data_saved = read_jsonl(rewrite_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            return False
    else:
        total_len = len(dataset)

    with tqdm(total=len(dataset), desc="重写问题") as t:
        for data in dataset:
            prompt_path = os.path.join(args.prompt_dir.format(args.prompt), "remove_rewrite.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                rewrite_prompt = f.read()
            
            rewrites = []
            for condition, analysis in zip(data["extracted_condition"], data["remove_analysis"]):
                if analysis.strip() == "":
                    rewrites.append("")
                    continue
                
                input_prompt = rewrite_prompt.format(
                    original_math_question=data["question"],
                    original_answer=data["ground_truth"],
                    extracted_condition=condition,
                    analysis=analysis
                )
                
                generation = get_response_openai(
                    input_prompt,
                    persona="你是一个优秀的数学问题改写者。",
                    model=args.model
                )
                rewrites.append(generation)
            
            data["remove"] = rewrites
            t.update(1)
            dump_jsonl(data, rewrite_path, append=True)
    
    if len(read_jsonl(rewrite_path)) == total_len:
        jsonl2json(rewrite_path, rewrite_path)
        return True
    return False

def rewrite_process(rewrite_path):
    """处理重写结果"""
    dataset = read_json(rewrite_path)
    
    for data in dataset:
        cleaned_rewrites = []
        for rewrite in data["remove"]:
            # 清理格式
            rewrite = rewrite.replace("### 改写后的数学问题 ###：", "### 改写后的数学问题 ###")
            if "### 改写后的数学问题 ###" in rewrite:
                rewrite = rewrite.split("### 改写后的数学问题 ###")[-1]
            rewrite = rewrite.strip()
            cleaned_rewrites.append(rewrite)
        
        data["remove"] = cleaned_rewrites
    
    write_json(rewrite_path, dataset)

def verify_removal(dataset, verify_path):
    """验证条件是否成功移除"""
    if os.path.exists(verify_path):
        try:
            logging.info(f"继续验证，输出文件：{verify_path}")
            total_len = len(dataset)
            data_saved = read_jsonl(verify_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            pass
    else:
        total_len = len(dataset)
    
    with tqdm(total=len(dataset), desc="验证条件移除") as t:
        for data in dataset:
            prompt_path = os.path.join(args.prompt_dir.format(args.prompt), "verify_removal.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                verify_prompt = f.read()
            
            removal_results = []
            for condition, rewritten in zip(data["extracted_condition"], data["remove"]):
                input_prompt = verify_prompt.format(
                    original_math_question=data["question"],
                    extracted_condition=condition,
                    rewritten_math_question=rewritten
                )
                
                result = get_response_openai(
                    input_prompt,
                    persona="你是一个精确的验证员。",
                    model=args.model
                )
                
                # 提取True/False
                result = result.strip()
                if "True" in result:
                    removal_results.append("True")
                elif "False" in result:
                    removal_results.append("False")
                else:
                    removal_results.append("Unknown")
            
            data["removal_verified"] = removal_results
            t.update(1)
            dump_jsonl(data, verify_path, append=True)
    
    if len(read_jsonl(verify_path)) == total_len:
        jsonl2json(verify_path, verify_path)
        return True
    return False

def verify_unsolvable(dataset, verify_path):
    """验证问题是否无法求解"""
    if os.path.exists(verify_path):
        try:
            logging.info(f"继续验证，输出文件：{verify_path}")
            total_len = len(dataset)
            data_saved = read_jsonl(verify_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            pass
    else:
        total_len = len(dataset)
    
    with tqdm(total=len(dataset), desc="验证无法求解") as t:
        for data in dataset:
            prompt_path = os.path.join(args.prompt_dir.format(args.prompt), "verify_unsolvable.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                verify_prompt = f.read()
            
            unsolvable_results = []
            for rewritten in data["remove"]:
                input_prompt = verify_prompt.format(
                    rewritten_math_question=rewritten,
                    original_answer=data["ground_truth"]
                )
                
                result = get_response_openai(
                    input_prompt,
                    persona="你是一个数学问题求解专家。",
                    model=args.model
                )
                
                # 提取True/False
                result = result.strip()
                if "True" in result:
                    unsolvable_results.append("True")
                elif "False" in result:
                    unsolvable_results.append("False")
                else:
                    unsolvable_results.append("Unknown")
            
            data["unsolvable_verified"] = unsolvable_results
            t.update(1)
            dump_jsonl(data, verify_path, append=True)
    
    if len(read_jsonl(verify_path)) == total_len:
        jsonl2json(verify_path, verify_path)
        return True
    return False

def filter_valid_data(final_path):
    """筛选通过所有验证的数据"""
    dataset = read_json(final_path)
    valid_data = []
    
    for data in dataset:
        for i, (removed, unsolvable) in enumerate(zip(data["removal_verified"], data["unsolvable_verified"])):
            if removed == "True" and unsolvable == "True":
                valid_item = {
                    "id": f"{data['id']}_{i}",
                    "data_source": data.get("data_source", ""),
                    "difficulty": data.get("difficulty", ""),
                    "original_question": data["question"],
                    "original_answer": data["ground_truth"],
                    "removed_condition": data["extracted_condition"][i],
                    "unsolvable_question": data["remove"][i],
                    "analysis": data["remove_analysis"][i]
                }
                valid_data.append(valid_item)
    
    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)
    print(f"筛选完成，共{len(valid_data)}条有效数据")
    
    # 生成统计信息
    print(f"原始数据：{len(dataset)}条")
    print(f"生成数据：{sum(len(d['remove']) for d in dataset)}条")
    print(f"有效数据：{len(valid_data)}条")
    print(f"成功率：{len(valid_data) / sum(len(d['remove']) for d in dataset) * 100:.2f}%")

# ============= 主工作流 =============

def construction_workflow():
    """完整的数据集构造流程"""
    input_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    dataset = read_json(input_path)
    
    output_dir = os.path.join(args.output_dir, args.prompt)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*50)
    print("开始数据集改造流程")
    print(f"输入数据集：{input_path}")
    print(f"输出目录：{output_dir}")
    print("="*50)
    
    # Step 1: 条件提取
    print("\nStep 1: 提取关键条件")
    extract_path = os.path.join(output_dir, f"{args.dataset}_extract.json")
    if extract_condition(dataset, extract_path):
        print("条件提取完成，处理格式...")
        condition_process(extract_path)
        print("条件处理完成")
    
    # Step 2: 移除分析
    print("\nStep 2: 分析移除影响")
    dataset = read_json(extract_path)
    analysis_path = os.path.join(output_dir, f"{args.dataset}_analysis.json")
    if remove_analysis(dataset, analysis_path):
        print("分析完成，处理格式...")
        analysis_process(analysis_path)
        print("分析处理完成")
    
    # Step 3: 问题重写
    print("\nStep 3: 重写问题")
    dataset = read_json(analysis_path)
    rewrite_path = os.path.join(output_dir, f"{args.dataset}_rewrite.json")
    if condition_rewrite(dataset, rewrite_path):
        print("重写完成，处理格式...")
        rewrite_process(rewrite_path)
        print("重写处理完成")
    
    # Step 4: 验证移除
    print("\nStep 4: 验证条件移除")
    dataset = read_json(rewrite_path)
    removal_verify_path = os.path.join(output_dir, f"{args.dataset}_removal_verified.json")
    if verify_removal(dataset, removal_verify_path):
        print("移除验证完成")
    
    # Step 5: 验证无法求解
    print("\nStep 5: 验证无法求解")
    dataset = read_json(removal_verify_path)
    final_path = os.path.join(output_dir, f"{args.dataset}_final.json")
    if verify_unsolvable(dataset, final_path):
        print("无法求解验证完成")
    
    # Step 6: 筛选有效数据
    print("\nStep 6: 筛选有效数据")
    filter_valid_data(final_path)
    
    print("\n" + "="*50)
    print("数据集构造完成！")
    print("="*50)

if __name__ == "__main__":
    construction_workflow()