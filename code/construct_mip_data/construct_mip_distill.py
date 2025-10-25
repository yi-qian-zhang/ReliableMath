#!/usr/bin/env python3
"""
MIP (Missing Information Problem) Construction Pipeline
移除关键条件使数学问题不可解
# 1.确认original question可解出groundtruth
# 2.remove --》改写后的缺省问题  + 缺省条件
# 3.直接给缺省问题+ 缺省条件 让模型回答 如果与groundtruth一样就保留
"""

import os
import json
import time
import logging
import argparse
from openai import OpenAI
import random
from tqdm import tqdm
import tiktoken
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="MIP Dataset Construction")
parser.add_argument("--model", default="gpt-4o", help="Model for extraction/rewrite")
parser.add_argument("--verify_model", default="DeepSeek-R1-Distill-Qwen-7B", help="Model for verification")
parser.add_argument("--judge_model", default="gpt-4o", help="Model for judge")
parser.add_argument("--data_dir", default="data/solve", help="Input directory")
parser.add_argument("--output_dir", default="data/construct_mip_distill", help="Output directory")
parser.add_argument("--prompt_dir", default="prompt/construct_mip_distill", help="Prompt directory")
parser.add_argument("--dataset", default="polaris_easy_20", help="Dataset name")
parser.add_argument("--temperature", default=0.0, type=float, help="Temperature")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 5 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items")
args = parser.parse_args()

# Load API config - 相对于运行目录 ~/ReliableMath
try:
    api_config_path = "data/api_keys.json"
    model_options = json.load(open(api_config_path, "r"))
except FileNotFoundError:
    logging.error(f"api_keys.json not found at {api_config_path}!")
    logging.error(f"Please make sure you run this script from ~/ReliableMath directory")
    exit(1)

# ============= Utility Functions =============

def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data

def dump_jsonl(data, filepath, append=False):
    mode = 'a' if append else 'w'
    try:
        json_str = json.dumps(data, ensure_ascii=False)
    except:
        return False
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(json_str + '\n')
        f.flush()
    return True

def count_tokens(text, model_name="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except:
        return len(text) // 4

# ============= API Functions =============

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    if model is None:
        model = args.model
    
    if model not in model_options:
        logging.error(f"Model {model} not found in api_keys.json")
        logging.error(f"Available models: {list(model_options.keys())}")
        return "", 0, 0
    
    model_name, key, url = random.choice(model_options[model])
    client = OpenAI(api_key=key, base_url=url)
    
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": input_prompt})
    
    prompt_text = (persona + "\n" if persona else "") + input_prompt
    prompt_tokens = count_tokens(prompt_text, model_name)
    
    # 判断是否为本地模型
    is_local_model = "localhost" in url or "127.0.0.1" in url
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
                stream=False
            )
            
            response_text = completion.choices[0].message.content
            
            # 处理 token 统计
            try:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            except:
                # 本地模型可能不返回 usage，使用估算
                if is_local_model:
                    logging.debug(f"Local model: estimating tokens")
                completion_tokens = count_tokens(response_text, model_name)
            
            return response_text, prompt_tokens, completion_tokens
            
        except Exception as e:
            logging.warning(f'API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                # 本地模型重试间隔短一些
                wait_time = 3 if is_local_model else 10
                time.sleep(wait_time * (attempt + 1))
    
    logging.error(f"All {max_retries} API call attempts failed")
    return "", 0, 0

def parse_json_response(response, fallback=None):
    """简化的 JSON 解析"""
    try:
        # 尝试找到 JSON 数组（优先）
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        
        # 尝试找到 JSON 对象
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
            
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}")
        logging.error(f"Response (first 500 chars): {response[:500]}")
    
    return fallback if fallback is not None else {}

# ============= Answer Processing =============

def extract_answer_tag(response):
    """从响应中提取 <answer> 标签内容，兼容多种格式"""
    try:
        # 方法 1: 标准 <answer> 标签
        start = response.find('<answer>')
        end = response.find('</answer>')
        
        if start >= 0 and end > start:
            answer = response[start + 8:end].strip()
            return answer
        
        # 方法 2: DeepSeek 可能使用其他格式，尝试提取最后的答案
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Step'):
                continue
            
            # 移除常见前缀
            for prefix in ['Answer:', 'Final answer:', 'The answer is', 'Therefore,', 'Thus,']:
                if prefix.lower() in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        return parts[-1].strip()
                    return line.replace(prefix, '').strip()
            
            # 如果这行看起来像答案（简短且不是句子）
            if len(line) < 100 and not line.endswith('.'):
                return line
        
        # 如果都失败了，返回 None
        return None
        
    except Exception as e:
        logging.error(f"Failed to extract answer tag: {e}")
        return None

def judge_answer_equivalence(question, model_answer, ground_truth):
    """使用 LLM-as-Judge 判断答案等价性"""
    prompt_path = os.path.join(args.prompt_dir, "judge_equivalence.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Judge prompt not found: {prompt_path}")
        return False
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        question=question,
        model_answer=model_answer,
        ground_truth=ground_truth
    )
    
    response, _, _ = get_response_openai(
        input_prompt,
        persona="You are an expert mathematical equivalence judge.",
        model=args.judge_model,
        temperature=0.0
    )
    
    # 提取判断结果
    response_lower = response.strip().lower()
    
    # 判断是否为 true
    if 'true' in response_lower and 'false' not in response_lower:
        return True
    elif response_lower == 'true':
        return True
    else:
        return False

# ============= Step 0: Verify Original Solvability =============

def verify_original_solvable(data):
    """Step 0: 验证原始问题模型能否解出ground_truth"""
    prompt_path = os.path.join(args.prompt_dir, "verify_original.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["original_solvable"] = False
        data["skip_reason"] = "prompt_not_found"
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        question=data["question"]
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert mathematical problem solver.",
        model=args.verify_model,
        temperature=0.0
    )
    
    if "prompt_lengths" not in data:
        data["prompt_lengths"] = []
        data["completion_lengths"] = []
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # 提取 <answer> 标签内容
    model_answer = extract_answer_tag(response)
    ground_truth = str(data.get("ground_truth", "")).strip()
    
    # 如果提取不到答案，直接判定为不可解
    if model_answer is None:
        is_solvable = False
        judge_result = "no_answer_tag"
        logging.info(f"ID {data['id']}: SKIP - No <answer> tag found")
    else:
        # 使用 LLM-as-Judge 判断等价性
        is_solvable = judge_answer_equivalence(
            data["question"],
            model_answer,
            ground_truth
        )
        judge_result = "equivalent" if is_solvable else "not_equivalent"
        
        # 记录 judge 使用的 token（粗略估计）
        judge_tokens = count_tokens(data["question"] + model_answer + ground_truth)
        data["prompt_lengths"].append(judge_tokens)
        data["completion_lengths"].append(10)  # judge 响应很短
    
    # 在顶层添加 answer 字段，方便查看
    data["answer"] = model_answer if model_answer else "N/A"
    
    data["original_verification"] = {
        "model_answer": model_answer if model_answer else "N/A",
        "ground_truth": ground_truth,
        "judge_result": judge_result,
        "is_solvable": is_solvable,
        "full_response": response  # 不限制长度
    }
    data["original_solvable"] = is_solvable
    
    if not is_solvable:
        data["skip_reason"] = "original_unsolvable"
        logging.info(f"ID {data['id']}: SKIP - Original problem unsolvable (judge: {judge_result}, answer: {model_answer[:50] if model_answer else 'N/A'}...)")
    else:
        answer_preview = model_answer[:50] if model_answer else "N/A"
        logging.info(f"ID {data['id']}: Original problem solvable ✓ (answer: {answer_preview}...)")
    
    return data

# ============= Step 1: Extract and Generate Variants =============

def extract_and_generate_variants(data):
    """Step 1: 一次性提取条件并生成所有移除变体"""
    # 如果原始问题不可解，跳过
    if not data.get("original_solvable", False):
        data["removal_variants"] = []
        return data
    
    prompt_path = os.path.join(args.prompt_dir, "extract_and_remove.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["removal_variants"] = []
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        original_question=data["question"],
        ground_truth=data.get("ground_truth", "")
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at analyzing and rewriting mathematical problems.",
        model=args.model,
        temperature=0.0
    )
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Parse response - 期望得到一个变体列表
    parsed = parse_json_response(response, {"variants": []})
    
    # 处理两种可能的 JSON 格式
    if isinstance(parsed, list):
        variants_data = parsed
    else:
        variants_data = parsed.get("variants", [])
    
    removal_variants = []
    
    for i, variant_data in enumerate(variants_data):
        # 清理 incomplete_question
        incomplete_question = variant_data.get("incomplete_question", "").strip()
        
        # Remove common prefixes
        for prefix in ["Rewritten Problem:", "Incomplete Problem:", "Problem:", "**Problem:**"]:
            if prefix in incomplete_question:
                incomplete_question = incomplete_question.split(prefix)[-1].strip()
        
        incomplete_question = incomplete_question.replace("**", "").strip()
        
        # Remove quotes if present
        if incomplete_question.startswith('"') and incomplete_question.endswith('"'):
            incomplete_question = incomplete_question[1:-1].strip()
        
        variant = {
            "variant_id": f"{data['id']}_remove_{i}",
            "removed_condition_index": i,
            "removed_condition": variant_data.get("removed_condition", ""),
            "remaining_conditions": variant_data.get("remaining_conditions", []),
            "incomplete_question": incomplete_question
        }
        
        removal_variants.append(variant)
    
    data["removal_variants"] = removal_variants
    
    logging.info(f"ID {data['id']}: Generated {len(removal_variants)} removal variants")
    
    return data

# ============= Step 2: Verify Incomplete Questions =============

def verify_incomplete_questions(data):
    """Step 2: 验证"缺省问题 + 移除的条件"能否解出 ground_truth"""
    # 如果原始问题不可解，跳过
    if not data.get("original_solvable", False):
        return data
    
    # 使用 verify_with_condition.txt
    prompt_path = os.path.join(args.prompt_dir, "verify_with_condition.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    ground_truth = str(data.get("ground_truth", "")).strip()
    
    for variant in data.get("removal_variants", []):
        incomplete_question = variant["incomplete_question"]
        removed_condition = variant["removed_condition"]
        
        # 构造输入：缺省问题 + 移除的条件
        input_prompt = prompt_template.format(
            incomplete_question=incomplete_question,
            removed_condition=removed_condition
        )
        
        response, prompt_tokens, completion_tokens = get_response_openai(
            input_prompt,
            persona="You are an expert mathematical problem solver.",
            model=args.verify_model,
            temperature=0.0
        )
        
        data["prompt_lengths"].append(prompt_tokens)
        data["completion_lengths"].append(completion_tokens)
        
        # 提取 <answer> 标签
        model_answer = extract_answer_tag(response)
        
        # 在变体中添加 answer 字段
        variant["answer"] = model_answer if model_answer else "N/A"
        
        # 保存完整响应（限制长度）
        variant["full_response"] = response[:1000] if len(response) > 1000 else response
        
        # 如果提取不到答案，认为无效
        if model_answer is None:
            is_correct = False
            judge_result = "no_answer_tag"
        else:
            # 使用 LLM-as-Judge 判断是否等价于 ground_truth
            is_correct = judge_answer_equivalence(
                incomplete_question + " [With condition: " + removed_condition + "]",
                model_answer,
                ground_truth
            )
            judge_result = "equivalent" if is_correct else "not_equivalent"
            
            # 记录 judge token
            judge_tokens = count_tokens(incomplete_question + removed_condition + model_answer + ground_truth)
            data["prompt_lengths"].append(judge_tokens)
            data["completion_lengths"].append(10)
        
        # CRITICAL 验证逻辑：
        # 给回条件后，答案 = ground_truth → 说明条件是关键的 → 保留
        # 给回条件后，答案 ≠ ground_truth → 说明条件不重要 → 丢弃
        variant["verification"] = {
            "model_answer": model_answer if model_answer else "N/A",
            "ground_truth": ground_truth,
            "judge_result": judge_result,
            "answers_match": is_correct,
            "is_valid": is_correct  # 答案正确才是有效的
        }
        
        status = "VALID ✓" if is_correct else "INVALID ✗"
        answer_preview = model_answer[:30] if model_answer else "N/A"
        logging.info(f"  {variant['variant_id']}: {status}")
        logging.info(f"    Model: {answer_preview}... | Ground truth: {ground_truth}")
        logging.info(f"    Condition: {removed_condition[:60]}...")
    
    return data

# ============= Pipeline Functions =============

def process_with_jsonl(dataset, output_path, process_func, desc):
    total_len = len(dataset)
    jsonl_path = output_path.replace('.json', '.jsonl')
    
    existing_data = []
    if os.path.exists(jsonl_path):
        existing_data = read_jsonl(jsonl_path)
        if existing_data:
            saved_ids = {item['id'] for item in existing_data}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
            logging.info(f"{desc}: Continuing from {len(existing_data)} items")
    elif os.path.exists(output_path):
        try:
            existing_data = read_json(output_path)
            saved_ids = {item['id'] for item in existing_data}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except:
            pass
    
    if not dataset:
        logging.info(f"{desc}: All items processed")
        return True
    
    with tqdm(total=len(dataset), desc=desc) as t:
        for data in dataset:
            try:
                processed_data = process_func(data)
                if processed_data:
                    t.update(1)
                    dump_jsonl(processed_data, jsonl_path, append=True)
            except Exception as e:
                logging.error(f"Error processing {data.get('id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                t.update(1)
                continue
    
    all_data = existing_data + read_jsonl(jsonl_path)[len(existing_data):]
    
    if all_data:
        write_json(output_path, all_data)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
    
    return len(all_data) == total_len

def filter_valid_data(final_path):
    """筛选有效的缺省问题"""
    dataset = read_json(final_path)
    valid_data = []
    
    total_prompt = sum(sum(d.get("prompt_lengths", [])) for d in dataset)
    total_completion = sum(sum(d.get("completion_lengths", [])) for d in dataset)
    
    total_original = len(dataset)
    solvable_original = sum(1 for d in dataset if d.get("original_solvable", False))
    total_variants = 0
    valid_variants = 0
    
    for data in dataset:
        # 只处理原始可解的问题
        if not data.get("original_solvable", False):
            continue
        
        for variant in data.get("removal_variants", []):
            total_variants += 1
            
            verification = variant.get("verification", {})
            
            # 只保留有效的 pair（加回条件后能解出 ground_truth）
            if verification.get("is_valid", False):
                valid_item = {
                    "id": variant["variant_id"],
                    "data_source": data.get("data_source", ""),
                    "difficulty": data.get("difficulty", ""),
                    "transformation_type": "condition_removal",
                    "original_question": data["question"],
                    "original_answer": data.get("answer", "N/A"),  # 原始问题的答案
                    "ground_truth": data.get("ground_truth", ""),
                    "removed_condition": variant["removed_condition"],
                    "removed_condition_index": variant["removed_condition_index"],
                    "remaining_conditions": variant["remaining_conditions"],
                    "incomplete_question": variant["incomplete_question"],
                    "answer_with_condition": variant.get("answer", "N/A"),  # 加回条件后的答案
                    "verification": verification,
                    "original_id": data["id"]
                }
                valid_data.append(valid_item)
                valid_variants += 1
    
    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)
    
    # Calculate token usage for valid items only
    total_valid_prompt = sum(
        sum(d.get("prompt_lengths", [])) 
        for d in dataset 
        if d.get("original_solvable", False) and any(
            v.get("verification", {}).get("is_valid", False) 
            for v in d.get("removal_variants", [])
        )
    )
    total_valid_completion = sum(
        sum(d.get("completion_lengths", [])) 
        for d in dataset 
        if d.get("original_solvable", False) and any(
            v.get("verification", {}).get("is_valid", False) 
            for v in d.get("removal_variants", [])
        )
    )
    
    # Statistics
    print("\n" + "="*70)
    print("MISSING INFORMATION PROBLEM (MIP) DATASET STATISTICS")
    print("="*70)
    print(f"Original problems: {total_original}")
    print(f"Solvable by model: {solvable_original} ({solvable_original/total_original*100:.1f}%)")
    print(f"Unsolvable (skipped): {total_original - solvable_original}")
    print(f"\nTotal removal variants generated: {total_variants}")
    print(f"Valid removal variants (condition necessary): {valid_variants}")
    if total_variants > 0:
        print(f"Success rate: {valid_variants / total_variants * 100:.2f}%")
    
    print(f"\nToken Usage (ALL):")
    print(f"  Prompt: {total_prompt:,}")
    print(f"  Completion: {total_completion:,}")
    print(f"  Total: {total_prompt + total_completion:,}")
    
    print(f"\nToken Usage (VALID only):")
    print(f"  Prompt: {total_valid_prompt:,}")
    print(f"  Completion: {total_valid_completion:,}")
    print(f"  Total: {total_valid_prompt + total_valid_completion:,}")
    
    if total_prompt + total_completion > 0:
        efficiency = ((total_valid_prompt + total_valid_completion) / 
                     (total_prompt + total_completion)) * 100
        print(f"\nToken efficiency: {efficiency:.2f}%")
    
    print(f"\nOutput: {output_path}")
    print("="*70)

# ============= Main Workflow =============

def construction_workflow():
    # 直接使用 args 中的路径（相对于 ~/ReliableMath）
    input_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    output_dir = args.output_dir
    
    if not os.path.exists(input_path):
        logging.error(f"Input not found: {input_path}")
        logging.error(f"Current working directory: {os.getcwd()}")
        logging.error(f"Please make sure you run this script from ~/ReliableMath directory")
        return
    
    dataset = read_json(input_path)
    
    if args.test_mode:
        dataset = dataset[:5]
        logging.info("TEST MODE: First 5 items")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Force cleanup
    if args.force:
        logging.info("Force mode: Cleaning up existing intermediate files...")
        for pattern in [f"{args.dataset}_*.json", f"{args.dataset}_*.jsonl"]:
            for file in glob.glob(os.path.join(output_dir, pattern)):
                try:
                    os.remove(file)
                    logging.info(f"Removed: {file}")
                except Exception as e:
                    logging.warning(f"Could not remove {file}: {e}")
        logging.info("Cleanup completed.")
    
    print("="*70)
    print("MISSING INFORMATION PROBLEM (MIP) CONSTRUCTION PIPELINE")
    print("="*70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Prompt: {args.prompt_dir}")
    print(f"Model (extract/rewrite): {args.model}")
    print(f"Model (verify): {args.verify_model}")
    print(f"Model (judge): {args.judge_model}")
    print(f"Items: {len(dataset)}")
    if args.force:
        print(f"Mode: FORCE (reprocessing all)")
    print("="*70)
    
    # Step 0: Verify Original Solvability
    print("\n[0/3] Verifying original problem solvability")
    verify_path = os.path.join(output_dir, f"{args.dataset}_verify_original.json")
    process_with_jsonl(dataset, verify_path, verify_original_solvable, "Verifying original")
    
    # Step 1: Extract and Generate Variants
    print("\n[1/3] Extracting conditions and generating removal variants")
    dataset = read_json(verify_path)
    extract_path = os.path.join(output_dir, f"{args.dataset}_variants.json")
    process_with_jsonl(dataset, extract_path, extract_and_generate_variants, "Generating variants")
    
    # Step 2: Verify Incomplete Questions
    print("\n[2/3] Verifying incomplete questions with conditions")
    dataset = read_json(extract_path)
    final_path = os.path.join(output_dir, f"{args.dataset}_final.json")
    process_with_jsonl(dataset, final_path, verify_incomplete_questions, "Verifying with condition")
    
    # Filter
    print("\n[3/3] Filtering valid data")
    filter_valid_data(final_path)
    
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()