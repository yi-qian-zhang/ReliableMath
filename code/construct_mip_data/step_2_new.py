#!/usr/bin/env python3
"""
输入数据:原始数学问题 (question)+标准答案 (ground_truth)  +难度标签 (difficulty)

Step 1. GPT-4o提取条件并改写 (extract_and_generate_variants): 
   → 让LLM提取该问题中的[关键条件]，改写剩余内容为{缺省问题},[关键条件]+{缺省问题}组成pairs
   ↓

生成多个 removal_variants (移除变体)

Step 2.验证缺省问题 (verify_incomplete_questions_multi_attempt) 
   → distill 7B\qwen3-8B回答8次 ，回答对了就通过，回答错了就继续试到8次
   → 给模型：缺省问题(incomplete_question) + 被移除的[关键条件] （removed_condition）

   → 调用 Judge（gpt-4o-mini） 判断等价性：                

​        答案 = ground_truth → 保留（条件必要）
​        答案 ≠ ground_truth → 丢弃（条件非必要）
   ↓
最终数据集：只包含移除关键条件后的有效缺省问题:-------------------------每个难度500条
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
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="MIP Dataset Construction - 2 Steps")
parser.add_argument("--model", default="gpt-4o", help="Model for extraction/rewrite")
parser.add_argument("--verify_model", default="deepseek-r1-distill-qwen-7b", help="Model for verification")
parser.add_argument("--judge_model", default="gpt-4o-mini", help="Model for LLM-as-Judge")
parser.add_argument("--data_dir", default="data/solve", help="Input directory")
parser.add_argument("--output_dir", default="data/construct_mip_data", help="Output directory")
parser.add_argument("--prompt_dir", default="prompt/construct_mip_data", help="Prompt directory")
parser.add_argument("--dataset", default="polaris_easy_20", help="Dataset name")
parser.add_argument("--temperature", default=0.9, type=float, help="Temperature for verification")
parser.add_argument("--max_attempts", default=8, type=int, help="Max attempts for verification")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 5 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items")
args = parser.parse_args()

# Load API config
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

def record_tokens(data, model_type, prompt_tokens, completion_tokens):
    """
    根据模型类型记录 token 使用量
    
    参数：
        data: 数据字典
        model_type: "gpt-4o" / "gpt-4o-mini" / "local"
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数
    """
    # 初始化字段
    if "gpt4o_prompt_lengths" not in data:
        data["gpt4o_prompt_lengths"] = []
        data["gpt4o_completion_lengths"] = []
    if "gpt4o_mini_prompt_lengths" not in data:
        data["gpt4o_mini_prompt_lengths"] = []
        data["gpt4o_mini_completion_lengths"] = []
    if "local_prompt_lengths" not in data:
        data["local_prompt_lengths"] = []
        data["local_completion_lengths"] = []
    
    # 根据模型类型记录
    if model_type == "gpt-4o":
        data["gpt4o_prompt_lengths"].append(prompt_tokens)
        data["gpt4o_completion_lengths"].append(completion_tokens)
    elif model_type == "gpt-4o-mini":
        data["gpt4o_mini_prompt_lengths"].append(prompt_tokens)
        data["gpt4o_mini_completion_lengths"].append(completion_tokens)
    elif model_type == "local":
        data["local_prompt_lengths"].append(prompt_tokens)
        data["local_completion_lengths"].append(completion_tokens)

# ============= API Functions =============

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    """
    调用 OpenAI-compatible API
    
    返回：
        (response_text, prompt_tokens, completion_tokens, model_type)
        model_type: "gpt-4o" / "gpt-4o-mini" / "local"
    """
    if model is None:
        model = args.model
    
    if model not in model_options:
        logging.error(f"Model {model} not found")
        return "", 0, 0, "unknown"
    
    model_name, key, url = random.choice(model_options[model])
    client = OpenAI(api_key=key, base_url=url)
    
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": input_prompt})
    
    prompt_text = (persona + "\n" if persona else "") + input_prompt
    prompt_tokens = count_tokens(prompt_text, model_name)
    
    # 判断模型类型
    is_local_model = "localhost" in url or "127.0.0.1" in url
    
    if is_local_model:
        model_type = "local"
    elif "gpt-4o-mini" in model_name.lower():
        model_type = "gpt-4o-mini"
    elif "gpt-4o" in model_name.lower():
        model_type = "gpt-4o"
    else:
        model_type = "gpt-4o"
    
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
            
            try:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            except:
                if is_local_model:
                    logging.debug(f"Local model: estimating tokens")
                completion_tokens = count_tokens(response_text, model_name)
            
            return response_text, prompt_tokens, completion_tokens, model_type
            
        except Exception as e:
            logging.warning(f'API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                wait_time = 3 if is_local_model else 10
                time.sleep(wait_time * (attempt + 1))
    
    return "", 0, 0, model_type

def parse_json_response(response, fallback=None):
    """简化的 JSON 解析"""
    try:
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
            
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}")
        logging.error(f"Full response: {response}")
    
    return fallback if fallback is not None else {}

# ============= Answer Processing =============

def extract_answer_tag(response):
    """从响应中提取答案（支持多种格式）"""
    try:
        # 方法 1: 优先查找 <answer> 标签
        start = response.find('<answer>')
        end = response.find('</answer>')
        
        if start >= 0 and end > start:
            answer = response[start + 8:end].strip()
            if '\\boxed{' in answer:
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
                if boxed_match:
                    return boxed_match.group(1).strip()
            return answer
        
        # 方法 2: 查找 $\boxed{...}$ 或 \boxed{...} 格式
        boxed_pattern = r'\$?\\boxed\{([^}]+)\}\$?'
        boxed_matches = re.findall(boxed_pattern, response)
        
        if boxed_matches:
            answer = boxed_matches[-1].strip()
            answer = answer.replace('$', '').strip()
            return answer
        
        # 方法 3: 查找常见的答案标记
        answer_patterns = [
            r'[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$)',
            r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
            r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                answer = match.group(1).strip()
                if '\\boxed{' in answer:
                    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
                    if boxed_match:
                        return boxed_match.group(1).strip()
                return answer
        
        return None
        
    except Exception as e:
        logging.error(f"Failed to extract answer: {e}")
        return None

def judge_answer_equivalence(question, model_answer, ground_truth):
    """使用 LLM-as-Judge 判断答案等价性"""
    prompt_path = os.path.join(args.prompt_dir, "judge_equivalence.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Judge prompt not found: {prompt_path}")
        return False, 0, 0, "unknown"
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        question=question,
        model_answer=model_answer,
        ground_truth=ground_truth
    )
    
    response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        input_prompt,
        persona="You are an expert mathematical equivalence judge.",
        model=args.judge_model,
        temperature=0.0
    )
    
    response_lower = response.strip().lower()
    
    if 'true' in response_lower and 'false' not in response_lower:
        result = True
    elif response_lower == 'true':
        result = True
    else:
        result = False
    
    return result, prompt_tokens, completion_tokens, model_type

# ============= Step 1: Extract and Generate Variants =============

def extract_and_generate_variants(data):
    """Step 1: 一次性提取条件并生成所有移除变体"""
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
    
    response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        input_prompt,
        persona="You are an expert at analyzing and rewriting mathematical problems.",
        model=args.model,
        temperature=0.0
    )
    
    # 记录 token
    record_tokens(data, model_type, prompt_tokens, completion_tokens)
    
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

# ============= Step 2: Verify with Multiple Attempts =============

def verify_incomplete_questions_multi_attempt(data):
    """Step 2: 验证"缺省问题 + 移除的条件"能否解出 ground_truth（最多 8 次尝试）"""
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
        
        # 记录所有尝试
        all_attempts = []
        success_at_attempt = None
        is_valid = False
        
        # 最多尝试 max_attempts 次
        for attempt_num in range(1, args.max_attempts + 1):
            input_prompt = prompt_template.format(
                incomplete_question=incomplete_question,
                removed_condition=removed_condition
            )
            
            response, prompt_tokens, completion_tokens, model_type = get_response_openai(
                input_prompt,
                persona="You are an expert mathematical problem solver.",
                model=args.verify_model,
                temperature=args.temperature
            )
            
            # 记录 token
            record_tokens(data, model_type, prompt_tokens, completion_tokens)
            
            # 提取答案
            model_answer = extract_answer_tag(response)
            
            # 判断是否正确
            if model_answer is None:
                is_correct = False
                judge_result = "no_answer_tag"
            else:
                is_correct, judge_prompt_tokens, judge_completion_tokens, judge_model_type = judge_answer_equivalence(
                    incomplete_question + " [With condition: " + removed_condition + "]",
                    model_answer,
                    ground_truth
                )
                judge_result = "equivalent" if is_correct else "not_equivalent"
                
                # 记录 judge token
                record_tokens(data, judge_model_type, judge_prompt_tokens, judge_completion_tokens)
            
            # 记录本次尝试
            attempt_record = {
                "attempt": attempt_num,
                "model_answer": model_answer if model_answer else "N/A",
                "judge_result": judge_result,
                "is_correct": is_correct
            }
            all_attempts.append(attempt_record)
            
            # 如果答对了，立即停止
            if is_correct:
                success_at_attempt = attempt_num
                is_valid = True
                logging.info(f"ID {variant['variant_id']}: ✓ VALID at attempt {attempt_num}/{args.max_attempts} (answer: {model_answer[:30]}...)")
                break
            else:
                logging.info(f"ID {variant['variant_id']}: Attempt {attempt_num}/{args.max_attempts} failed (answer: {model_answer[:30] if model_answer else 'N/A'}...)")
        
        # 如果8次都失败
        if not is_valid:
            logging.info(f"ID {variant['variant_id']}: ✗ INVALID - All {args.max_attempts} attempts failed")
        
        # 保存验证结果
        variant["verification"] = {
            "total_attempts": len(all_attempts),
            "success_at_attempt": success_at_attempt,
            "is_valid": is_valid,
            "all_attempts": all_attempts,
            "ground_truth": ground_truth
        }
    
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
    
    # 分别统计三类模型的 token
    total_gpt4o_prompt = sum(sum(d.get("gpt4o_prompt_lengths", [])) for d in dataset)
    total_gpt4o_completion = sum(sum(d.get("gpt4o_completion_lengths", [])) for d in dataset)
    
    total_gpt4o_mini_prompt = sum(sum(d.get("gpt4o_mini_prompt_lengths", [])) for d in dataset)
    total_gpt4o_mini_completion = sum(sum(d.get("gpt4o_mini_completion_lengths", [])) for d in dataset)
    
    total_local_prompt = sum(sum(d.get("local_prompt_lengths", [])) for d in dataset)
    total_local_completion = sum(sum(d.get("local_completion_lengths", [])) for d in dataset)
    
    total_original = len(dataset)
    total_variants = 0
    valid_variants = 0
    
    # 统计尝试次数分布
    attempt_distribution = {}
    
    for data in dataset:
        for variant in data.get("removal_variants", []):
            total_variants += 1
            
            verification = variant.get("verification", {})
            
            # 只保留有效的 pair（加回条件后能解出 ground_truth）
            if verification.get("is_valid", False):
                success_attempt = verification.get("success_at_attempt", 0)
                attempt_distribution[success_attempt] = attempt_distribution.get(success_attempt, 0) + 1
                
                valid_item = {
                    "id": variant["variant_id"],
                    "data_source": data.get("data_source", ""),
                    "difficulty": data.get("difficulty", ""),
                    "transformation_type": "condition_removal",
                    "original_question": data["question"],
                    "ground_truth": data.get("ground_truth", ""),
                    "removed_condition": variant["removed_condition"],
                    "removed_condition_index": variant["removed_condition_index"],
                    "remaining_conditions": variant["remaining_conditions"],
                    "incomplete_question": variant["incomplete_question"],
                    "verification": verification,
                    "original_id": data["id"]
                }
                valid_data.append(valid_item)
                valid_variants += 1
    
    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)
    
    # Statistics
    print("\n" + "="*70)
    print("MISSING INFORMATION PROBLEM (MIP) DATASET STATISTICS")
    print("="*70)
    print(f"Original problems: {total_original}")
    print(f"\nTotal removal variants generated: {total_variants}")
    print(f"Valid removal variants (condition necessary): {valid_variants}")
    if total_variants > 0:
        print(f"Success rate: {valid_variants / total_variants * 100:.2f}%")
    
    print(f"\nAttempt Distribution (when successful):")
    for attempt in sorted(attempt_distribution.keys()):
        count = attempt_distribution[attempt]
        print(f"  Attempt {attempt}: {count} variants ({count/valid_variants*100:.1f}%)")
    
    # 单价（每 1M tokens）
    gpt4o_prompt_rate = 2.5
    gpt4o_completion_rate = 10.0
    gpt4o_mini_prompt_rate = 0.15
    gpt4o_mini_completion_rate = 0.6

    # GPT-4o Token 统计
    print(f"\n💰 GPT-4o Token Usage:")
    print(f"  Prompt: {total_gpt4o_prompt:,}")
    print(f"  Completion: {total_gpt4o_completion:,}")
    print(
        f"  Cost = {total_gpt4o_prompt}/1e6*{gpt4o_prompt_rate} "
        f"+ {total_gpt4o_completion}/1e6*{gpt4o_completion_rate} "
        f"= ${total_gpt4o_prompt/1e6*gpt4o_prompt_rate + total_gpt4o_completion/1e6*gpt4o_completion_rate:.6f}"
    )

    # GPT-4o-mini Token 统计
    print(f"\n💰 GPT-4o-mini Token Usage:")
    print(f"  Prompt: {total_gpt4o_mini_prompt:,}")
    print(f"  Completion: {total_gpt4o_mini_completion:,}")
    print(
        f"  Cost = {total_gpt4o_mini_prompt}/1e6*{gpt4o_mini_prompt_rate} "
        f"+ {total_gpt4o_mini_completion}/1e6*{gpt4o_mini_completion_rate} "
        f"= ${total_gpt4o_mini_prompt/1e6*gpt4o_mini_prompt_rate + total_gpt4o_mini_completion/1e6*gpt4o_mini_completion_rate:.6f}"
    )

    
    # 本地模型 Token 统计
    print(f"\n🖥️  Local Model Token Usage:")
    print(f"  Prompt: {total_local_prompt:,}")
    print(f"  Completion: {total_local_completion:,}")
    
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
    print("MISSING INFORMATION PROBLEM (MIP) CONSTRUCTION PIPELINE - 2 STEPS")
    print("="*70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Prompt: {args.prompt_dir}")
    print(f"Model (extract/rewrite): {args.model}")
    print(f"Model (verify): {args.verify_model}")
    print(f"Model (judge): {args.judge_model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max attempts: {args.max_attempts}")
    print(f"Items: {len(dataset)}")
    if args.force:
        print(f"Mode: FORCE (reprocessing all)")
    print("="*70)
    
    # Step 1: Extract and Generate Variants
    print("\n[1/2] Extracting conditions and generating removal variants")
    extract_path = os.path.join(output_dir, f"{args.dataset}_variants.json")
    process_with_jsonl(dataset, extract_path, extract_and_generate_variants, "Generating variants")
    
    # Step 2: Verify with Multiple Attempts
    print(f"\n[2/2] Verifying incomplete questions (max {args.max_attempts} attempts)")
    dataset = read_json(extract_path)
    final_path = os.path.join(output_dir, f"{args.dataset}_final.json")
    process_with_jsonl(dataset, final_path, verify_incomplete_questions_multi_attempt, "Verifying")
    
    # Filter
    print("\n[3/2] Filtering valid data")
    filter_valid_data(final_path)
    
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()