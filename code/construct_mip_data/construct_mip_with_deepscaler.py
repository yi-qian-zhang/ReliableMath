#!/usr/bin/env python3
"""
输入数据:原始数学问题 (question)+标准答案 (ground_truth)  +难度标签 (difficulty)

Step 1. GPT-4o提取条件并改写 (extract_and_generate_variants): 
   → 让LLM提取该问题中的[关键条件]，改写剩余内容为{缺省问题},[关键条件]+{缺省问题}组成pairs
   ↓

生成多个 removal_variants (移除变体)

Step 2.验证缺省问题 (verify_incomplete_questions_with_two_rounds) 
   → 验证 A：缺省条件下问题不可解
        给模型缺省问题(incomplete_question)，验证它在缺少[关键条件]的情况下能否解出ground_truth
        vLLM sampling 8次，用 Deepscaler 判断等价性：                
        全都 ≠ ground_truth → 通过验证A（条件必要）
        至少1个 = ground_truth → 丢弃（条件非必要）
   
   → 验证 B：条件完整拼装的情况下问题可解
        给模型缺省问题(incomplete_question) + 被移除的[关键条件] (removed_condition)
        vLLM sampling 8次，用 Deepscaler 判断等价性：                
        至少1个 = ground_truth → 保留（条件充分）
        全都 ≠ ground_truth → 丢弃（条件不充分）
   ↓
最终数据集：只包含移除关键条件后的有效缺省问题
"""
import sys
import os
# 将 code/ 目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# =========================================================
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============= 新增：导入 Deepscaler 模块 =============
from deepscaler.rewards.math_utils.utils import (
    grade_answer_mathd, 
    grade_answer_sympy, 
    extract_answer  # 自带的提取boxed{}逻辑
)
from deepscaler.system_prompts import ORM_PROMPT
# ====================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="MIP Dataset Construction - 2 Steps with Sampling + Deepscaler")
parser.add_argument("--model", default="gpt-4o", help="Model for extraction/rewrite")
parser.add_argument("--verify_model", default="deepseek-r1-distill-qwen-7b", help="Model for verification")
parser.add_argument("--judge_model", default="gpt-4o-mini", help="Model for LLM-as-Judge (ORM fallback)")
parser.add_argument("--data_dir", default="data/solve", help="Input directory")
parser.add_argument("--output_dir", default="data/construct_mip_data", help="Output directory")
parser.add_argument("--prompt_dir", default="prompt/construct_mip_with_deepscaler", help="Prompt directory")
parser.add_argument("--dataset", default="polaris_easy_20", help="Dataset name")
parser.add_argument("--temperature", default=0.9, type=float, help="Temperature for verification")
parser.add_argument("--max_attempts", default=8, type=int, help="Max attempts for verification")
parser.add_argument("--threads", default=8, type=int, help="Number of parallel threads")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 5 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items")
# ============= 新增：ORM 开关 =============
parser.add_argument("--use_math_orm", action='store_true', help="Enable LLM ORM for answer verification when heuristics fail")
# =======================================
args = parser.parse_args()

# Load API config
try:
    api_config_path = "data/api_keys.json"
    model_options = json.load(open(api_config_path, "r"))
except FileNotFoundError:
    logging.error(f"api_keys.json not found at {api_config_path}!")
    logging.error(f"Please make sure you run this script from ~/ReliableMath directory")
    exit(1)

# 全局锁，用于保护 JSONL 文件写入
jsonl_write_lock = threading.Lock()

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
    """线程安全的 JSONL 写入"""
    mode = 'a' if append else 'w'
    try:
        json_str = json.dumps(data, ensure_ascii=False)
    except:
        return False
    
    with jsonl_write_lock:
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
        model_type: "gpt-4o" / "gpt-4o-mini" / "local" / "heuristic"
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
    # ============= 新增：heuristic 统计 =============
    if "heuristic_count" not in data:
        data["heuristic_count"] = 0
    # =============================================
    
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
    elif model_type == "heuristic":
        # 启发式方法是免费的，只记录使用次数
        data["heuristic_count"] += 1

# ============= API Functions =============

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    """
    调用 OpenAI-compatible API（单个响应）
    
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
                max_tokens=4096,
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

def get_response_openai_with_sampling(input_prompt, persona="", model=None, temperature=0.0, n=1):
    """
    调用 OpenAI-compatible API，支持 sampling（一次生成多个候选答案）
    
    参数：
        n: 生成的候选答案数量（默认 1）
    
    返回：
        {
            "candidates": [答案1, 答案2, ..., 答案n],
            "prompt_tokens": xxx,
            "completion_tokens": xxx,
            "model_type": "gpt-4o" / "gpt-4o-mini" / "local"
        }
        或者失败时返回 None
    """
    if model is None:
        model = args.model
    
    if model not in model_options:
        logging.error(f"Model {model} not found")
        return None
    
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
                n=n,  
                max_tokens=4096,
                stream=False
            )
            
            # 提取所有候选答案
            candidates = [choice.message.content for choice in completion.choices]
            
            # 获取 token 统计
            try:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            except:
                if is_local_model:
                    logging.debug(f"Local model: estimating tokens")
                # 估算所有候选的总 token
                completion_tokens = sum(count_tokens(text, model_name) for text in candidates)
            
            return {
                "candidates": candidates,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_type": model_type
            }
            
        except Exception as e:
            logging.warning(f'API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                wait_time = 3 if is_local_model else 10
                time.sleep(wait_time * (attempt + 1))
    
    return None

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

# def extract_answer_tag(response):
#     """从响应中提取答案（支持多种格式）"""
#     try:
#         # 方法 1: 优先查找 <answer> 标签
#         start = response.find('<answer>')
#         end = response.find('</answer>')
        
#         if start >= 0 and end > start:
#             answer = response[start + 8:end].strip()
#             if '\\boxed{' in answer:
#                 boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
#                 if boxed_match:
#                     return boxed_match.group(1).strip()
#             return answer
        
#         # 方法 2: 查找 $\boxed{...}$ 或 \boxed{...} 格式
#         boxed_pattern = r'\$?\\boxed\{([^}]+)\}\$?'
#         boxed_matches = re.findall(boxed_pattern, response)
        
#         if boxed_matches:
#             answer = boxed_matches[-1].strip()
#             answer = answer.replace('$', '').strip()
#             return answer
        
#         # 方法 3: 查找常见的答案标记
#         answer_patterns = [
#             r'[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$)',
#             r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
#             r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
#         ]
        
#         for pattern in answer_patterns:
#             match = re.search(pattern, response)
#             if match:
#                 answer = match.group(1).strip()
#                 if '\\boxed{' in answer:
#                     boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
#                     if boxed_match:
#                         return boxed_match.group(1).strip()
#                 return answer
        
#         return None
        
#     except Exception as e:
#         logging.error(f"Failed to extract answer: {e}")
#         return None

# ============= 修改：使用 Deepscaler 的判断逻辑 =============
def judge_answer_equivalence(question, model_answer, ground_truth):
    """
    使用 Deepscaler 的多层验证逻辑判断答案等价性
    
    返回：
        (is_correct, prompt_tokens, completion_tokens, model_type)
        model_type: "heuristic" / "gpt-4o-mini" / "gpt-4o"
    """
    # ========== 第一层：启发式方法（免费）==========
    is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
    
    if is_correct:
        logging.debug(f"✓ Heuristic match: {model_answer} ≈ {ground_truth}")
        return True, 0, 0, "heuristic"
    
    # ========== 第二层：LLM ORM（如果启用）==========
    if args.use_math_orm:
        logging.debug(f"Heuristic failed, trying ORM: {model_answer} vs {ground_truth}")
        
        ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""
        
        input_prompt = ORM_USER_TEMPLATE.format(
            problem=question,
            answer_1=model_answer,
            answer_2=ground_truth
        )
        
        # 主裁判：使用配置的 judge_model (默认 gpt-4o-mini)
        try:
            response, prompt_tokens, completion_tokens, model_type = get_response_openai(
                input_prompt,
                persona=ORM_PROMPT,
                model=args.judge_model,
                temperature=0.0
            )
            
            if "[[YES]]" in response:
                logging.debug(f"✓ ORM confirmed: {model_answer} ≈ {ground_truth}")
                return True, prompt_tokens, completion_tokens, model_type
            else:
                logging.debug(f"✗ ORM rejected: {model_answer} ≠ {ground_truth}")
                return False, prompt_tokens, completion_tokens, model_type
                
        except Exception as e:
            logging.error(f"ORM call failed: {e}")
            return False, 0, 0, "unknown"
    
    # 启发式失败且未启用 ORM
    logging.debug(f"✗ No match: {model_answer} ≠ {ground_truth}")
    return False, 0, 0, "heuristic"
# =========================================================

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
    
    # Parse response
    parsed = parse_json_response(response, {"variants": []})
    
    if isinstance(parsed, list):
        variants_data = parsed
    else:
        variants_data = parsed.get("variants", [])
    
    # ============= 新增：推断所有条件 =============
    # 方法 1：从第一个 variant 推断
    all_conditions = []
    if variants_data:
        first_variant = variants_data[0]
        all_conditions = [first_variant.get("removed_condition", "")] + \
                        first_variant.get("remaining_conditions", [])
    
    # 方法 2：或者从所有 variants 合并（更准确）
    all_conditions_set = set()
    for variant_data in variants_data:
        all_conditions_set.add(variant_data.get("removed_condition", ""))
        all_conditions_set.update(variant_data.get("remaining_conditions", []))
    
    all_conditions = sorted(list(all_conditions_set), key=lambda x: len(x), reverse=True)
    # ==========================================
    
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
    
    # ============= 新增：保存到数据中 =============
    data["all_extracted_conditions"] = all_conditions
    data["num_conditions_extracted"] = len(all_conditions)
    # ==========================================
    
    data["removal_variants"] = removal_variants
    
    logging.info(f"ID {data['id']}: Extracted {len(all_conditions)} conditions, generated {len(removal_variants)} removal variants")
    
    return data

# ============= Step 2: 两轮验证 =============
def verify_single_variant(data, variant, prompt_template_incomplete, prompt_template_complete, ground_truth):
    """
    验证单个变体（两轮验证）
    
    验证 A：不加条件 → 8次sampling → 全都≠ground_truth → ✓通过
    验证 B：加条件 → 8次sampling → 至少1个=ground_truth → ✓保留
    """
    incomplete_question = variant["incomplete_question"]
    removed_condition = variant["removed_condition"]
    
    # ========== 验证 A：缺省条件下问题不可解 ==========
    logging.info(f"ID {variant['variant_id']}: Starting Round A - Testing imcomplete question...")
    
    input_prompt_incomplete = prompt_template_incomplete.format(
        incomplete_question=incomplete_question
    )
    
    response_data_a = get_response_openai_with_sampling(
        input_prompt_incomplete,
        persona="You are an expert mathematical problem solver.",
        model=args.verify_model,
        temperature=args.temperature,
        n=args.max_attempts
    )
    
    if not response_data_a:
        logging.error(f"ID {variant['variant_id']}: Round A generation failed")
        variant["verification"] = {
            "round_a_passed": False,
            "round_b_passed": False,
            "is_valid": False,
            "round_a": {"total_attempts": 0, "all_attempts": []},
            "round_b": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    
    # 记录 Round A 的 token
    record_tokens(data, response_data_a["model_type"], 
                  response_data_a["prompt_tokens"], 
                  response_data_a["completion_tokens"])
    
    # 检查 Round A 的所有候选
    round_a_attempts = []
    round_a_has_correct = False
    
    for attempt_num, candidate_text in enumerate(response_data_a["candidates"], start=1):
        # model_answer = extract_answer_tag(candidate_text)  # ← 使用自己的提取逻辑函数
        model_answer = extract_answer(candidate_text) 
        
        if model_answer is None:
            is_correct = False
            judge_result = "no_answer_tag"
            judge_method = "none"
        else:
            is_correct, judge_prompt_tokens, judge_completion_tokens, judge_model_type = judge_answer_equivalence(
                incomplete_question,
                model_answer,
                ground_truth
            )
            
            if judge_model_type == "heuristic":
                judge_result = "heuristic_match" if is_correct else "heuristic_fail"
                judge_method = "heuristic"
            else:
                judge_result = "orm_match" if is_correct else "orm_fail"
                judge_method = "orm"
            
            record_tokens(data, judge_model_type, judge_prompt_tokens, judge_completion_tokens)
        
        # ============= 修改：添加完整响应记录 =============
        attempt_record = {
            "attempt": attempt_num,
            "full_response": candidate_text,  # ← 新增：保存完整生成内容
            "model_answer": model_answer if model_answer else "N/A",
            "judge_result": judge_result,
            "judge_method": judge_method,
            "is_correct": is_correct
        }
        # ==============================================
        round_a_attempts.append(attempt_record)
        
        if is_correct:
            round_a_has_correct = True
    
    # Round A 结果判定
    round_a_passed = not round_a_has_correct  # 全都不对才通过
    
    if round_a_passed:
        logging.info(f"ID {variant['variant_id']}: ✓ Round A PASSED - All {args.max_attempts} answers ≠ ground_truth")
    else:
        logging.info(f"ID {variant['variant_id']}: ✗ Round A FAILED - At least 1 answer = ground_truth (condition not necessary)")
        variant["verification"] = {
            "round_a_passed": False,
            "round_b_passed": False,
            "is_valid": False,
            "round_a": {
                "total_attempts": len(round_a_attempts),
                "all_attempts": round_a_attempts
            },
            "round_b": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    
    # ========== 验证 B：条件完整拼装的情况下问题可解 ==========
    logging.info(f"ID {variant['variant_id']}: Starting Round B - Testing WITH removed_condition...")
    
    input_prompt_complete = prompt_template_complete.format(
        incomplete_question=incomplete_question,
        removed_condition=removed_condition
    )
    
    response_data_b = get_response_openai_with_sampling(
        input_prompt_complete,
        persona="You are an expert mathematical problem solver.",
        model=args.verify_model,
        temperature=args.temperature,
        n=args.max_attempts
    )
    
    if not response_data_b:
        logging.error(f"ID {variant['variant_id']}: Round B generation failed")
        variant["verification"] = {
            "round_a_passed": True,
            "round_b_passed": False,
            "is_valid": False,
            "round_a": {
                "total_attempts": len(round_a_attempts),
                "all_attempts": round_a_attempts
            },
            "round_b": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    
    # 记录 Round B 的 token
    record_tokens(data, response_data_b["model_type"], 
                  response_data_b["prompt_tokens"], 
                  response_data_b["completion_tokens"])
    
    # 检查 Round B 的所有候选
    round_b_attempts = []
    round_b_has_correct = False
    success_at_attempt = None
    
    for attempt_num, candidate_text in enumerate(response_data_b["candidates"], start=1):
        # model_answer = extract_answer_tag(candidate_text)  # 旧函数
        model_answer = extract_answer(candidate_text) 
        
        if model_answer is None:
            is_correct = False
            judge_result = "no_answer_tag"
            judge_method = "none"
        else:
            is_correct, judge_prompt_tokens, judge_completion_tokens, judge_model_type = judge_answer_equivalence(
                incomplete_question + " [With condition: " + removed_condition + "]",
                model_answer,
                ground_truth
            )
            
            if judge_model_type == "heuristic":
                judge_result = "heuristic_match" if is_correct else "heuristic_fail"
                judge_method = "heuristic"
            else:
                judge_result = "orm_match" if is_correct else "orm_fail"
                judge_method = "orm"
            
            record_tokens(data, judge_model_type, judge_prompt_tokens, judge_completion_tokens)
        
        # ============= 修改：添加完整响应记录 =============
        attempt_record = {
            "attempt": attempt_num,
            "full_response": candidate_text,  # ← 新增：保存完整生成内容
            "model_answer": model_answer if model_answer else "N/A",
            "judge_result": judge_result,
            "judge_method": judge_method,
            "is_correct": is_correct
        }
        # ==============================================
        round_b_attempts.append(attempt_record)
        
        if is_correct and not round_b_has_correct:
            round_b_has_correct = True
            success_at_attempt = attempt_num
    
    # Round B 结果判定
    round_b_passed = round_b_has_correct  # 至少有1个对才通过
    
    if round_b_passed:
        logging.info(f"ID {variant['variant_id']}: ✓ Round B PASSED - Answer {success_at_attempt}/{args.max_attempts} = ground_truth (via {round_b_attempts[success_at_attempt-1]['judge_method']})")
    else:
        logging.info(f"ID {variant['variant_id']}: ✗ Round B FAILED - All {args.max_attempts} answers ≠ ground_truth")
    
    # 最终判定
    is_valid = round_a_passed and round_b_passed
    
    if is_valid:
        logging.info(f"ID {variant['variant_id']}: 🎉 VALID - Both rounds passed!")
    else:
        logging.info(f"ID {variant['variant_id']}: ✗ INVALID")
    
    # 保存验证结果
    variant["verification"] = {
        "round_a_passed": round_a_passed,
        "round_b_passed": round_b_passed,
        "is_valid": is_valid,
        "round_a": {
            "total_attempts": len(round_a_attempts),
            "all_attempts": round_a_attempts
        },
        "round_b": {
            "total_attempts": len(round_b_attempts),
            "success_at_attempt": success_at_attempt,
            "all_attempts": round_b_attempts
        },
        "ground_truth": ground_truth
    }
    
    return variant


def verify_incomplete_questions_with_two_rounds(data):
    """Step 2: 两轮验证（并行处理变体）"""
    prompt_path_incomplete = os.path.join(args.prompt_dir, "verify_without_condition.txt")
    prompt_path_complete = os.path.join(args.prompt_dir, "verify_with_condition.txt")
    
    if not os.path.exists(prompt_path_incomplete):
        logging.error(f"Prompt file not found: {prompt_path_incomplete}")
        return data
    
    if not os.path.exists(prompt_path_complete):
        logging.error(f"Prompt file not found: {prompt_path_complete}")
        return data
    
    with open(prompt_path_incomplete, 'r', encoding='utf-8') as f:
        prompt_template_incomplete = f.read()
    
    with open(prompt_path_complete, 'r', encoding='utf-8') as f:
        prompt_template_complete = f.read()
    
    ground_truth = str(data.get("ground_truth", "")).strip()
    variants = data.get("removal_variants", [])
    
    if not variants:
        return data
    
    # 并行处理所有变体
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_variant = {
            executor.submit(verify_single_variant, data, variant, 
                          prompt_template_incomplete, prompt_template_complete, ground_truth): variant
            for variant in variants
        }
        
        for future in as_completed(future_to_variant):
            try:
                verified_variant = future.result()
                # 更新 data 中对应的 variant
                variant_id = verified_variant["variant_id"]
                for i, v in enumerate(data["removal_variants"]):
                    if v["variant_id"] == variant_id:
                        data["removal_variants"][i] = verified_variant
                        break
            except Exception as e:
                variant = future_to_variant[future]
                logging.error(f"Error verifying {variant['variant_id']}: {e}")
                import traceback
                traceback.print_exc()
    
    return data

# ============= Pipeline Functions =============

def process_with_jsonl_parallel(dataset, output_path, process_func, desc):
    """并行处理数据集"""
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
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 提交所有任务
        future_to_data = {executor.submit(process_func, data): data for data in dataset}
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(dataset), desc=desc) as pbar:
            for future in as_completed(future_to_data):
                try:
                    processed_data = future.result()
                    if processed_data:
                        dump_jsonl(processed_data, jsonl_path, append=True)
                    pbar.update(1)
                except Exception as e:
                    data = future_to_data[future]
                    logging.error(f"Error processing {data.get('id', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)
    
    # 合并数据
    all_data = existing_data + read_jsonl(jsonl_path)[len(existing_data):]
    
    if all_data:
        all_data.sort(key=lambda x: x.get('id', 0))
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
    
    # ============= heuristic 统计 =============
    total_heuristic_count = sum(d.get("heuristic_count", 0) for d in dataset)
    # =========================================
    
    total_original = len(dataset)
    total_variants = 0
    valid_variants = 0
    
    # 统计验证结果分布
    round_a_pass_count = 0
    round_b_pass_count = 0
    both_pass_count = 0
    
    # 统计 Round B 成功时的尝试次数分布
    round_b_attempt_distribution = {}
    
    # 统计判断方法分布
    judge_method_distribution = {"heuristic": 0, "orm": 0}
    
    for data in dataset:
        for variant in data.get("removal_variants", []):
            total_variants += 1
            
            verification = variant.get("verification", {})
            
            # ========== 统计两轮验证结果 ==========
            round_a_passed = verification.get("round_a_passed", False)
            round_b_passed = verification.get("round_b_passed", False)
            
            if round_a_passed:
                round_a_pass_count += 1
            if round_b_passed:
                round_b_pass_count += 1
            if round_a_passed and round_b_passed:
                both_pass_count += 1
            # ====================================
            
            # 只保留有效的 pair（两轮都通过）
            if verification.get("is_valid", False):
                # ========== 统计 Round B 成功时的尝试次数 ==========
                round_b_info = verification.get("round_b", {})
                success_at_attempt = round_b_info.get("success_at_attempt")
                
                if success_at_attempt:
                    # 修复：使用正确的变量名
                    round_b_attempt_distribution[success_at_attempt] = \
                        round_b_attempt_distribution.get(success_at_attempt, 0) + 1
                    
                    # 统计判断方法（从成功的那次尝试中获取）
                    all_attempts = round_b_info.get("all_attempts", [])
                    if success_at_attempt <= len(all_attempts):
                        success_attempt_record = all_attempts[success_at_attempt - 1]
                        judge_method = success_attempt_record.get("judge_method", "orm")
                        judge_method_distribution[judge_method] = \
                            judge_method_distribution.get(judge_method, 0) + 1
                # ==================================================
                
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
                    "original_id": data["id"],
                    # ========== 保留所有提取的条件 ==========
                    "all_extracted_conditions": data.get("all_extracted_conditions", []),
                    "num_conditions_extracted": data.get("num_conditions_extracted", 0)
                    # ====================================
                }
                valid_data.append(valid_item)
                valid_variants += 1
    
    # 按 ID 排序
    valid_data.sort(key=lambda x: (x.get('original_id', 0), x.get('removed_condition_index', 0)))
    
    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)
    
    # ========== Statistics ==========
    print("\n" + "="*70)
    print("MISSING INFORMATION PROBLEM (MIP) DATASET STATISTICS")
    print("="*70)
    print(f"Original problems: {total_original}")
    print(f"\nTotal removal variants generated: {total_variants}")
    
    print(f"\n📊 Two-Round Verification Results:")
    print(f"  Round A passed (no condition → can't solve): {round_a_pass_count} ({round_a_pass_count/total_variants*100:.1f}%)")
    print(f"  Round B passed (with condition → can solve): {round_b_pass_count} ({round_b_pass_count/total_variants*100:.1f}%)")
    print(f"  Both rounds passed (VALID): {both_pass_count} ({both_pass_count/total_variants*100:.1f}%)")
    print(f"\nValid removal variants: {valid_variants}")
    
    if valid_variants > 0:
        print(f"\nRound B Success Distribution (when valid):")
        for attempt in sorted(round_b_attempt_distribution.keys()):
            count = round_b_attempt_distribution[attempt]
            print(f"  Candidate {attempt}: {count} variants ({count/valid_variants*100:.1f}%)")
        
        # 判断方法统计
        print(f"\nJudge Method Distribution (Round B success):")
        for method, count in judge_method_distribution.items():
            print(f"  {method.capitalize()}: {count} ({count/valid_variants*100:.1f}%)")
    
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
    
    # Heuristic 统计
    print(f"\n🎯 Heuristic Checks (free):")
    print(f"  Total heuristic validations: {total_heuristic_count:,}")
    
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
    print("MIP CONSTRUCTION - TWO-ROUND VERIFICATION WITH DEEPSCALER")
    print("="*70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Prompt: {args.prompt_dir}")
    print(f"Model (extract/rewrite): {args.model}")
    print(f"Model (verify): {args.verify_model}")
    print(f"Model (judge ORM fallback): {args.judge_model}")
    print(f"Use Math ORM: {'✓ Enabled' if args.use_math_orm else '✗ Disabled (heuristic only)'}")
    print(f"Temperature: {args.temperature}")
    print(f"Sampling n: {args.max_attempts}")
    print(f"Parallel threads: {args.threads}")
    print(f"Items: {len(dataset)}")
    if args.force:
        print(f"Mode: FORCE (reprocessing all)")
    print("="*70)
    
    # Step 1: Extract and Generate Variants (并行)
    print("\n[1/3] Extracting conditions and generating removal variants (parallel)")
    extract_path = os.path.join(output_dir, f"{args.dataset}_variants.json")
    process_with_jsonl_parallel(dataset, extract_path, extract_and_generate_variants, "Generating variants")
    
    # Step 2: Two-Round Verification (并行处理变体)
    print(f"\n[2/3] Two-round verification with Deepscaler (n={args.max_attempts}, parallel)")
    print(f"  Round A: WITHOUT condition (must all fail)")
    print(f"  Round B: WITH condition (at least one succeeds)")
    dataset = read_json(extract_path)
    final_path = os.path.join(output_dir, f"{args.dataset}_final.json")
    process_with_jsonl_parallel(dataset, final_path, verify_incomplete_questions_with_two_rounds, "Two-round verification")
    
    # Filter
    print("\n[3/3] Filtering valid data")
    filter_valid_data(final_path)
    
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()