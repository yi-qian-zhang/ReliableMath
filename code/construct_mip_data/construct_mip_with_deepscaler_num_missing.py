#!/usr/bin/env python3
"""
Missing Information Problem (MIP) Dataset Construction - 支持可变缺省条件数量

输入数据: 原始数学问题 (question) + 标准答案 (ground_truth) + 难度标签 (difficulty)

新架构 (4步流程):

Step 1. 提取条件 (extract_conditions_only):
   → 使用 GPT-4o 提取问题中的所有关键条件
   → 输出: extracted_conditions = [c1, c2, c3, ..., cN]
   ↓

Step 2. 生成移除变体 (generate_removal_variants):
   → 根据参数 --num_missing=n，生成所有 C(N,n) 种组合
   → 对每个组合，调用 LLM 改写问题（移除指定条件）
   → 输出: removal_variants (包含所有变体)
   ↓

Step 3. 验证 A - 条件必要性 (verify_round_a):
   → 给模型缺省问题 (incomplete_question)
   → vLLM sampling 8次，用 Deepscaler 判断等价性
   → 全都 ≠ ground_truth → 通过（条件必要）
   ↓

Step 4. 验证 B - 条件充分性 (verify_round_b):
   → 给模型缺省问题 + 被移除的条件们
   → vLLM sampling 8次，用 Deepscaler 判断等价性
   → 至少1个 = ground_truth → 通过（条件充分）
   ↓

最终数据集: 只包含两轮验证都通过的有效缺省问题
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from itertools import combinations

# ============= Deepscaler 模块 =============
from deepscaler.rewards.math_utils.utils import (
    grade_answer_mathd,
    grade_answer_sympy,
    extract_answer
)
from deepscaler.system_prompts import ORM_PROMPT
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============= 命令行参数 =============
parser = argparse.ArgumentParser(description="MIP Dataset Construction - Variable Missing Conditions")
parser.add_argument("--model", default="gpt-4o-mini", help="Model for extraction/rewrite")
parser.add_argument("--verify_model", default="deepseek-r1-distill-qwen-7b", help="Model for verification")
parser.add_argument("--judge_model", default="gpt-4o-mini", help="Model for LLM-as-Judge (ORM fallback)")
parser.add_argument("--data_dir", default="data/solve", help="Input directory")
parser.add_argument("--output_dir", default="data/construct_mip_data", help="Output directory")
parser.add_argument("--prompt_dir", default="prompt/construct_mip_with_deepscaler_num_missing", help="Prompt directory")
parser.add_argument("--dataset", default="polaris_easy_20", help="Dataset name")
parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for verification")
parser.add_argument("--max_attempts", default=8, type=int, help="Max attempts for verification")
parser.add_argument("--threads", default=8, type=int, help="Number of parallel threads")
parser.add_argument("--num_missing", default=1, type=int, help="Number of conditions to remove (n in C(N,n))")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 5 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items")
parser.add_argument("--use_math_orm", action='store_true', help="Enable LLM ORM for answer verification")
args = parser.parse_args()

# ============= Load API Config =============
try:
    api_config_path = "data/api_keys.json"
    model_options = json.load(open(api_config_path, "r"))
except FileNotFoundError:
    logging.error(f"api_keys.json not found at {api_config_path}!")
    logging.error(f"Please make sure you run this script from ~/ReliableMath directory")
    exit(1)

# 全局锁
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
    """记录 token 使用量"""
    if "gpt4o_prompt_lengths" not in data:
        data["gpt4o_prompt_lengths"] = []
        data["gpt4o_completion_lengths"] = []
    if "gpt4o_mini_prompt_lengths" not in data:
        data["gpt4o_mini_prompt_lengths"] = []
        data["gpt4o_mini_completion_lengths"] = []
    if "local_prompt_lengths" not in data:
        data["local_prompt_lengths"] = []
        data["local_completion_lengths"] = []
    if "heuristic_count" not in data:
        data["heuristic_count"] = 0

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
        data["heuristic_count"] += 1

# ============= API Functions =============

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    """调用 OpenAI-compatible API（单个响应）"""
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
                completion_tokens = count_tokens(response_text, model_name)

            return response_text, prompt_tokens, completion_tokens, model_type

        except Exception as e:
            logging.warning(f'API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                wait_time = 3 if is_local_model else 10
                time.sleep(wait_time * (attempt + 1))

    return "", 0, 0, model_type

def get_response_openai_with_sampling(input_prompt, persona="", model=None, temperature=0.0, n=1):
    """调用 OpenAI-compatible API，支持 sampling"""
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

            candidates = [choice.message.content for choice in completion.choices]

            try:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            except:
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
    """修复 LaTeX 表达式中的反斜杠"""
    try:
        # 提取 JSON 部分
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
        else:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
            else:
                return fallback if fallback is not None else {}

        # 转义反斜杠
        placeholder = "<<<DOUBLE_BACKSLASH>>>"
        json_str = json_str.replace("\\\\", placeholder)
        json_str = json_str.replace("\\", "\\\\")
        json_str = json_str.replace(placeholder, "\\\\")

        return json.loads(json_str)
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}")
    return fallback if fallback is not None else {}

# ============= Answer Processing =============

def extract_answer_from_response(response_text):
    """
    DeepSeek-R1 系列模型会生成如下格式：
    <think>...</think>
    最终答案 \\boxed{正确答案}
    """
    if "</think>" not in response_text:
        return None

    response_text = response_text.split("</think>", 1)[1].strip()
    return extract_answer(response_text)

def judge_answer_equivalence(question, model_answer, ground_truth):
    """使用 Deepscaler 的多层验证逻辑判断答案等价性"""
    # 第一层：启发式方法
    is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)

    if is_correct:
        logging.debug(f"✓ Heuristic match: {model_answer} ≈ {ground_truth}")
        return True, 0, 0, "heuristic"

    # 第二层：LLM ORM
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

    logging.debug(f"✗ No match: {model_answer} ≠ {ground_truth}")
    return False, 0, 0, "heuristic"

# ============= Multiple Choice Handling =============

def extract_multiple_choice_options(question):
    """从问题中提取选择题选项部分"""
    if not question:
        return None

    match_a = re.search(r'\(A\)', question)
    if not match_a:
        return None

    options_start = match_a.start()
    options_text = question[options_start:].strip()

    option_count = len(re.findall(r'\([A-E]\)', options_text))
    if option_count >= 2:
        return options_text

    return None

def ensure_options_in_question(incomplete_question, original_question):
    """确保改写后的问题包含原始选项（如果是选择题）"""
    original_options = extract_multiple_choice_options(original_question)

    if not original_options:
        return incomplete_question

    has_options = extract_multiple_choice_options(incomplete_question)
    if has_options:
        return incomplete_question

    logging.debug(f"Multiple-choice options missing, restoring...")

    enter_patterns = [
        r'Enter the letters?.*',
        r'Enter the correct options?.*',
        r'separated by commas.*'
    ]

    enter_match = None
    for pattern in enter_patterns:
        enter_match = re.search(pattern, incomplete_question, re.IGNORECASE | re.DOTALL)
        if enter_match:
            break

    if enter_match:
        before_enter = incomplete_question[:enter_match.start()].strip()
        enter_text = incomplete_question[enter_match.start():].strip()

        pure_options = original_options
        for pattern in enter_patterns:
            pure_options = re.sub(pattern, '', pure_options, flags=re.IGNORECASE | re.DOTALL).strip()

        return f"{before_enter}\n\n{pure_options}\n\n{enter_text}"
    else:
        return f"{incomplete_question}\n\n{original_options}"

# ============= Step 1: Extract Conditions Only =============

def extract_conditions_only(data):
    """
    Step 1: 只提取条件，不做移除和改写

    输出:
      data["extracted_conditions"] = [c1, c2, c3, ...]
      data["num_conditions"] = N
    """
    prompt_path = os.path.join(args.prompt_dir, "extract_conditions.txt")

    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["extracted_conditions"] = []
        data["num_conditions"] = 0
        return data

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    input_prompt = prompt_template.format(
        original_question=data["question"]
    )

    response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        input_prompt,
        persona="You are an expert at analyzing mathematical problems.",
        model=args.model,
        temperature=0.0
    )

    # 记录 token
    record_tokens(data, model_type, prompt_tokens, completion_tokens)

    # 解析响应
    conditions = parse_json_response(response, fallback=[])

    # 确保是列表
    if not isinstance(conditions, list):
        logging.warning(f"ID {data['id']}: Expected list, got {type(conditions)}")
        conditions = []

    # 清理条件文本
    cleaned_conditions = []
    for cond in conditions:
        if isinstance(cond, str):
            cond = cond.strip()
            # 移除常见前缀
            for prefix in ["Condition:", "条件:", "-", "•"]:
                if cond.startswith(prefix):
                    cond = cond[len(prefix):].strip()
            if cond:
                cleaned_conditions.append(cond)

    data["extracted_conditions"] = cleaned_conditions
    data["num_conditions"] = len(cleaned_conditions)

    # 检测是否为选择题
    is_multiple_choice = extract_multiple_choice_options(data["question"]) is not None
    data["is_multiple_choice"] = is_multiple_choice

    logging.info(f"ID {data['id']}: Extracted {len(cleaned_conditions)} conditions" +
                 (" (multiple-choice)" if is_multiple_choice else ""))

    return data

# ============= Step 2: Generate Removal Variants =============

def generate_removal_variants(data, num_missing):
    """
    Step 2: 根据 num_missing 生成所有 C(N, num_missing) 种移除组合

    参数:
      num_missing: 要移除的条件数量

    输出:
      data["removal_variants"] = [
        {
          "variant_id": "xxx_remove_0",
          "removed_conditions": [c1, c2],
          "remaining_conditions": [c3],
          "incomplete_question": "..."
        },
        ...
      ]
    """
    conditions = data.get("extracted_conditions", [])
    N = len(conditions)

    if N == 0:
        logging.warning(f"ID {data['id']}: No conditions extracted, skipping")
        data["removal_variants"] = []
        return data

    if num_missing > N:
        logging.warning(f"ID {data['id']}: num_missing={num_missing} > N={N}, skipping")
        data["removal_variants"] = []
        return data

    if num_missing <= 0:
        logging.warning(f"ID {data['id']}: num_missing={num_missing} <= 0, skipping")
        data["removal_variants"] = []
        return data

    # 生成所有 C(N, num_missing) 种组合
    removal_combos = list(combinations(range(N), num_missing))

    logging.info(f"ID {data['id']}: Generating C({N},{num_missing}) = {len(removal_combos)} variants")

    # 读取改写 prompt
    prompt_path = os.path.join(args.prompt_dir, "rewrite_without_conditions.txt")

    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["removal_variants"] = []
        return data

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    variants = []

    for combo_idx, combo_indices in enumerate(removal_combos):
        # 确定移除和保留的条件
        removed_conditions = [conditions[i] for i in combo_indices]
        remaining_conditions = [conditions[i] for i in range(N) if i not in combo_indices]

        # 格式化 prompt
        all_conditions_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(conditions))
        removed_conditions_text = "\n".join(f"- {c}" for c in removed_conditions)
        remaining_conditions_text = "\n".join(f"- {c}" for c in remaining_conditions) if remaining_conditions else "(None - all conditions removed)"

        input_prompt = prompt_template.format(
            original_question=data["question"],
            all_conditions=all_conditions_text,
            removed_conditions=removed_conditions_text,
            remaining_conditions=remaining_conditions_text
        )

        # 调用 LLM 改写
        response, prompt_tokens, completion_tokens, model_type = get_response_openai(
            input_prompt,
            persona="You are an expert at rewriting mathematical problems.",
            model=args.model,
            temperature=0.0
        )

        # 记录 token
        record_tokens(data, model_type, prompt_tokens, completion_tokens)

        # 清理响应
        incomplete_question = response.strip()

        # 移除常见前缀/后缀
        for prefix in ["Rewritten Question:", "Rewritten Problem:", "Answer:", "###", "**", '"', "'"]:
            incomplete_question = incomplete_question.replace(prefix, "").strip()

        # 确保选择题选项存在
        if data.get("is_multiple_choice"):
            original_incomplete = incomplete_question
            incomplete_question = ensure_options_in_question(
                incomplete_question,
                data["question"]
            )

            if original_incomplete != incomplete_question:
                logging.debug(f"ID {data['id']}_remove_{combo_idx}: ✓ Restored multiple-choice options")

        variant = {
            "variant_id": f"{data['id']}_remove_{combo_idx}",
            "removed_conditions": removed_conditions,
            "remaining_conditions": remaining_conditions,
            "incomplete_question": incomplete_question,
            "num_missing": num_missing
        }

        variants.append(variant)

    data["removal_variants"] = variants

    logging.info(f"ID {data['id']}: Generated {len(variants)} removal variants")

    return data

# ============= Step 3-4: Two-Round Verification =============

def verify_single_variant(data, variant, prompt_template_incomplete, prompt_template_complete, ground_truth):
    """验证单个变体（两轮验证）"""
    incomplete_question = variant["incomplete_question"]
    removed_conditions = variant["removed_conditions"]

    # ========== 验证 A：缺省条件下问题不可解 ==========
    logging.info(f"ID {variant['variant_id']}: Starting Round A - Testing incomplete question...")

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
        model_answer = extract_answer_from_response(candidate_text)

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

        attempt_record = {
            "attempt": attempt_num,
            "full_response": candidate_text,
            "model_answer": model_answer if model_answer else "N/A",
            "judge_result": judge_result,
            "judge_method": judge_method,
            "is_correct": is_correct
        }
        round_a_attempts.append(attempt_record)

        if is_correct:
            round_a_has_correct = True

    # Round A 结果判定
    round_a_passed = not round_a_has_correct

    if round_a_passed:
        logging.info(f"ID {variant['variant_id']}: ✓ Round A PASSED - All {args.max_attempts} answers ≠ ground_truth")
    else:
        logging.info(f"ID {variant['variant_id']}: ✗ Round A FAILED - At least 1 answer = ground_truth")
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
    logging.info(f"ID {variant['variant_id']}: Starting Round B - Testing WITH removed conditions...")

    # 格式化移除的条件（支持多个）
    removed_conditions_text = "\n".join(f"- {c}" for c in removed_conditions)

    input_prompt_complete = prompt_template_complete.format(
        incomplete_question=incomplete_question,
        removed_conditions=removed_conditions_text
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
        model_answer = extract_answer_from_response(candidate_text)

        if model_answer is None:
            is_correct = False
            judge_result = "no_answer_tag"
            judge_method = "none"
        else:
            is_correct, judge_prompt_tokens, judge_completion_tokens, judge_model_type = judge_answer_equivalence(
                incomplete_question + " [With conditions: " + ", ".join(removed_conditions) + "]",
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

        attempt_record = {
            "attempt": attempt_num,
            "full_response": candidate_text,
            "model_answer": model_answer if model_answer else "N/A",
            "judge_result": judge_result,
            "judge_method": judge_method,
            "is_correct": is_correct
        }
        round_b_attempts.append(attempt_record)

        if is_correct and not round_b_has_correct:
            round_b_has_correct = True
            success_at_attempt = attempt_num

    # Round B 结果判定
    round_b_passed = round_b_has_correct

    if round_b_passed:
        logging.info(f"ID {variant['variant_id']}: ✓ Round B PASSED - Answer {success_at_attempt}/{args.max_attempts} = ground_truth")
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
    """Step 3-4: 两轮验证（串行处理变体）"""
    # 读取验证 prompt（这两个 prompt 保持不变）
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

    # 串行处理所有变体
    for variant in variants:
        try:
            verified_variant = verify_single_variant(
                data, variant,
                prompt_template_incomplete,
                prompt_template_complete,
                ground_truth
            )

            # 更新 data 中对应的 variant
            variant_id = verified_variant["variant_id"]
            for i, v in enumerate(data["removal_variants"]):
                if v["variant_id"] == variant_id:
                    data["removal_variants"][i] = verified_variant
                    break

        except Exception as e:
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
        future_to_data = {executor.submit(process_func, data): data for data in dataset}

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

def filter_valid_data(final_path, num_missing):
    """筛选有效的缺省问题"""
    dataset = read_json(final_path)
    valid_data = []

    # 统计 token
    total_gpt4o_prompt = sum(sum(d.get("gpt4o_prompt_lengths", [])) for d in dataset)
    total_gpt4o_completion = sum(sum(d.get("gpt4o_completion_lengths", [])) for d in dataset)

    total_gpt4o_mini_prompt = sum(sum(d.get("gpt4o_mini_prompt_lengths", [])) for d in dataset)
    total_gpt4o_mini_completion = sum(sum(d.get("gpt4o_mini_completion_lengths", [])) for d in dataset)

    total_local_prompt = sum(sum(d.get("local_prompt_lengths", [])) for d in dataset)
    total_local_completion = sum(sum(d.get("local_completion_lengths", [])) for d in dataset)

    total_heuristic_count = sum(d.get("heuristic_count", 0) for d in dataset)

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

            round_a_passed = verification.get("round_a_passed", False)
            round_b_passed = verification.get("round_b_passed", False)

            if round_a_passed:
                round_a_pass_count += 1
            if round_b_passed:
                round_b_pass_count += 1
            if round_a_passed and round_b_passed:
                both_pass_count += 1

            # 只保留有效的 pair
            if verification.get("is_valid", False):
                round_b_info = verification.get("round_b", {})
                success_at_attempt = round_b_info.get("success_at_attempt")

                if success_at_attempt:
                    round_b_attempt_distribution[success_at_attempt] = \
                        round_b_attempt_distribution.get(success_at_attempt, 0) + 1

                    all_attempts = round_b_info.get("all_attempts", [])
                    if success_at_attempt <= len(all_attempts):
                        success_attempt_record = all_attempts[success_at_attempt - 1]
                        judge_method = success_attempt_record.get("judge_method", "orm")
                        judge_method_distribution[judge_method] = \
                            judge_method_distribution.get(judge_method, 0) + 1

                valid_item = {
                    "id": variant["variant_id"],
                    "data_source": data.get("data_source", ""),
                    "difficulty": data.get("difficulty", ""),
                    "transformation_type": "condition_removal",
                    "num_missing": num_missing,
                    "original_question": data["question"],
                    "ground_truth": data.get("ground_truth", ""),
                    "removed_conditions": variant["removed_conditions"],
                    "remaining_conditions": variant["remaining_conditions"],
                    "incomplete_question": variant["incomplete_question"],
                    "verification": verification,
                    "original_id": data["id"],
                    "all_extracted_conditions": data.get("extracted_conditions", []),
                    "num_conditions_extracted": data.get("num_conditions", 0)
                }
                valid_data.append(valid_item)
                valid_variants += 1

    # 按 ID 排序
    valid_data.sort(key=lambda x: x.get('original_id', 0))

    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)

    # ========== Statistics ==========
    print("\n" + "="*70)
    print("MISSING INFORMATION PROBLEM (MIP) DATASET STATISTICS")
    print("="*70)
    print(f"Configuration: num_missing = {num_missing}")
    print(f"Original problems: {total_original}")
    print(f"\nTotal removal variants generated: {total_variants}")

    print(f"\n📊 Two-Round Verification Results:")
    print(f"  Round A passed (without conditions → can't solve): {round_a_pass_count} ({round_a_pass_count/total_variants*100:.1f}%)")
    print(f"  Round B passed (with conditions → can solve): {round_b_pass_count} ({round_b_pass_count/total_variants*100:.1f}%)")
    print(f"  Both rounds passed (VALID): {both_pass_count} ({both_pass_count/total_variants*100:.1f}%)")
    print(f"\nValid removal variants: {valid_variants}")

    if valid_variants > 0:
        print(f"\nRound B Success Distribution (when valid):")
        for attempt in sorted(round_b_attempt_distribution.keys()):
            count = round_b_attempt_distribution[attempt]
            print(f"  Candidate {attempt}: {count} variants ({count/valid_variants*100:.1f}%)")

        print(f"\nJudge Method Distribution (Round B success):")
        for method, count in judge_method_distribution.items():
            print(f"  {method.capitalize()}: {count} ({count/valid_variants*100:.1f}%)")

    # 成本统计
    gpt4o_prompt_rate = 2.5
    gpt4o_completion_rate = 10.0
    gpt4o_mini_prompt_rate = 0.15
    gpt4o_mini_completion_rate = 0.6

    print(f"\n💰 GPT-4o Token Usage:")
    print(f"  Prompt: {total_gpt4o_prompt:,}")
    print(f"  Completion: {total_gpt4o_completion:,}")
    print(
        f"  Cost = {total_gpt4o_prompt}/1e6*{gpt4o_prompt_rate} "
        f"+ {total_gpt4o_completion}/1e6*{gpt4o_completion_rate} "
        f"= ${total_gpt4o_prompt/1e6*gpt4o_prompt_rate + total_gpt4o_completion/1e6*gpt4o_completion_rate:.6f}"
    )

    print(f"\n💰 GPT-4o-mini Token Usage:")
    print(f"  Prompt: {total_gpt4o_mini_prompt:,}")
    print(f"  Completion: {total_gpt4o_mini_completion:,}")
    print(
        f"  Cost = {total_gpt4o_mini_prompt}/1e6*{gpt4o_mini_prompt_rate} "
        f"+ {total_gpt4o_mini_completion}/1e6*{gpt4o_mini_completion_rate} "
        f"= ${total_gpt4o_mini_prompt/1e6*gpt4o_mini_prompt_rate + total_gpt4o_mini_completion/1e6*gpt4o_mini_completion_rate:.6f}"
    )

    print(f"\n🖥️  Local Model Token Usage:")
    print(f"  Prompt: {total_local_prompt:,}")
    print(f"  Completion: {total_local_completion:,}")

    print(f"\n🎯 Heuristic Checks (free):")
    print(f"  Total heuristic validations: {total_heuristic_count:,}")

    print(f"\nOutput: {output_path}")
    print("="*70)

# ============= Main Workflow =============

def construction_workflow():
    """主流程：4步构建 MIP 数据集"""
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
    print("MIP CONSTRUCTION - VARIABLE MISSING CONDITIONS")
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
    print(f"Num missing conditions: {args.num_missing}")
    print(f"Items: {len(dataset)}")
    if args.force:
        print(f"Mode: FORCE (reprocessing all)")
    print("="*70)

    # ========== Step 1: Extract Conditions Only ==========
    extract_path = os.path.join(output_dir, f"{args.dataset}_conditions.json")

    if os.path.exists(extract_path) and not args.force:
        existing_conditions = read_json(extract_path)
        if len(existing_conditions) == len(dataset):
            print(f"\n[1/4] ✓ Conditions already extracted ({len(existing_conditions)} items), skipping...")
            dataset = existing_conditions
        else:
            print(f"\n[1/4] Extracting conditions (continuing from {len(existing_conditions)}/{len(dataset)})")
            process_with_jsonl_parallel(dataset, extract_path, extract_conditions_only, "Extracting conditions")
            dataset = read_json(extract_path)
    else:
        print("\n[1/4] Extracting conditions (parallel)")
        process_with_jsonl_parallel(dataset, extract_path, extract_conditions_only, "Extracting conditions")
        dataset = read_json(extract_path)

    # ========== Step 2: Generate Removal Variants ==========
    variants_path = os.path.join(output_dir, f"{args.dataset}_variants_n{args.num_missing}.json")

    # 使用 lambda 包装以传递 num_missing 参数
    generate_func = lambda data: generate_removal_variants(data, args.num_missing)

    if os.path.exists(variants_path) and not args.force:
        existing_variants = read_json(variants_path)
        if len(existing_variants) == len(dataset):
            print(f"\n[2/4] ✓ Variants already generated ({len(existing_variants)} items), skipping...")
            dataset = existing_variants
        else:
            print(f"\n[2/4] Generating removal variants (n={args.num_missing}, continuing from {len(existing_variants)}/{len(dataset)})")
            process_with_jsonl_parallel(dataset, variants_path, generate_func, f"Generating variants (n={args.num_missing})")
            dataset = read_json(variants_path)
    else:
        print(f"\n[2/4] Generating removal variants (n={args.num_missing}, parallel)")
        process_with_jsonl_parallel(dataset, variants_path, generate_func, f"Generating variants (n={args.num_missing})")
        dataset = read_json(variants_path)

    # ========== Step 3-4: Two-Round Verification ==========
    final_path = os.path.join(output_dir, f"{args.dataset}_final_n{args.num_missing}.json")

    if os.path.exists(final_path) and not args.force:
        existing_final = read_json(final_path)
        if len(existing_final) == len(dataset):
            print(f"\n[3/4] ✓ Verification already complete ({len(existing_final)} items), skipping...")
        else:
            print(f"\n[3/4] Two-round verification (n={args.max_attempts}, continuing from {len(existing_final)}/{len(dataset)})")
            print(f"  Round A: WITHOUT conditions (must all fail)")
            print(f"  Round B: WITH conditions (at least one succeeds)")
            process_with_jsonl_parallel(dataset, final_path, verify_incomplete_questions_with_two_rounds, "Two-round verification")
    else:
        print(f"\n[3/4] Two-round verification (n={args.max_attempts}, parallel)")
        print(f"  Round A: WITHOUT conditions (must all fail)")
        print(f"  Round B: WITH conditions (at least one succeeds)")
        process_with_jsonl_parallel(dataset, final_path, verify_incomplete_questions_with_two_rounds, "Two-round verification")

    # ========== Step 4: Filter Valid Data ==========
    print("\n[4/4] Filtering valid data")
    filter_valid_data(final_path, args.num_missing)

    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()
