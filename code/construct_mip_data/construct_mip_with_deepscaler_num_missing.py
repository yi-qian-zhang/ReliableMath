#!/usr/bin/env python3
"""
Missing Information Problem (MIP) Dataset Construction - 支持可变缺省条件数量
输入数据: 原始数学问题 (question) + 标准答案 (ground_truth) + 难度标签 (difficulty)

新架构 (5步流程):
Step 1. 提取条件 (extract_conditions_only): 使用 GPT-4o 提取问题中的所有关键条件
Step 1.5. 过滤样本 (filter_by_num_conditions): 只保留 num_conditions >= num_missing + 1 的样本
Step 2. 生成移除变体 (generate_removal_variants): 根据参数 --num_missing=n，生成所有 C(N,n) 种组合
Step 3. 验证 A - 改写质量检查: LLM 快速验证改写正确性和问题有效性
Step 4. 验证 B - 条件必要性: 给模型缺省问题，vLLM sampling 8次，全都 ≠ ground_truth → 通过
Step 5. 验证 C - 条件充分性: 给模型缺省问题 + 被移除的条件们，至少1个 = ground_truth → 通过
最终数据集: 只包含三轮验证都通过的有效缺省问题
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
from deepscaler.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy, extract_answer
from deepscaler.system_prompts import ORM_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_format(template, **kwargs):
    """
    Safely format a template string, escaping curly braces in both template and arguments.

    This prevents KeyError when mathematical notation like {m/s} or {x, y} appears
    in questions or arguments.

    Args:
        template: Template string with placeholders like {placeholder_name}
        **kwargs: Keyword arguments to fill into the template

    Returns:
        Formatted string with all curly braces properly escaped
    """
    # Step 1: Escape ALL curly braces in the template
    escaped_template = template.replace('{', '{{').replace('}', '}}')

    # Step 2: Restore ONLY the valid placeholders (kwargs keys)
    for key in kwargs.keys():
        # Replace {{key}} back to {key} for actual placeholders
        escaped_template = escaped_template.replace('{{' + key + '}}', '{' + key + '}')

    # Step 3: Escape curly braces in all argument values
    escaped_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            # Escape curly braces in string arguments
            escaped_kwargs[key] = value.replace('{', '{{').replace('}', '}}')
        else:
            escaped_kwargs[key] = value

    # Step 4: Format with escaped template and escaped arguments
    return escaped_template.format(**escaped_kwargs)

parser = argparse.ArgumentParser(description="MIP Dataset Construction - Variable Missing Conditions")
parser.add_argument("--extract_model", default="gpt-4o-mini", help="Model for condition extraction")
parser.add_argument("--rewrite_model", default="DeepSeek-R1-Distill-Qwen-32B-8715", help="Model for question rewriting (defaults to --extract_model if not specified)")
parser.add_argument("--verify_model", default="DeepSeek-R1-Distill-Qwen-32B-8715", help="Model for verification")
parser.add_argument("--judge_model", default="gpt-4o-mini", help="Model for LLM-as-Judge (ORM fallback)")
parser.add_argument("--data_dir", default="data/solve", help="Input directory")
parser.add_argument("--output_dir", default="data/construct_mip_data", help="Output directory")
parser.add_argument("--prompt_dir", default="prompt/construct_mip_with_deepscaler_num_missing", help="Prompt directory")
parser.add_argument("--dataset", default="polaris_20", help="Dataset name")
parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for verification")
parser.add_argument("--max_attempts", default=8, type=int, help="Max attempts for verification")
parser.add_argument("--threads", default=8, type=int, help="Number of parallel threads")
parser.add_argument("--num_missing", default=1, type=int, help="Number of conditions to remove (n in C(N,n))")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 5 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items")
parser.add_argument("--use_math_orm", action='store_true', help="Enable LLM ORM for answer verification")
args = parser.parse_args()

# 🔧 如果未指定 rewrite_model，默认使用 extract_model
if args.rewrite_model is None:
    args.rewrite_model = args.extract_model

try:
    api_config_path = "data/api_keys.json"
    model_options = json.load(open(api_config_path, "r"))
except FileNotFoundError:
    logging.error(f"api_keys.json not found at {api_config_path}!")
    logging.error(f"Please make sure you run this script from ~/ReliableMath directory")
    exit(1)

jsonl_write_lock = threading.Lock()

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

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    if model is None:
        model = args.extract_model
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
                model=model_name, messages=messages, temperature=temperature,
                max_tokens=4096, stream=False
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
    if model is None:
        model = args.extract_model
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
                model=model_name, messages=messages, temperature=temperature,
                n=n, max_tokens=4096, stream=False
            )
            candidates = [choice.message.content for choice in completion.choices]
            try:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            except:
                completion_tokens = sum(count_tokens(text, model_name) for text in candidates)
            return {
                "candidates": candidates, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "model_type": model_type
            }
        except Exception as e:
            logging.warning(f'API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                wait_time = 3 if is_local_model else 10
                time.sleep(wait_time * (attempt + 1))
    return None

def parse_json_response(response, fallback=None):
    try:
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
        placeholder = "<<<DOUBLE_BACKSLASH>>>"
        json_str = json_str.replace("\\\\", placeholder)
        json_str = json_str.replace("\\", "\\\\")
        json_str = json_str.replace(placeholder, "\\\\")
        return json.loads(json_str)
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}")
    return fallback if fallback is not None else {}

def extract_answer_from_response(response_text):
    if "</think>" not in response_text:
        return None
    response_text = response_text.split("</think>", 1)[1].strip()
    return extract_answer(response_text)

def judge_answer_equivalence(question, model_answer, ground_truth):
    is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
    if is_correct:
        logging.debug(f"✓ Heuristic match: {model_answer} ≈ {ground_truth}")
        return True, 0, 0, "heuristic"
    if args.use_math_orm:
        logging.debug(f"Heuristic failed, trying ORM: {model_answer} vs {ground_truth}")
        ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""
        input_prompt = safe_format(
            ORM_USER_TEMPLATE,
            problem=question, answer_1=model_answer, answer_2=ground_truth
        )
        try:
            response, prompt_tokens, completion_tokens, model_type = get_response_openai(
                input_prompt, persona=ORM_PROMPT, model=args.judge_model, temperature=0.0
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

def extract_multiple_choice_options(question):
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
    original_options = extract_multiple_choice_options(original_question)
    if not original_options:
        return incomplete_question
    has_options = extract_multiple_choice_options(incomplete_question)
    if has_options:
        return incomplete_question
    logging.debug(f"Multiple-choice options missing, restoring...")
    enter_patterns = [
        r'Enter the letters?.*', r'Enter the correct options?.*', r'separated by commas.*'
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

def extract_conditions_only(data):
    prompt_path = os.path.join(args.prompt_dir, "extract_conditions.txt")
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["extracted_conditions"] = []
        data["num_conditions"] = 0
        return data
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    input_prompt = safe_format(prompt_template, original_question=data["original_question"])
    response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        input_prompt, persona="You are an expert at analyzing mathematical problems.",
        model=args.extract_model, temperature=0.0
    )
    record_tokens(data, model_type, prompt_tokens, completion_tokens)
    conditions = parse_json_response(response, fallback=[])
    if not isinstance(conditions, list):
        logging.warning(f"ID {data['id']}: Expected list, got {type(conditions)}")
        conditions = []
    cleaned_conditions = []
    for cond in conditions:
        if isinstance(cond, str):
            cond = cond.strip()
            for prefix in ["Condition:", "条件:", "-", "•"]:
                if cond.startswith(prefix):
                    cond = cond[len(prefix):].strip()
            if cond:
                cleaned_conditions.append(cond)
    data["extracted_conditions"] = cleaned_conditions
    data["num_conditions"] = len(cleaned_conditions)
    is_multiple_choice = extract_multiple_choice_options(data["original_question"]) is not None
    data["is_multiple_choice"] = is_multiple_choice
    logging.info(f"ID {data['id']}: Extracted {len(cleaned_conditions)} conditions" +
                 (" (multiple-choice)" if is_multiple_choice else ""))
    return data

def filter_by_num_conditions(conditions_path, num_missing):
    """
    在 extract 之后，根据 num_conditions 过滤样本
    规则: 只保留 num_conditions >= num_missing + 1 的样本
    
    Args:
        conditions_path: _conditions.json 文件路径
        num_missing: 要移除的条件数
    
    Returns:
        filtered_path: 过滤后的文件路径
        remaining_count: 剩余样本数量
    """
    dataset = read_json(conditions_path)
    
    min_conditions_required = num_missing + 1
    
    original_count = len(dataset)
    filtered_dataset = []
    filtered_out_ids = []
    
    for data in dataset:
        num_conditions = data.get("num_conditions", 0)
        
        if num_conditions >= min_conditions_required:
            filtered_dataset.append(data)
        else:
            filtered_out_ids.append(data.get('id', 'unknown'))
            logging.debug(f"ID {data['id']}: Filtered out (num_conditions={num_conditions} < {min_conditions_required})")
    
    # 生成新的文件名，包含 num_missing 信息
    filtered_path = conditions_path.replace(
        "_conditions.json",
        f"_conditions_filtered_n{num_missing}.json"
    )
    
    write_json(filtered_path, filtered_dataset)
    
    print(f"\n📊 Filtering by num_conditions >= {min_conditions_required}:")
    print(f"  Original samples: {original_count}")
    print(f"  Filtered out: {len(filtered_out_ids)} ({len(filtered_out_ids)/original_count*100:.1f}%)")
    print(f"  Remaining: {len(filtered_dataset)} ({len(filtered_dataset)/original_count*100:.1f}%)")
    if len(filtered_out_ids) > 0 and len(filtered_out_ids) <= 10:
        print(f"  Filtered out IDs: {filtered_out_ids}")
    print(f"  Output: {filtered_path}")
    
    return filtered_path, len(filtered_dataset)

def verify_rewrite_with_llm(data, rewritten_question, removed_conditions, remaining_conditions, combo_idx):
    """
    Round A: LLM Pre-verification - 改写质量检查
    Verify 1: 改写是否只改了指定条件
    Verify 2: 问题是否有效（没有删除question stem，不是无穷多解）
    """
    original_question = data["original_question"]
    removed_conditions_text = "\n".join(f"- {c}" for c in removed_conditions)
    remaining_conditions_text = "\n".join(f"- {c}" for c in remaining_conditions) if remaining_conditions else "(None)"

    # Verification 1: 改写正确性
    correctness_prompt_path = os.path.join(args.prompt_dir, "verify_rewrite_correctness.txt")
    if not os.path.exists(correctness_prompt_path):
        logging.warning(f"Correctness prompt not found: {correctness_prompt_path}")
        return {"correctness_passed": None, "validity_passed": None}

    with open(correctness_prompt_path, 'r', encoding='utf-8') as f:
        correctness_template = f.read()

    correctness_prompt = safe_format(
        correctness_template,
        original_question=original_question,
        rewritten_question=rewritten_question,
        removed_conditions=removed_conditions_text,
        remaining_conditions=remaining_conditions_text
    )

    correctness_response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        correctness_prompt,
        persona="You are an expert verifier.",
        model=args.judge_model,
        temperature=0.0
    )
    record_tokens(data, model_type, prompt_tokens, completion_tokens)

    # 解析判断结果 - 更严格的判断
    correctness_analysis = correctness_response.strip()

    # 提取 ### Judgement ### 部分
    if "### Judgement ###" in correctness_response:
        judgement_part = correctness_response.split("### Judgement ###")[1].strip()
        # 只有明确是 "True" 才通过
        correctness_passed = judgement_part.lower().startswith("true")
    else:
        # Fallback to simple check
        correctness_passed = "True" in correctness_response or "true" in correctness_response.lower()

    # Verification 2: 问题有效性
    validity_prompt_path = os.path.join(args.prompt_dir, "verify_problem_validity.txt")
    if not os.path.exists(validity_prompt_path):
        logging.warning(f"Validity prompt not found: {validity_prompt_path}")
        return {
            "correctness_passed": correctness_passed,
            "correctness_analysis": correctness_analysis,
            "validity_passed": None
        }

    with open(validity_prompt_path, 'r', encoding='utf-8') as f:
        validity_template = f.read()

    validity_prompt = safe_format(
        validity_template,
        original_question=original_question,
        rewritten_question=rewritten_question,
        removed_conditions=removed_conditions_text
    )

    validity_response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        validity_prompt,
        persona="You are an expert verifier.",
        model=args.judge_model,
        temperature=0.0
    )
    record_tokens(data, model_type, prompt_tokens, completion_tokens)

    # 解析判断结果 - 更严格的判断
    validity_analysis = validity_response.strip()

    # 提取 ### Judgement ### 部分
    if "### Judgement ###" in validity_response:
        judgement_part = validity_response.split("### Judgement ###")[1].strip()
        # 只有明确包含 "Valid" 且不包含 "Invalid" 才通过
        validity_passed = judgement_part.lower().startswith("valid") and "invalid" not in judgement_part.lower()
    else:
        # Fallback to simple check
        validity_passed = "Valid" in validity_response and "Invalid" not in validity_response

    # 综合结果
    overall_passed = correctness_passed and validity_passed

    if overall_passed:
        logging.info(f"ID {data['id']}_remove_{combo_idx}: ✓ LLM verification PASSED")
    else:
        reason = []
        if not correctness_passed:
            reason.append("改写不正确")
        if not validity_passed:
            reason.append("问题无效")
        logging.warning(f"ID {data['id']}_remove_{combo_idx}: ✗ LLM verification FAILED ({', '.join(reason)})")

    return {
        "overall_passed": overall_passed,
        "correctness_passed": correctness_passed,
        "correctness_analysis": correctness_analysis,
        "validity_passed": validity_passed,
        "validity_analysis": validity_analysis
    }

def generate_removal_variants(data, num_missing):
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
    removal_combos = list(combinations(range(N), num_missing))
    logging.info(f"ID {data['id']}: Generating C({N},{num_missing}) = {len(removal_combos)} variants")
    prompt_path = os.path.join(args.prompt_dir, "rewrite_without_conditions.txt")
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["removal_variants"] = []
        return data
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    variants = []
    for combo_idx, combo_indices in enumerate(removal_combos):
        removed_conditions = [conditions[i] for i in combo_indices]
        remaining_conditions = [conditions[i] for i in range(N) if i not in combo_indices]
        all_conditions_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(conditions))
        removed_conditions_text = "\n".join(f"- {c}" for c in removed_conditions)
        remaining_conditions_text = "\n".join(f"- {c}" for c in remaining_conditions) if remaining_conditions else "(None - all conditions removed)"
        input_prompt = safe_format(
            prompt_template,
            original_question=data["original_question"], all_conditions=all_conditions_text,
            removed_conditions=removed_conditions_text, remaining_conditions=remaining_conditions_text
        )
        response, prompt_tokens, completion_tokens, model_type = get_response_openai(
            input_prompt, persona="You are an expert at rewriting mathematical problems.",
            model=args.rewrite_model, temperature=0.0
        )
        record_tokens(data, model_type, prompt_tokens, completion_tokens)
        response_text = response.strip()

        # 🔧 移除 <think> 标签内容（deepseek-r1 等模型会输出思考过程）
        # 移除 <think>...</think> 之间的所有内容（包括标签本身）
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        # 🔧 解析 Analysis 和 Rewritten Mathematical Question
        analysis = ""
        incomplete_question = ""

        if "### Analysis ###" in response_text and "### Rewritten Mathematical Question ###" in response_text:
            # 按标记分割
            parts = response_text.split("### Rewritten Mathematical Question ###")
            if len(parts) == 2:
                # 提取 analysis 部分
                analysis_part = parts[0].split("### Analysis ###")
                if len(analysis_part) == 2:
                    analysis = analysis_part[1].strip()
                # 提取问题部分
                incomplete_question = parts[1].strip()
        else:
            # Fallback: 如果没有标记，使用整个 response
            logging.warning(f"ID {data['id']}_remove_{combo_idx}: No Analysis/Question markers found, using raw response")
            incomplete_question = response_text

        # 清理常见前缀
        for prefix in ["Rewritten Question:", "Rewritten Problem:", "Answer:", "**", '"', "'"]:
            incomplete_question = incomplete_question.replace(prefix, "").strip()
            analysis = analysis.replace(prefix, "").strip()

        if data.get("is_multiple_choice"):
            original_incomplete = incomplete_question
            incomplete_question = ensure_options_in_question(incomplete_question, data["original_question"])
            if original_incomplete != incomplete_question:
                logging.debug(f"ID {data['id']}_remove_{combo_idx}: ✓ Restored multiple-choice options")

        # 🔧 Round A: LLM Pre-verification (必须执行)
        llm_verification = verify_rewrite_with_llm(
            data, incomplete_question, removed_conditions,
            remaining_conditions, combo_idx
        )

        variant = {
            "variant_id": f"{data['id']}_remove_{combo_idx}",
            "removed_conditions": removed_conditions,
            "remaining_conditions": remaining_conditions,
            "incomplete_question": incomplete_question,
            "analysis": analysis,
            "llm_verification": llm_verification,  # 🔧 新增字段
            "num_missing": num_missing
        }
        variants.append(variant)
    data["removal_variants"] = variants
    logging.info(f"ID {data['id']}: Generated {len(variants)} removal variants")
    return data

def verify_single_variant(data, variant, prompt_template_incomplete, prompt_template_complete, ground_truth):
    incomplete_question = variant["incomplete_question"]
    removed_conditions = variant["removed_conditions"]

    # 🔧 获取 Round A (LLM Pre-verification) 的结果
    llm_verification = variant.get("llm_verification")
    round_A_passed = llm_verification.get("overall_passed", None) if llm_verification else None
    round_A_info = llm_verification if llm_verification else {}

    # 🔧 Early exit: 如果 Round A 失败，直接跳过 Round B 和 C
    if round_A_passed is False:
        logging.info(f"ID {variant['variant_id']}: ✗ Round A FAILED - Skipping Round B and C")
        variant["verification"] = {
            "round_A_passed": False,
            "round_B_passed": False,
            "round_C_passed": False,
            "is_valid": False,
            "round_A": round_A_info,
            "round_B": {"total_attempts": 0, "all_attempts": []},
            "round_C": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant

    logging.info(f"ID {variant['variant_id']}: Starting Round B - Testing incomplete question...")
    input_prompt_incomplete = safe_format(prompt_template_incomplete, incomplete_question=incomplete_question)
    response_data_b = get_response_openai_with_sampling(
        input_prompt_incomplete, persona="You are an expert mathematical problem solver.",
        model=args.verify_model, temperature=args.temperature, n=args.max_attempts
    )
    if not response_data_b:
        logging.error(f"ID {variant['variant_id']}: Round B generation failed")
        variant["verification"] = {
            "round_A_passed": round_A_passed, "round_B_passed": False, "round_C_passed": False, "is_valid": False,
            "round_A": round_A_info,
            "round_B": {"total_attempts": 0, "all_attempts": []},
            "round_C": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    record_tokens(data, response_data_b["model_type"],
                  response_data_b["prompt_tokens"], response_data_b["completion_tokens"])
    round_b_attempts = []
    round_b_has_correct = False
    for attempt_num, candidate_text in enumerate(response_data_b["candidates"], start=1):
        model_answer = extract_answer_from_response(candidate_text)
        if model_answer is None:
            is_correct = False
            judge_result = "no_answer_tag"
            judge_method = "none"
        else:
            is_correct, judge_prompt_tokens, judge_completion_tokens, judge_model_type = judge_answer_equivalence(
                incomplete_question, model_answer, ground_truth
            )
            if judge_model_type == "heuristic":
                judge_result = "heuristic_match" if is_correct else "heuristic_fail"
                judge_method = "heuristic"
            else:
                judge_result = "orm_match" if is_correct else "orm_fail"
                judge_method = "orm"
            record_tokens(data, judge_model_type, judge_prompt_tokens, judge_completion_tokens)
        attempt_record = {
            "attempt": attempt_num, "full_response": candidate_text,
            "model_answer": model_answer if model_answer else "N/A",
            "judge_result": judge_result, "judge_method": judge_method, "is_correct": is_correct
        }
        round_b_attempts.append(attempt_record)
        if is_correct:
            round_b_has_correct = True
    round_b_passed = not round_b_has_correct
    if round_b_passed:
        logging.info(f"ID {variant['variant_id']}: ✓ Round B PASSED - All {args.max_attempts} answers ≠ ground_truth")
    else:
        logging.info(f"ID {variant['variant_id']}: ✗ Round B FAILED - At least 1 answer = ground_truth")
        variant["verification"] = {
            "round_A_passed": round_A_passed, "round_B_passed": False, "round_C_passed": False, "is_valid": False,
            "round_A": round_A_info,
            "round_B": {"total_attempts": len(round_b_attempts), "all_attempts": round_b_attempts},
            "round_C": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    logging.info(f"ID {variant['variant_id']}: Starting Round C - Testing WITH removed conditions...")
    removed_conditions_text = "\n".join(f"- {c}" for c in removed_conditions)
    input_prompt_complete = safe_format(
        prompt_template_complete,
        incomplete_question=incomplete_question, removed_conditions=removed_conditions_text
    )
    response_data_c = get_response_openai_with_sampling(
        input_prompt_complete, persona="You are an expert mathematical problem solver.",
        model=args.verify_model, temperature=args.temperature, n=args.max_attempts
    )
    if not response_data_c:
        logging.error(f"ID {variant['variant_id']}: Round C generation failed")
        variant["verification"] = {
            "round_A_passed": round_A_passed, "round_B_passed": True, "round_C_passed": False, "is_valid": False,
            "round_A": round_A_info,
            "round_B": {"total_attempts": len(round_b_attempts), "all_attempts": round_b_attempts},
            "round_C": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    record_tokens(data, response_data_c["model_type"],
                  response_data_c["prompt_tokens"], response_data_c["completion_tokens"])
    round_c_attempts = []
    round_c_has_correct = False
    success_at_attempt = None
    for attempt_num, candidate_text in enumerate(response_data_c["candidates"], start=1):
        model_answer = extract_answer_from_response(candidate_text)
        if model_answer is None:
            is_correct = False
            judge_result = "no_answer_tag"
            judge_method = "none"
        else:
            is_correct, judge_prompt_tokens, judge_completion_tokens, judge_model_type = judge_answer_equivalence(
                incomplete_question + " [With conditions: " + ", ".join(removed_conditions) + "]",
                model_answer, ground_truth
            )
            if judge_model_type == "heuristic":
                judge_result = "heuristic_match" if is_correct else "heuristic_fail"
                judge_method = "heuristic"
            else:
                judge_result = "orm_match" if is_correct else "orm_fail"
                judge_method = "orm"
            record_tokens(data, judge_model_type, judge_prompt_tokens, judge_completion_tokens)
        attempt_record = {
            "attempt": attempt_num, "full_response": candidate_text,
            "model_answer": model_answer if model_answer else "N/A",
            "judge_result": judge_result, "judge_method": judge_method, "is_correct": is_correct
        }
        round_c_attempts.append(attempt_record)
        if is_correct and not round_c_has_correct:
            round_c_has_correct = True
            success_at_attempt = attempt_num
    round_c_passed = round_c_has_correct
    if round_c_passed:
        logging.info(f"ID {variant['variant_id']}: ✓ Round C PASSED - Answer {success_at_attempt}/{args.max_attempts} = ground_truth")
    else:
        logging.info(f"ID {variant['variant_id']}: ✗ Round C FAILED - All {args.max_attempts} answers ≠ ground_truth")
    is_valid = round_b_passed and round_c_passed
    if is_valid:
        logging.info(f"ID {variant['variant_id']}: 🎉 VALID - Both rounds passed!")
    else:
        logging.info(f"ID {variant['variant_id']}: ✗ INVALID")
    variant["verification"] = {
        "round_A_passed": round_A_passed, "round_B_passed": round_b_passed, "round_C_passed": round_c_passed, "is_valid": is_valid,
        "round_A": round_A_info,
        "round_B": {"total_attempts": len(round_b_attempts), "all_attempts": round_b_attempts},
        "round_C": {"total_attempts": len(round_c_attempts), "success_at_attempt": success_at_attempt, "all_attempts": round_c_attempts},
        "ground_truth": ground_truth
    }
    return variant

def verify_incomplete_questions_with_three_rounds(data):
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
    for variant in variants:
        try:
            verified_variant = verify_single_variant(
                data, variant, prompt_template_incomplete, prompt_template_complete, ground_truth
            )
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

def process_with_jsonl_parallel(dataset, output_path, process_func, desc):
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
    all_data = existing_data + read_jsonl(jsonl_path)[len(existing_data):]
    if all_data:
        all_data.sort(key=lambda x: x.get('id', 0))
        write_json(output_path, all_data)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
    return len(all_data) == total_len

def filter_valid_data(final_path, num_missing):
    try:
        dataset = read_json(final_path)
    except FileNotFoundError:
        logging.error(f"Final data file not found: {final_path}")
        logging.error(f"Please run the verification step first.")
        return
    except Exception as e:
        logging.error(f"Failed to read {final_path}: {e}")
        return

    if not dataset:
        logging.error(f"Dataset is empty in {final_path}")
        return

    valid_data = []
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
    # Round A (LLM pre-verification) 统计
    round_a_enabled_count = 0
    round_a_pass_count = 0
    round_a_correctness_pass = 0
    round_a_validity_pass = 0
    # Round B/C 统计
    round_b_pass_count = 0
    round_c_pass_count = 0
    both_pass_count = 0
    round_c_attempt_distribution = {}
    judge_method_distribution = {"heuristic": 0, "orm": 0}

    # 数据完整性检查
    seen_variant_ids = set()
    integrity_errors = []

    for data in dataset:
        for variant in data.get("removal_variants", []):
            total_variants += 1

            # 数据完整性检查
            variant_id = variant.get("variant_id", "")

            # 检查1: variant_id 唯一性
            if variant_id in seen_variant_ids:
                integrity_errors.append(f"Duplicate variant_id: {variant_id}")
            seen_variant_ids.add(variant_id)

            # 检查2: removed + remaining = total conditions
            removed_conditions = variant.get("removed_conditions", [])
            remaining_conditions = variant.get("remaining_conditions", [])
            num_conditions_extracted = data.get("num_conditions", 0)
            if len(removed_conditions) + len(remaining_conditions) != num_conditions_extracted:
                integrity_errors.append(
                    f"{variant_id}: removed({len(removed_conditions)}) + remaining({len(remaining_conditions)}) "
                    f"!= total({num_conditions_extracted})"
                )

            # 检查3: num_missing = len(removed_conditions)
            variant_num_missing = variant.get("num_missing", 0)
            if variant_num_missing != len(removed_conditions):
                integrity_errors.append(
                    f"{variant_id}: num_missing({variant_num_missing}) != len(removed_conditions)({len(removed_conditions)})"
                )

            # Round A 统计
            llm_verification = variant.get("llm_verification")
            if llm_verification is not None:
                round_a_enabled_count += 1
                if llm_verification.get("overall_passed", False):
                    round_a_pass_count += 1
                if llm_verification.get("correctness_passed", False):
                    round_a_correctness_pass += 1
                if llm_verification.get("validity_passed", False):
                    round_a_validity_pass += 1

            # Round B/C 统计
            verification = variant.get("verification", {})
            round_b_passed = verification.get("round_B_passed", False)
            round_c_passed = verification.get("round_C_passed", False)
            if round_b_passed:
                round_b_pass_count += 1
            if round_c_passed:
                round_c_pass_count += 1
            if round_b_passed and round_c_passed:
                both_pass_count += 1
            if verification.get("is_valid", False):
                round_c_info = verification.get("round_C", {})
                success_at_attempt = round_c_info.get("success_at_attempt")
                if success_at_attempt:
                    round_c_attempt_distribution[success_at_attempt] = \
                        round_c_attempt_distribution.get(success_at_attempt, 0) + 1
                    all_attempts = round_c_info.get("all_attempts", [])
                    if success_at_attempt <= len(all_attempts):
                        success_attempt_record = all_attempts[success_at_attempt - 1]
                        judge_method = success_attempt_record.get("judge_method", "orm")
                        judge_method_distribution[judge_method] = \
                            judge_method_distribution.get(judge_method, 0) + 1
                valid_item = {
                    "id": variant["variant_id"],
                    "original_id": data.get("id"),
                    "data_source": data.get("data_source", ""),
                    "difficulty": data.get("difficulty", ""),
                    "transformation_type": "condition_removal",
                    "num_missing": num_missing,
                    "original_question": data["original_question"],
                    "ground_truth": data.get("ground_truth", ""),
                    "incomplete_question": variant["incomplete_question"],
                    "all_extracted_conditions": data.get("extracted_conditions", []),
                    "num_conditions_extracted": data.get("num_conditions", 0),
                    "removed_conditions": variant["removed_conditions"],
                    "remaining_conditions": variant["remaining_conditions"],
                    "verification": verification
                }
                valid_data.append(valid_item)
                valid_variants += 1
    # 按照原始ID排序
    valid_data.sort(key=lambda x: x.get('original_id', 0))
    output_path = final_path.replace(f"_final_n{num_missing}.json", f"_valid_n{num_missing}.json")
    write_json(output_path, valid_data)

    # 生成简略版 sample_valid.json
    sample_valid_data = []
    for item in valid_data:
        sample_item = {
            "id": item.get("id"),
            "difficulty": item.get("difficulty"),
            "num_missing": item.get("num_missing"),
            "original_question": item.get("original_question"),
            "ground_truth": item.get("ground_truth"),
            "incomplete_question": item.get("incomplete_question"),
            "all_extracted_conditions": item.get("all_extracted_conditions"),
            "num_conditions_extracted": item.get("num_conditions_extracted"),
            "removed_conditions": item.get("removed_conditions"),
            "remaining_conditions": item.get("remaining_conditions")
        }
        sample_valid_data.append(sample_item)

    sample_output_path = final_path.replace(f"_final_n{num_missing}.json", f"_sample_valid_n{num_missing}.json")
    write_json(sample_output_path, sample_valid_data)
    logging.info(f"Sample valid data saved to: {sample_output_path}")

    # 报告完整性检查结果
    if integrity_errors:
        logging.warning(f"Found {len(integrity_errors)} data integrity issues:")
        for error in integrity_errors[:10]:  # 只显示前10个错误
            logging.warning(f"  - {error}")
        if len(integrity_errors) > 10:
            logging.warning(f"  ... and {len(integrity_errors) - 10} more errors")

    print("\n" + "="*70)
    print("MISSING INFORMATION PROBLEM (MIP) DATASET STATISTICS")
    print("="*70)
    print(f"Configuration: num_missing = {num_missing}")
    print(f"Minimum conditions required: {num_missing + 1}")
    print(f"Original problems (after filtering): {total_original}")
    print(f"\nTotal removal variants generated: {total_variants}")

    # 数据完整性检查报告
    if integrity_errors:
        print(f"\n⚠️  Data Integrity Issues: {len(integrity_errors)} errors found")
        print(f"   (Check logs for details)")
    else:
        print(f"\n✓ Data Integrity Check: All passed")

    if total_variants == 0:
        print(f"\n⚠️  WARNING: No variants found in final data!")
        print(f"   This usually means the final_n{num_missing}.json file is corrupted.")
        print(f"   Please rerun with --force to regenerate all data.")
        return

    print(f"\n📊 Three-Round Verification Results:")

    # Round A 统计（必须执行）
    print(f"\n  Round A (LLM Pre-verification - Rewrite Quality):")
    if round_a_enabled_count > 0:
        print(f"    Overall passed: {round_a_pass_count}/{total_variants} ({round_a_pass_count/total_variants*100:.1f}%)")
        print(f"    ├─ Correctness passed: {round_a_correctness_pass}/{total_variants} ({round_a_correctness_pass/total_variants*100:.1f}%)")
        print(f"    └─ Validity passed: {round_a_validity_pass}/{total_variants} ({round_a_validity_pass/total_variants*100:.1f}%)")
    else:
        print(f"    ⚠️  Round A data missing - this should not happen!")

    # Round B 和 C 的分母是通过 Round A 的变体数
    round_b_c_denominator = round_a_pass_count if round_a_pass_count > 0 else total_variants

    print(f"\n  Round B (Necessity - without conditions → can't solve):")
    if round_b_c_denominator > 0:
        print(f"    Passed: {round_b_pass_count}/{round_b_c_denominator} ({round_b_pass_count/round_b_c_denominator*100:.1f}%)")
    else:
        print(f"    Passed: 0/0 (N/A - all variants failed Round A)")

    print(f"\n  Round C (Sufficiency - with conditions → can solve):")
    if round_b_c_denominator > 0:
        print(f"    Passed: {round_c_pass_count}/{round_b_c_denominator} ({round_c_pass_count/round_b_c_denominator*100:.1f}%)")
    else:
        print(f"    Passed: 0/0 (N/A - all variants failed Round A)")

    print(f"\n  Final Result (Round A + B + C all passed):")
    print(f"    VALID variants: {both_pass_count}/{total_variants} ({both_pass_count/total_variants*100:.1f}%)")
    print(f"\nValid removal variants: {valid_variants}")
    if valid_variants > 0:
        print(f"\nRound C Success Distribution (when valid):")
        for attempt in sorted(round_c_attempt_distribution.keys()):
            count = round_c_attempt_distribution[attempt]
            print(f"  Candidate {attempt}: {count} variants ({count/valid_variants*100:.1f}%)")
        print(f"\nJudge Method Distribution (Round C success):")
        for method, count in judge_method_distribution.items():
            print(f"  {method.capitalize()}: {count} ({count/valid_variants*100:.1f}%)")
    gpt4o_prompt_rate = 2.5
    gpt4o_completion_rate = 10.0
    gpt4o_mini_prompt_rate = 0.15
    gpt4o_mini_completion_rate = 0.6
    print(f"\n💰 GPT-4o Token Usage:")
    print(f"  Prompt: {total_gpt4o_prompt:,}")
    print(f"  Completion: {total_gpt4o_completion:,}")
    print(f"  Cost = {total_gpt4o_prompt}/1e6*{gpt4o_prompt_rate} + {total_gpt4o_completion}/1e6*{gpt4o_completion_rate} = ${total_gpt4o_prompt/1e6*gpt4o_prompt_rate + total_gpt4o_completion/1e6*gpt4o_completion_rate:.6f}")
    print(f"\n💰 GPT-4o-mini Token Usage:")
    print(f"  Prompt: {total_gpt4o_mini_prompt:,}")
    print(f"  Completion: {total_gpt4o_mini_completion:,}")
    print(f"  Cost = {total_gpt4o_mini_prompt}/1e6*{gpt4o_mini_prompt_rate} + {total_gpt4o_mini_completion}/1e6*{gpt4o_mini_completion_rate} = ${total_gpt4o_mini_prompt/1e6*gpt4o_mini_prompt_rate + total_gpt4o_mini_completion/1e6*gpt4o_mini_completion_rate:.6f}")
    print(f"\n🖥️  Local Model Token Usage:")
    print(f"  Prompt: {total_local_prompt:,}")
    print(f"  Completion: {total_local_completion:,}")
    print(f"\n🎯 Heuristic Checks (free):")
    print(f"  Total heuristic validations: {total_heuristic_count:,}")
    print(f"\nOutput (full): {output_path}")
    print(f"Output (sample): {sample_output_path}")
    print("="*70)

def construction_workflow():
    input_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    output_dir = args.output_dir
    if not os.path.exists(input_path):
        logging.error(f"Input not found: {input_path}")
        logging.error(f"Current working directory: {os.getcwd()}")
        logging.error(f"Please make sure you run this script from ~/ReliableMath directory")
        return
    dataset = read_json(input_path)

    # 统一字段名：将 question 重命名为 original_question
    for item in dataset:
        if "question" in item and "original_question" not in item:
            item["original_question"] = item.pop("question")

    if args.test_mode:
        dataset = dataset[:5]
        logging.info("TEST MODE: First 5 items")
    os.makedirs(output_dir, exist_ok=True)
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
    print(f"Model (extract): {args.extract_model}")
    print(f"Model (rewrite): {args.rewrite_model}")
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
    
    # ============================================================
    # [1/5] Extract Conditions
    # ============================================================
    extract_path = os.path.join(output_dir, f"{args.dataset}_conditions.json")
    if os.path.exists(extract_path) and not args.force:
        existing_conditions = read_json(extract_path)
        if len(existing_conditions) == len(dataset):
            print(f"\n[1/5] ✓ Conditions already extracted ({len(existing_conditions)} items), skipping...")
            dataset = existing_conditions
        else:
            print(f"\n[1/5] Extracting conditions (continuing from {len(existing_conditions)}/{len(dataset)})")
            process_with_jsonl_parallel(dataset, extract_path, extract_conditions_only, "Extracting conditions")
            dataset = read_json(extract_path)
    else:
        print("\n[1/5] Extracting conditions (parallel)")
        process_with_jsonl_parallel(dataset, extract_path, extract_conditions_only, "Extracting conditions")
        dataset = read_json(extract_path)
    
    # ============================================================
    # [2/5] Filter by num_conditions  ★新增★
    # ============================================================
    print(f"\n[2/5] Filtering samples by num_conditions (must be >= {args.num_missing + 1})")
    
    filtered_conditions_path = os.path.join(
        output_dir,
        f"{args.dataset}_conditions_filtered_n{args.num_missing}.json"
    )
    
    # 检查是否已经过滤过
    if os.path.exists(filtered_conditions_path) and not args.force:
        print(f"  ✓ Filtered conditions already exist, loading...")
        dataset = read_json(filtered_conditions_path)
        remaining_count = len(dataset)
        print(f"  Loaded {remaining_count} samples")
    else:
        # 执行过滤
        filtered_conditions_path, remaining_count = filter_by_num_conditions(extract_path, args.num_missing)
        dataset = read_json(filtered_conditions_path)
    
    # 如果过滤后没有数据，终止流程
    if remaining_count == 0:
        print(f"\n⚠️  WARNING: No samples remain after filtering!")
        print(f"   All samples have num_conditions < {args.num_missing + 1}")
        print(f"   Suggestions:")
        print(f"   1. Use a smaller --num_missing value")
        print(f"   2. Use a different dataset with more complex problems")
        print(f"   3. Check your extraction prompt (maybe conditions are under-extracted)")
        return
    
    print(f"  ✓ Proceeding with {remaining_count} samples for variant generation...")
    
    # ============================================================
    # [3/5] Generate Removal Variants (使用过滤后的数据)
    # ============================================================
    variants_path = os.path.join(output_dir, f"{args.dataset}_variants_n{args.num_missing}.json")
    generate_func = lambda data: generate_removal_variants(data, args.num_missing)
    if os.path.exists(variants_path) and not args.force:
        existing_variants = read_json(variants_path)
        if len(existing_variants) == remaining_count:
            print(f"\n[3/5] ✓ Variants already generated ({len(existing_variants)} items), skipping...")
            dataset = existing_variants
        else:
            print(f"\n[3/5] Generating removal variants (n={args.num_missing}, continuing from {len(existing_variants)}/{remaining_count})")
            process_with_jsonl_parallel(dataset, variants_path, generate_func, f"Generating variants (n={args.num_missing})")
            dataset = read_json(variants_path)
    else:
        print(f"\n[3/5] Generating removal variants (n={args.num_missing}, parallel)")
        process_with_jsonl_parallel(dataset, variants_path, generate_func, f"Generating variants (n={args.num_missing})")
        dataset = read_json(variants_path)
    
    # ============================================================
    # [4/5] Three-Round Verification
    # ============================================================
    final_path = os.path.join(output_dir, f"{args.dataset}_final_n{args.num_missing}.json")
    if os.path.exists(final_path) and not args.force:
        existing_final = read_json(final_path)
        if len(existing_final) == len(dataset):
            print(f"\n[4/5] ✓ Verification already complete ({len(existing_final)} items), skipping...")
        else:
            print(f"\n[4/5] Three-round verification (n={args.max_attempts}, continuing from {len(existing_final)}/{len(dataset)})")
            print(f"  Round A: LLM pre-verification (rewrite quality)")
            print(f"  Round B: WITHOUT conditions (must all fail)")
            print(f"  Round C: WITH conditions (at least one succeeds)")
            process_with_jsonl_parallel(dataset, final_path, verify_incomplete_questions_with_three_rounds, "Three-round verification")
    else:
        print(f"\n[4/5] Three-round verification (n={args.max_attempts}, parallel)")
        print(f"  Round A: LLM pre-verification (rewrite quality)")
        print(f"  Round B: WITHOUT conditions (must all fail)")
        print(f"  Round C: WITH conditions (at least one succeeds)")
        process_with_jsonl_parallel(dataset, final_path, verify_incomplete_questions_with_three_rounds, "Three-round verification")
    
    # ============================================================
    # [5/5] Filter Valid Data
    # ============================================================
    print("\n[5/5] Filtering valid data")
    filter_valid_data(final_path, args.num_missing)
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()