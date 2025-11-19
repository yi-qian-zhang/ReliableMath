#!/usr/bin/env python3
"""
Contradiction Dataset Construction - 矛盾条件生成（与Removal模块深度整合）
输入数据: 原始数学问题 (question) + 标准答案 (ground_truth) + 难度标签 (difficulty)

新架构 (5步流程):
Step 1. 提取条件 (extract_conditions_only): 使用 GPT-4o-mini 提取问题中的所有关键条件
Step 2. 生成矛盾变体 (generate_contradiction_variants): 为每个条件生成对应的矛盾版本
Step 3. 验证 A - 改写质量检查: LLM 快速验证改写正确性和问题有效性
Step 4. 验证 B - 矛盾有效性: 给模型矛盾问题，vLLM sampling 8次，全都 ≠ ground_truth → 通过
Step 5. 验证 C - 原题可解性: 给模型原始问题，vLLM sampling 8次，至少1个 = ground_truth → 通过
最终数据集: 只包含三轮验证都通过的有效矛盾问题

Deployment Locations:
- Development: /home/user/ReliableMath/code/contradiction_construction.py
- Production: /data2/yiqianzhang/ReliableMath/code/contradiction_construction/contradiction_construction.py
"""
import sys
import os

# Smart path detection for different deployment scenarios
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)

# Case 1: /data2/yiqianzhang/ReliableMath/code/contradiction_construction/contradiction_construction.py
# Case 2: /home/user/ReliableMath/code/contradiction_construction.py
if script_name == 'contradiction_construction.py':
    parent_dir = os.path.dirname(script_dir)
    parent_name = os.path.basename(parent_dir)

    # If parent is 'contradiction_construction', we're in Case 1
    if parent_name == 'contradiction_construction':
        # Add /data2/yiqianzhang/ReliableMath/code to path
        code_dir = os.path.dirname(parent_dir)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
    # If parent is 'code', we're in Case 2
    elif parent_name == 'code':
        # Add /home/user/ReliableMath to path
        repo_root = os.path.dirname(parent_dir)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        # Also add code directory
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
    else:
        # Fallback: add parent directory
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

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

# Import from removal module for answer verification
# This will work in both deployment scenarios after path setup above
try:
    from deepscaler.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy, extract_answer
    from deepscaler.system_prompts import ORM_PROMPT
except ImportError as e:
    logging.error(f"Failed to import from deepscaler: {e}")
    logging.error(f"sys.path: {sys.path}")
    logging.error(f"script_dir: {script_dir}")
    logging.error(f"Please ensure deepscaler is in the correct location:")
    logging.error(f"  - Case 1: /data2/yiqianzhang/ReliableMath/code/deepscaler")
    logging.error(f"  - Case 2: /home/user/ReliableMath/code/deepscaler")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Contradiction Dataset Construction - Three-Round Verification")
parser.add_argument("--extract_model", default="gpt-4o-mini", help="Model for condition extraction")
parser.add_argument("--analysis_model", default="DeepSeek-R1-Distill-Qwen-7B", help="Model for analysis and rewrite")
parser.add_argument("--verify_model", default="DeepSeek-R1-Distill-Qwen-7B", help="Model for vLLM sampling verification")
parser.add_argument("--judge_model", default="gpt-4o-mini", help="Model for LLM-as-Judge (ORM fallback)")
parser.add_argument("--data_dir", default="data/solve", help="Input directory")
parser.add_argument("--output_dir", default="data/construct_contradiction", help="Output directory")
parser.add_argument("--prompt_dir", default="prompt/v4-comp/rewrite", help="Prompt directory")
parser.add_argument("--dataset", default="aime", help="Dataset name")
parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for vLLM sampling")
parser.add_argument("--max_attempts", default=8, type=int, help="Max sampling attempts for verification")
parser.add_argument("--threads", default=8, type=int, help="Number of parallel threads")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 5 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items")
parser.add_argument("--use_math_orm", action='store_true', help="Enable LLM ORM for answer verification")
parser.add_argument("--use_llm_verification", action='store_true', help="Enable Round A LLM pre-verification before Round B/C")
args = parser.parse_args()

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
        logging.error(f"Model {model} not found in api_keys.json")
        return "", 0, 0, "unknown"
    model_name, key, url = random.choice(model_options[model])
    client = OpenAI(api_key=key, base_url=url)
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": input_prompt})
    prompt_text = (persona + "\n" if persona else "") + input_prompt
    prompt_tokens = count_tokens(prompt_text, model_name)
    # Determine model type for token tracking
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
    """vLLM sampling - returns n candidates"""
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
        model_type = "local"
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
    """解析JSON响应"""
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
    """Extract answer from model response (handles <think> tags)"""
    if "</think>" not in response_text:
        return None
    response_text = response_text.split("</think>", 1)[1].strip()
    return extract_answer(response_text)
def judge_answer_equivalence(question, model_answer, ground_truth):
    """Judge if model answer equals ground truth (reused from removal module)"""
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
        input_prompt = ORM_USER_TEMPLATE.format(
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
    """提取多选题选项"""
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

def ensure_options_in_question(contradicted_question, original_question):
    """确保多选题选项在改写后的问题中保留"""
    original_options = extract_multiple_choice_options(original_question)
    if not original_options:
        return contradicted_question
    has_options = extract_multiple_choice_options(contradicted_question)
    if has_options:
        return contradicted_question
    logging.debug(f"Multiple-choice options missing, restoring...")
    enter_patterns = [
        r'Enter the letters?.*', r'Enter the correct options?.*', r'separated by commas.*'
    ]
    enter_match = None
    for pattern in enter_patterns:
        enter_match = re.search(pattern, contradicted_question, re.IGNORECASE | re.DOTALL)
        if enter_match:
            break
    if enter_match:
        before_enter = contradicted_question[:enter_match.start()].strip()
        enter_text = contradicted_question[enter_match.start():].strip()
        pure_options = original_options
        for pattern in enter_patterns:
            pure_options = re.sub(pattern, '', pure_options, flags=re.IGNORECASE | re.DOTALL).strip()
        return f"{before_enter}\n\n{pure_options}\n\n{enter_text}"
    else:
        return f"{contradicted_question}\n\n{original_options}"

def extract_conditions_only(data):
    """Step 1: 提取问题中的所有关键条件"""
    prompt_path = os.path.join(args.prompt_dir, "extract.txt")
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["extracted_condition"] = []
        return data
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    input_prompt = prompt_template.format(original_math_question=data["question"])
    response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        input_prompt,
        persona="You are an expert at analyzing mathematical problems.",
        model=args.extract_model,
        temperature=0.0
    )
    record_tokens(data, model_type, prompt_tokens, completion_tokens)
    # Parse conditions - expecting a list
    conditions = parse_json_response(response, fallback=[])
    if not isinstance(conditions, list):
        logging.warning(f"ID {data['id']}: Expected list, got {type(conditions)}")
        conditions = []
    # Clean up conditions
    cleaned_conditions = []
    for cond in conditions:
        if isinstance(cond, str):
            cond = cond.strip()
            # Remove common prefixes
            for prefix in ["Condition:", "条件:", "-", "•", "**", "1.", "2.", "3.", "4.", "5."]:
                if cond.startswith(prefix):
                    cond = cond[len(prefix):].strip()
            if cond:
                cleaned_conditions.append(cond)
    data["extracted_condition"] = cleaned_conditions
    data["num_conditions"] = len(cleaned_conditions)
    # 检测是否为多选题
    is_multiple_choice = extract_multiple_choice_options(data["question"]) is not None
    data["is_multiple_choice"] = is_multiple_choice
    logging.info(f"ID {data['id']}: Extracted {len(cleaned_conditions)} conditions" +
                 (" (multiple-choice)" if is_multiple_choice else ""))
    return data

def verify_rewrite_with_llm(data, contradicted_question, extracted_condition, other_conditions, variant_idx):
    """
    Round A: LLM Pre-verification - 改写质量检查
    Verify 1: 改写是否只修改了指定的1个条件
    Verify 2: 问题是否有效（没有删除question stem，不是无穷多解）
    """
    original_question = data["question"]
    other_conditions_text = "\n".join(f"- {c}" for c in other_conditions) if other_conditions else "(None - only one condition)"

    # Verification 1: 改写正确性
    correctness_prompt_path = os.path.join(args.prompt_dir, "contradict_verify_rewrite_correctness.txt")
    if not os.path.exists(correctness_prompt_path):
        logging.warning(f"Correctness prompt not found: {correctness_prompt_path}")
        return {"correctness_passed": None, "validity_passed": None}

    with open(correctness_prompt_path, 'r', encoding='utf-8') as f:
        correctness_template = f.read()

    correctness_prompt = correctness_template.format(
        original_question=original_question,
        rewritten_question=contradicted_question,
        contradicted_condition=extracted_condition,
        other_conditions=other_conditions_text
    )

    correctness_response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        correctness_prompt,
        persona="You are an expert verifier.",
        model=args.judge_model,
        temperature=0.0
    )
    record_tokens(data, model_type, prompt_tokens, completion_tokens)

    # 解析判断结果
    correctness_passed = "True" in correctness_response or "true" in correctness_response.lower()
    correctness_analysis = correctness_response.strip()

    # Verification 2: 问题有效性
    validity_prompt_path = os.path.join(args.prompt_dir, "contradict_verify_problem_validity.txt")
    if not os.path.exists(validity_prompt_path):
        logging.warning(f"Validity prompt not found: {validity_prompt_path}")
        return {
            "correctness_passed": correctness_passed,
            "correctness_analysis": correctness_analysis,
            "validity_passed": None
        }

    with open(validity_prompt_path, 'r', encoding='utf-8') as f:
        validity_template = f.read()

    validity_prompt = validity_template.format(
        original_question=original_question,
        rewritten_question=contradicted_question,
        contradicted_condition=extracted_condition
    )

    validity_response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        validity_prompt,
        persona="You are an expert verifier.",
        model=args.judge_model,
        temperature=0.0
    )
    record_tokens(data, model_type, prompt_tokens, completion_tokens)

    # 解析判断结果
    validity_passed = "Valid" in validity_response and "Invalid" not in validity_response
    validity_analysis = validity_response.strip()

    # 综合结果
    overall_passed = correctness_passed and validity_passed

    if overall_passed:
        logging.info(f"ID {data['id']}_contradict_{variant_idx}: ✓ LLM verification PASSED")
    else:
        reason = []
        if not correctness_passed:
            reason.append("改写不正确")
        if not validity_passed:
            reason.append("问题无效")
        logging.warning(f"ID {data['id']}_contradict_{variant_idx}: ✗ LLM verification FAILED ({', '.join(reason)})")

    return {
        "overall_passed": overall_passed,
        "correctness_passed": correctness_passed,
        "correctness_analysis": correctness_analysis,
        "validity_passed": validity_passed,
        "validity_analysis": validity_analysis
    }

def generate_contradiction_variants(data):
    """Step 2: 为每个条件生成对应的矛盾版本"""
    conditions = data.get("extracted_condition", [])
    N = len(conditions)
    if N == 0:
        logging.warning(f"ID {data['id']}: No conditions extracted, skipping")
        data["contradiction_variants"] = []
        return data
    logging.info(f"ID {data['id']}: Generating contradictions for {N} conditions")
    # Load prompts
    analysis_prompt_path = os.path.join(args.prompt_dir, "contradict_analysis.txt")
    rewrite_prompt_path = os.path.join(args.prompt_dir, "contradict_rewrite.txt")
    if not os.path.exists(analysis_prompt_path) or not os.path.exists(rewrite_prompt_path):
        logging.error(f"Prompt files not found")
        data["contradiction_variants"] = []
        return data
    with open(analysis_prompt_path, 'r', encoding='utf-8') as f:
        analysis_template = f.read()
    with open(rewrite_prompt_path, 'r', encoding='utf-8') as f:
        rewrite_template = f.read()
    variants = []
    for idx, condition in enumerate(conditions):
        # Step 2.1: Analyze how to contradict this condition
        analysis_prompt = analysis_template.format(
            original_math_question=data["question"],
            original_answer=data["ground_truth"],
            extracted_condition=condition
        )
        analysis_response, p_tokens, c_tokens, m_type = get_response_openai(
            analysis_prompt,
            persona="You are an expert mathematical problem analyzer.",
            model=args.analysis_model,
            temperature=0.0
        )
        record_tokens(data, m_type, p_tokens, c_tokens)
        # 解析 Analysis 部分
        analysis = ""
        if "### Analysis ###" in analysis_response and "### Rewritten Mathematical Question ###" in analysis_response:
            parts = analysis_response.split("### Rewritten Mathematical Question ###")
            if len(parts) == 2:
                analysis_part = parts[0].split("### Analysis ###")
                if len(analysis_part) == 2:
                    analysis = analysis_part[1].strip()
        else:
            analysis = analysis_response.strip()
        # If analysis is empty, skip this condition
        if not analysis.strip() or len(analysis.strip()) < 10:
            logging.warning(f"ID {data['id']}_contradict_{idx}: Analysis is empty, skipping")
            continue
        # Step 2.2: Generate contradicted question
        rewrite_prompt = rewrite_template.format(
            original_math_question=data["question"],
            original_answer=data["ground_truth"],
            extracted_condition=condition
        )
        rewrite_response, p_tokens, c_tokens, m_type = get_response_openai(
            rewrite_prompt,
            persona="You are an expert at rewriting mathematical problems.",
            model=args.analysis_model,
            temperature=0.0
        )
        record_tokens(data, m_type, p_tokens, c_tokens)
        # 解析 Rewritten Question 部分
        contradicted_question = ""
        if "### Rewritten Mathematical Question ###" in rewrite_response:
            contradicted_question = rewrite_response.split("### Rewritten Mathematical Question ###")[-1].strip()
        else:
            contradicted_question = rewrite_response.strip()
        # Clean up
        for prefix in ["Rewritten Question:", "Answer:", "###", "**", '"', "'"]:
            contradicted_question = contradicted_question.replace(prefix, "").strip()
        if not contradicted_question or len(contradicted_question) < 20:
            logging.warning(f"ID {data['id']}_contradict_{idx}: Rewritten question is too short, skipping")
            continue
        # 多选题保护
        if data.get("is_multiple_choice"):
            original_contradicted = contradicted_question
            contradicted_question = ensure_options_in_question(contradicted_question, data["question"])
            if original_contradicted != contradicted_question:
                logging.debug(f"ID {data['id']}_contradict_{idx}: ✓ Restored multiple-choice options")
        # Round A: LLM Pre-verification (可选)
        other_conditions = [c for i, c in enumerate(conditions) if i != idx]
        llm_verification = None
        if args.use_llm_verification:
            llm_verification = verify_rewrite_with_llm(
                data, contradicted_question, condition,
                other_conditions, idx
            )
        variant = {
            "variant_id": f"{data['id']}_contradict_{idx}",
            "extracted_condition": condition,
            "analysis": analysis,
            "contradicted_question": contradicted_question,
            "llm_verification": llm_verification
        }
        variants.append(variant)
        logging.info(f"ID {data['id']}_contradict_{idx}: ✓ Generated contradiction")
    data["contradiction_variants"] = variants
    logging.info(f"ID {data['id']}: Generated {len(variants)}/{N} contradiction variants")
    return data

def verify_single_variant(data, variant, prompt_template, ground_truth):
    """
    Three-round verification for a single variant
    Round B: Test contradicted question (should fail)
    Round C: Test original question (should succeed)
    """
    contradicted_question = variant["contradicted_question"]
    original_question = data["question"]
    variant_id = variant["variant_id"]

    # Round B: Test contradicted question (should all fail)
    logging.info(f"ID {variant_id}: Starting Round B - Testing contradicted question...")
    input_prompt_b = prompt_template.format(question=contradicted_question)
    response_data_b = get_response_openai_with_sampling(
        input_prompt_b,
        persona="You are an expert mathematical problem solver.",
        model=args.verify_model,
        temperature=args.temperature,
        n=args.max_attempts
    )
    if not response_data_b:
        logging.error(f"ID {variant_id}: Round B generation failed")
        variant["verification"] = {
            "round_b_passed": False,
            "round_c_passed": False,
            "is_valid": False,
            "round_b": {"total_attempts": 0, "all_attempts": []},
            "round_c": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    record_tokens(data, response_data_b["model_type"],
                  response_data_b["prompt_tokens"], response_data_b["completion_tokens"])

    # Check Round B attempts - they should ALL be wrong
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
                contradicted_question, model_answer, ground_truth
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
        if is_correct:
            round_b_has_correct = True

    round_b_passed = not round_b_has_correct
    if round_b_passed:
        logging.info(f"ID {variant_id}: ✓ Round B PASSED - All {args.max_attempts} answers ≠ ground_truth")
    else:
        logging.info(f"ID {variant_id}: ✗ Round B FAILED - At least 1 answer = ground_truth")
        variant["verification"] = {
            "round_b_passed": False,
            "round_c_passed": False,
            "is_valid": False,
            "round_b": {"total_attempts": len(round_b_attempts), "all_attempts": round_b_attempts},
            "round_c": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant

    # Round C: Test original question (at least one should succeed)
    logging.info(f"ID {variant_id}: Starting Round C - Testing original question...")
    input_prompt_c = prompt_template.format(question=original_question)
    response_data_c = get_response_openai_with_sampling(
        input_prompt_c,
        persona="You are an expert mathematical problem solver.",
        model=args.verify_model,
        temperature=args.temperature,
        n=args.max_attempts
    )
    if not response_data_c:
        logging.error(f"ID {variant_id}: Round C generation failed")
        variant["verification"] = {
            "round_b_passed": True,
            "round_c_passed": False,
            "is_valid": False,
            "round_b": {"total_attempts": len(round_b_attempts), "all_attempts": round_b_attempts},
            "round_c": {"total_attempts": 0, "all_attempts": []},
            "ground_truth": ground_truth
        }
        return variant
    record_tokens(data, response_data_c["model_type"],
                  response_data_c["prompt_tokens"], response_data_c["completion_tokens"])

    # Check Round C attempts - at least one should be correct
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
                original_question, model_answer, ground_truth
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
        round_c_attempts.append(attempt_record)
        if is_correct and not round_c_has_correct:
            round_c_has_correct = True
            success_at_attempt = attempt_num

    round_c_passed = round_c_has_correct
    if round_c_passed:
        logging.info(f"ID {variant_id}: ✓ Round C PASSED - Answer {success_at_attempt}/{args.max_attempts} = ground_truth")
    else:
        logging.info(f"ID {variant_id}: ✗ Round C FAILED - All {args.max_attempts} answers ≠ ground_truth (original question may be unsolvable)")

    is_valid = round_b_passed and round_c_passed
    if is_valid:
        logging.info(f"ID {variant_id}: 🎉 VALID - Both rounds passed!")
    else:
        logging.info(f"ID {variant_id}: ✗ INVALID")

    variant["verification"] = {
        "round_b_passed": round_b_passed,
        "round_c_passed": round_c_passed,
        "is_valid": is_valid,
        "round_b": {"total_attempts": len(round_b_attempts), "all_attempts": round_b_attempts},
        "round_c": {"total_attempts": len(round_c_attempts), "success_at_attempt": success_at_attempt, "all_attempts": round_c_attempts},
        "ground_truth": ground_truth
    }
    return variant

def verify_contradiction_validity_with_three_rounds(data):
    """Step 3-5: 三轮验证矛盾条件的有效性"""
    prompt_path = os.path.join(args.prompt_dir, "verify_question.txt")
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        return data
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    ground_truth = str(data.get("ground_truth", "")).strip()
    variants = data.get("contradiction_variants", [])
    if not variants:
        return data
    for variant in variants:
        try:
            verified_variant = verify_single_variant(data, variant, prompt_template, ground_truth)
            variant_id = verified_variant["variant_id"]
            for i, v in enumerate(data["contradiction_variants"]):
                if v["variant_id"] == variant_id:
                    data["contradiction_variants"][i] = verified_variant
                    break
        except Exception as e:
            logging.error(f"Error verifying {variant['variant_id']}: {e}")
            import traceback
            traceback.print_exc()
    return data

def process_with_jsonl_parallel(dataset, output_path, process_func, desc):
    """并行处理数据集，支持断点续传"""
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
    # Merge all data and save as JSON
    all_data = existing_data + read_jsonl(jsonl_path)[len(existing_data):]
    if all_data:
        all_data.sort(key=lambda x: x.get('id', 0))
        write_json(output_path, all_data)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
    return len(all_data) == total_len

def filter_valid_data(final_path):
    """过滤出有效的矛盾条件数据"""
    dataset = read_json(final_path)
    valid_data = []
    # Token usage statistics
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
    round_b_pass_count = 0
    round_c_pass_count = 0
    both_pass_count = 0
    round_c_attempt_distribution = {}
    judge_method_distribution = {"heuristic": 0, "orm": 0}
    for data in dataset:
        for variant in data.get("contradiction_variants", []):
            total_variants += 1
            verification = variant.get("verification", {})
            round_b_passed = verification.get("round_b_passed", False)
            round_c_passed = verification.get("round_c_passed", False)
            if round_b_passed:
                round_b_pass_count += 1
            if round_c_passed:
                round_c_pass_count += 1
            if round_b_passed and round_c_passed:
                both_pass_count += 1
            if verification.get("is_valid", False):
                round_c_info = verification.get("round_c", {})
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
                    "data_source": data.get("data_source", ""),
                    "difficulty": data.get("difficulty", ""),
                    "transformation_type": "contradiction",
                    "original_question": data["question"],
                    "ground_truth": data.get("ground_truth", ""),
                    "extracted_condition": variant["extracted_condition"],
                    "contradict_question": variant["contradicted_question"],
                    "verification": verification,
                    "original_id": data["id"],
                    "all_extracted_conditions": data.get("extracted_condition", []),
                    "num_conditions_extracted": data.get("num_conditions", 0)
                }
                valid_data.append(valid_item)
                valid_variants += 1
    valid_data.sort(key=lambda x: x.get('original_id', 0))
    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)
    print("\n" + "="*70)
    print("CONTRADICTION DATASET STATISTICS")
    print("="*70)
    print(f"Original problems: {total_original}")
    print(f"\nTotal contradiction variants generated: {total_variants}")
    if total_variants == 0:
        print(f"\n⚠️  WARNING: No variants found in final data!")
        print(f"   This usually means the final.json file is corrupted.")
        print(f"   Please rerun with --force to regenerate all data.")
        return
    print(f"\n📊 Three-Round Verification Results:")
    print(f"  Round B passed (contradicted → can't solve): {round_b_pass_count} ({round_b_pass_count/total_variants*100:.1f}%)")
    print(f"  Round C passed (original → can solve): {round_c_pass_count} ({round_c_pass_count/total_variants*100:.1f}%)")
    print(f"  Both rounds passed (VALID): {both_pass_count} ({both_pass_count/total_variants*100:.1f}%)")
    print(f"\nValid contradiction variants: {valid_variants}")
    if valid_variants > 0:
        print(f"\nRound C Success Distribution (when valid):")
        for attempt in sorted(round_c_attempt_distribution.keys()):
            count = round_c_attempt_distribution[attempt]
            print(f"  Candidate {attempt}: {count} variants ({count/valid_variants*100:.1f}%)")
        print(f"\nJudge Method Distribution (Round C success):")
        for method, count in judge_method_distribution.items():
            print(f"  {method.capitalize()}: {count} ({count/valid_variants*100:.1f}%)")
    # Cost estimation
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
    print(f"\nOutput: {output_path}")
    print("="*70)

def construction_workflow():
    """主流程：矛盾条件数据集构建"""
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
    print("CONTRADICTION CONSTRUCTION - THREE-ROUND VERIFICATION")
    print("="*70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Prompt: {args.prompt_dir}")
    print(f"Model (extract): {args.extract_model}")
    print(f"Model (analysis/rewrite): {args.analysis_model}")
    print(f"Model (verify vLLM): {args.verify_model}")
    print(f"Model (judge ORM fallback): {args.judge_model}")
    print(f"Use Math ORM: {'✓ Enabled' if args.use_math_orm else '✗ Disabled (heuristic only)'}")
    print(f"Use LLM Verification (Round A): {'✓ Enabled' if args.use_llm_verification else '✗ Disabled'}")
    print(f"Temperature: {args.temperature}")
    print(f"Sampling n: {args.max_attempts}")
    print(f"Parallel threads: {args.threads}")
    print(f"Items: {len(dataset)}")
    if args.force:
        print(f"Mode: FORCE (reprocessing all)")
    print("="*70)
    # Step 1: Extract conditions
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
    # Step 2: Generate contradiction variants
    variants_path = os.path.join(output_dir, f"{args.dataset}_contradictions.json")
    if os.path.exists(variants_path) and not args.force:
        existing_variants = read_json(variants_path)
        if len(existing_variants) == len(dataset):
            print(f"\n[2/4] ✓ Contradictions already generated ({len(existing_variants)} items), skipping...")
            dataset = existing_variants
        else:
            print(f"\n[2/4] Generating contradictions (continuing from {len(existing_variants)}/{len(dataset)})")
            process_with_jsonl_parallel(dataset, variants_path, generate_contradiction_variants, "Generating contradictions")
            dataset = read_json(variants_path)
    else:
        print(f"\n[2/4] Generating contradictions (parallel)")
        process_with_jsonl_parallel(dataset, variants_path, generate_contradiction_variants, "Generating contradictions")
        dataset = read_json(variants_path)
    # Step 3-5: Three-round verification
    final_path = os.path.join(output_dir, f"{args.dataset}_final.json")
    if os.path.exists(final_path) and not args.force:
        existing_final = read_json(final_path)
        if len(existing_final) == len(dataset):
            print(f"\n[3/4] ✓ Verification already complete ({len(existing_final)} items), skipping...")
        else:
            print(f"\n[3/4] Three-round verification (n={args.max_attempts}, continuing from {len(existing_final)}/{len(dataset)})")
            if args.use_llm_verification:
                print(f"  Round A: LLM pre-verification (rewrite quality)")
            print(f"  Round B: Contradicted question (must all fail)")
            print(f"  Round C: Original question (at least one succeeds)")
            process_with_jsonl_parallel(dataset, final_path, verify_contradiction_validity_with_three_rounds, "Three-round verification")
    else:
        print(f"\n[3/4] Three-round verification (n={args.max_attempts}, parallel)")
        if args.use_llm_verification:
            print(f"  Round A: LLM pre-verification (rewrite quality)")
        print(f"  Round B: Contradicted question (must all fail)")
        print(f"  Round C: Original question (at least one succeeds)")
        process_with_jsonl_parallel(dataset, final_path, verify_contradiction_validity_with_three_rounds, "Three-round verification")
    print("\n[4/4] Filtering valid data")
    filter_valid_data(final_path)
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()
