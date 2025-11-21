#!/usr/bin/env python3
"""
Contradiction Dataset Construction - 矛盾条件生成
输入数据: 原始数学问题 (question) + 标准答案 (ground_truth)

新架构 (3步流程):
Step 1. 提取条件 (extract_conditions): 使用 GPT-4o-mini 提取问题中的所有关键条件
Step 2. 生成矛盾变体 (generate_contradiction_variants):
    - 2.1 分析如何添加矛盾 (DeepSeek-R1-Distill-Qwen-7B)
    - 2.2 生成矛盾问题 (DeepSeek-R1-Distill-Qwen-7B)
Step 3. 验证矛盾条件 (verify_contradiction_validity):
    - 3.1 验证单条件修改 (GPT-4o-mini)
    - 3.2 提取矛盾描述 (DeepSeek-R1-Distill-Qwen-7B)
    - 3.3 vLLM采样验证不可解性 (DeepSeek-R1-Distill-Qwen-7B, n=8)
    - 3.4 提取不可解原因 (DeepSeek-V3)

修改说明:
- 统一术语: 只使用 "add contradiction"
- 明确定义: contradicted_condition = 添加的矛盾条件
- 简化prompt: 减少歧义，提高输出稳定性
"""
import sys
import os

# Smart path detection for different deployment scenarios
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)

if script_name == 'contradiction_construction.py':
    parent_dir = os.path.dirname(script_dir)
    parent_name = os.path.basename(parent_dir)

    if parent_name == 'contradiction_construction':
        code_dir = os.path.dirname(parent_dir)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
    elif parent_name == 'code':
        repo_root = os.path.dirname(parent_dir)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
    else:
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

def safe_format(template, **kwargs):
    """
    Safely format a template string, escaping curly braces in both template and arguments.
    
    This prevents KeyError when mathematical notation like {m/s} or {x, y} appears
    in questions or arguments.
    """
    escaped_template = template.replace('{', '{{').replace('}', '}}')
    
    for key in kwargs.keys():
        escaped_template = escaped_template.replace('{{' + key + '}}', '{' + key + '}')
    
    escaped_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            escaped_kwargs[key] = value.replace('{', '{{').replace('}', '}}')
        else:
            escaped_kwargs[key] = value
    
    return escaped_template.format(**escaped_kwargs)

parser = argparse.ArgumentParser(description="Contradiction Dataset Construction")
parser.add_argument("--model", default="gpt-4o-mini", help="Model for condition extraction and single-condition verification")
parser.add_argument("--analysis_model", default="DeepSeek-R1-Distill-Qwen-32B", help="Model for analysis and rewrite")
parser.add_argument("--verify_model", default="DeepSeek-R1-Distill-Qwen-32B", help="Model for vLLM sampling verification")
parser.add_argument("--extract_model", default="gpt-4o-mini", help="Model for extracting unsolvable reasons")
parser.add_argument("--judge_model", default="DeepSeek-R1-Distill-Qwen-32B", help="Model for LLM-as-Judge (ORM fallback)")
parser.add_argument("--data_dir", default="data/solve", help="Input directory")
parser.add_argument("--output_dir", default="data/construct_contradiction", help="Output directory")
parser.add_argument("--prompt_dir", default="prompt/contradict_data", help="Prompt directory")
parser.add_argument("--dataset", default="polaris", help="Dataset name")
parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for vLLM sampling")
parser.add_argument("--max_attempts", default=8, type=int, help="Max sampling attempts for verification")
parser.add_argument("--threads", default=8, type=int, help="Number of parallel threads")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 5 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items")
parser.add_argument("--use_math_orm", action='store_true', help="Enable LLM ORM for answer verification")
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
    if "deepseek_v3_prompt_lengths" not in data:
        data["deepseek_v3_prompt_lengths"] = []
        data["deepseek_v3_completion_lengths"] = []
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
    elif model_type == "deepseek-v3":
        data["deepseek_v3_prompt_lengths"].append(prompt_tokens)
        data["deepseek_v3_completion_lengths"].append(completion_tokens)
    elif model_type == "heuristic":
        data["heuristic_count"] += 1

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    if model is None:
        model = args.model
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
    
    is_local_model = "localhost" in url or "127.0.0.1" in url
    if is_local_model:
        model_type = "local"
    elif "deepseek" in model.lower() and ("v3" in model.lower() or "chat" in model.lower()):
        model_type = "deepseek-v3"
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

def extract_conditions(data):
    """Step 1: 提取问题中的所有关键条件"""
    prompt_path = os.path.join(args.prompt_dir, "extract.txt")
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["extracted_condition"] = []
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = safe_format(prompt_template, original_math_question=data["original_question"])
    
    response, prompt_tokens, completion_tokens, model_type = get_response_openai(
        input_prompt,
        persona="You are an expert at analyzing mathematical problems.",
        model=args.model,
        temperature=0.0
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
            for prefix in ["Condition:", "条件:", "-", "•", "**", "1.", "2.", "3.", "4.", "5."]:
                if cond.startswith(prefix):
                    cond = cond[len(prefix):].strip()
            if cond:
                cleaned_conditions.append(cond)
    
    data["extracted_condition"] = cleaned_conditions
    data["num_conditions"] = len(cleaned_conditions)
    logging.info(f"ID {data['id']}: Extracted {len(cleaned_conditions)} conditions")
    
    return data

def generate_contradiction_variants(data):
    """Step 2: 为每个条件生成对应的矛盾版本"""
    conditions = data.get("extracted_condition", [])
    N = len(conditions)
    
    if N == 0:
        logging.warning(f"ID {data['id']}: No conditions extracted, skipping")
        data["contradiction_variants"] = []
        return data
    
    logging.info(f"ID {data['id']}: Generating contradictions for {N} conditions")
    
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
        analysis_prompt = safe_format(
            analysis_template,
            original_math_question=data["original_question"],
            original_answer=data["ground_truth"],
            original_condition=condition
        )
        
        analysis, p_tokens, c_tokens, m_type = get_response_openai(
            analysis_prompt,
            persona="You are an expert mathematical problem analyzer.",
            model=args.analysis_model,
            temperature=0.0
        )
        
        record_tokens(data, m_type, p_tokens, c_tokens)
        
        if "### Analysis ###" in analysis:
            analysis = analysis.split("### Analysis ###")[-1].strip()
        if "### Rewritten Mathematical Question ###" in analysis:
            analysis = analysis.split("### Rewritten Mathematical Question ###")[0].strip()
        
        if not analysis.strip() or len(analysis.strip()) < 10:
            logging.warning(f"ID {data['id']}_contradict_{idx}: Analysis is empty, skipping")
            continue
        
        # Step 2.2: Generate contradicted question
        rewrite_prompt = safe_format(
            rewrite_template,
            original_math_question=data["original_question"],
            original_answer=data["ground_truth"],
            original_condition=condition,
            analysis=analysis
        )
        
        rewrite_response, p_tokens, c_tokens, m_type = get_response_openai(
            rewrite_prompt,
            persona="You are an expert at rewriting mathematical problems.",
            model=args.analysis_model,
            temperature=0.0
        )
        
        record_tokens(data, m_type, p_tokens, c_tokens)
        
        contradicted_question = rewrite_response.strip()
        if "### Rewritten Mathematical Question ###" in contradicted_question:
            contradicted_question = contradicted_question.split("### Rewritten Mathematical Question ###")[-1].strip()
        
        for prefix in ["Rewritten Question:", "Answer:", "###", "**", '"', "'"]:
            contradicted_question = contradicted_question.replace(prefix, "").strip()
        
        if not contradicted_question or len(contradicted_question) < 20:
            logging.warning(f"ID {data['id']}_contradict_{idx}: Rewritten question is too short, skipping")
            continue
        
        variant = {
            "variant_id": f"{data['id']}_contradict_{idx}",
            "original_condition": condition,
            "analysis": analysis,
            "contradicted_question": contradicted_question
        }
        variants.append(variant)
        logging.info(f"ID {data['id']}_contradict_{idx}: ✓ Generated contradiction")
    
    data["contradiction_variants"] = variants
    logging.info(f"ID {data['id']}: Generated {len(variants)}/{N} contradiction variants")
    
    return data

def verify_contradiction_validity(data):
    """Step 3: 验证矛盾条件的有效性（使用vLLM sampling）"""
    variants = data.get("contradiction_variants", [])
    if not variants:
        return data
    
    verify_s1_path = os.path.join(args.prompt_dir, "contradict_verify_s1.txt")
    verify_s2_path = os.path.join(args.prompt_dir, "contradict_verify_s2.txt")
    unsolve_s3_path = os.path.join(args.prompt_dir, "contradict_unsolve_s3.txt")
    
    prompt_files = [verify_s1_path, verify_s2_path, unsolve_s3_path]
    for path in prompt_files:
        if not os.path.exists(path):
            logging.error(f"Prompt file not found: {path}")
            return data
    
    with open(verify_s1_path, 'r', encoding='utf-8') as f:
        verify_s1_template = f.read()
    with open(verify_s2_path, 'r', encoding='utf-8') as f:
        verify_s2_template = f.read()
    with open(unsolve_s3_path, 'r', encoding='utf-8') as f:
        unsolve_s3_template = f.read()
    
    ground_truth = str(data.get("ground_truth", "")).strip()
    
    for variant in variants:
        variant_id = variant["variant_id"]
        logging.info(f"ID {variant_id}: Starting verification...")
        
        # Step 3.1: Verify single condition change
        verify_s1_prompt = safe_format(
            verify_s1_template,
            original_question=data["original_question"],
            rewritten_question=variant["contradicted_question"]
        )
        
        verify_s1_response, p_tokens, c_tokens, m_type = get_response_openai(
            verify_s1_prompt,
            persona="You are an excellent mathematical problem verifier.",
            model=args.model,
            temperature=0.0
        )
        
        record_tokens(data, m_type, p_tokens, c_tokens)
        
        single_condition_verified = "True" in verify_s1_response or "true" in verify_s1_response
        
        if not single_condition_verified:
            logging.warning(f"ID {variant_id}: ✗ Failed single condition verification")
            variant["verification"] = {
                "single_condition_verified": False,
                "is_valid": False,
                "failure_reason": "multiple_conditions_changed"
            }
            continue
        
        logging.info(f"ID {variant_id}: ✓ Single condition verified")
        
        # Step 3.2: Extract contradicted condition
        verify_s2_prompt = safe_format(
            verify_s2_template,
            original_question=data["original_question"],
            original_condition=variant["original_condition"],
            rewritten_question=variant["contradicted_question"]
        )
        
        contradicted_condition, p_tokens, c_tokens, m_type = get_response_openai(
            verify_s2_prompt,
            persona="You are an excellent mathematical problem verifier.",
            model=args.analysis_model,
            temperature=0.0
        )
        
        record_tokens(data, m_type, p_tokens, c_tokens)
        
        if "### Contradicted Condition ###" in contradicted_condition:
            contradicted_condition = contradicted_condition.split("### Contradicted Condition ###")[-1].strip()
        
        contradicted_condition = contradicted_condition.strip()
        
        if not contradicted_condition or len(contradicted_condition) < 5:
            logging.warning(f"ID {variant_id}: ✗ Failed to extract contradicted condition")
            variant["verification"] = {
                "single_condition_verified": True,
                "contradicted_condition_extracted": False,
                "is_valid": False,
                "failure_reason": "no_contradicted_condition"
            }
            continue
        
        logging.info(f"ID {variant_id}: ✓ Contradicted condition extracted")
        
        # Step 3.3: vLLM Sampling Verification - 核心验证！
        logging.info(f"ID {variant_id}: Starting vLLM sampling verification (n={args.max_attempts})...")
        
        verification_prompt = f"""Solve the following mathematical problem:
{variant["contradicted_question"]}
Provide your answer in the format: The answer is <answer>.
"""
        
        response_data = get_response_openai_with_sampling(
            verification_prompt,
            persona="You are an expert mathematical problem solver.",
            model=args.verify_model,
            temperature=args.temperature,
            n=args.max_attempts
        )
        
        if not response_data:
            logging.error(f"ID {variant_id}: vLLM sampling failed")
            variant["verification"] = {
                "single_condition_verified": True,
                "contradicted_condition_extracted": True,
                "contradicted_condition": contradicted_condition,
                "vllm_sampling_passed": False,
                "is_valid": False,
                "failure_reason": "vllm_sampling_failed"
            }
            continue
        
        record_tokens(data, response_data["model_type"],
                      response_data["prompt_tokens"], response_data["completion_tokens"])
        
        # Check all candidates - ALL should be WRONG
        sampling_attempts = []
        has_correct_answer = False
        
        for attempt_num, candidate_text in enumerate(response_data["candidates"], start=1):
            model_answer = extract_answer_from_response(candidate_text)
            
            if model_answer is None:
                is_correct = False
                judge_result = "no_answer_tag"
                judge_method = "none"
            else:
                is_correct, judge_prompt_tokens, judge_completion_tokens, judge_model_type = judge_answer_equivalence(
                    variant["contradicted_question"], model_answer, ground_truth
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
            sampling_attempts.append(attempt_record)
            
            if is_correct:
                has_correct_answer = True
        
        # Validation logic: ALL attempts should be WRONG (can't solve)
        vllm_sampling_passed = not has_correct_answer
        
        if vllm_sampling_passed:
            logging.info(f"ID {variant_id}: ✓ vLLM sampling passed - All {args.max_attempts} answers ≠ ground_truth")
        else:
            logging.warning(f"ID {variant_id}: ✗ vLLM sampling failed - At least 1 answer = ground_truth")
            variant["verification"] = {
                "single_condition_verified": True,
                "contradicted_condition_extracted": True,
                "contradicted_condition": contradicted_condition,
                "vllm_sampling_passed": False,
                "sampling_attempts": sampling_attempts,
                "is_valid": False,
                "failure_reason": "still_solvable"
            }
            continue
        
        # Step 3.4: Extract concise unsolvable reason
        unsolvability_evidence = f"The model failed to produce the correct answer '{ground_truth}' across all {args.max_attempts} attempts when solving the contradicted question."
        
        unsolve_s3_prompt = safe_format(
            unsolve_s3_template,
            original_question=data["original_question"],
            rewritten_question=variant["contradicted_question"],
            verification_result=unsolvability_evidence
        )
        
        unsolvable_reason, p_tokens, c_tokens, m_type = get_response_openai(
            unsolve_s3_prompt,
            persona="You are an excellent mathematical analyst.",
            model=args.extract_model,
            temperature=0.0
        )
        
        record_tokens(data, m_type, p_tokens, c_tokens)
        
        if "### Unsolvable Reason ###" in unsolvable_reason:
            unsolvable_reason = unsolvable_reason.split("### Unsolvable Reason ###")[-1].strip()
        
        unsolvable_reason = unsolvable_reason.strip()
        
        # Mark as valid
        variant["verification"] = {
            "single_condition_verified": True,
            "contradicted_condition_extracted": True,
            "contradicted_condition": contradicted_condition,
            "vllm_sampling_passed": True,
            "sampling_attempts": sampling_attempts,
            "unsolvable_reason": unsolvable_reason,
            "is_valid": True
        }
        
        variant["contradicted_condition"] = contradicted_condition
        variant["unsolvable_reason"] = unsolvable_reason
        
        logging.info(f"ID {variant_id}: 🎉 VALID - All checks passed!")
    
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
    
    total_gpt4o_prompt = sum(sum(d.get("gpt4o_prompt_lengths", [])) for d in dataset)
    total_gpt4o_completion = sum(sum(d.get("gpt4o_completion_lengths", [])) for d in dataset)
    total_gpt4o_mini_prompt = sum(sum(d.get("gpt4o_mini_prompt_lengths", [])) for d in dataset)
    total_gpt4o_mini_completion = sum(sum(d.get("gpt4o_mini_completion_lengths", [])) for d in dataset)
    total_local_prompt = sum(sum(d.get("local_prompt_lengths", [])) for d in dataset)
    total_local_completion = sum(sum(d.get("local_completion_lengths", [])) for d in dataset)
    total_deepseek_v3_prompt = sum(sum(d.get("deepseek_v3_prompt_lengths", [])) for d in dataset)
    total_deepseek_v3_completion = sum(sum(d.get("deepseek_v3_completion_lengths", [])) for d in dataset)
    total_heuristic_count = sum(d.get("heuristic_count", 0) for d in dataset)
    
    total_original = len(dataset)
    total_variants = 0
    valid_variants = 0
    failure_reasons = {}
    
    for data in dataset:
        for variant in data.get("contradiction_variants", []):
            total_variants += 1
            verification = variant.get("verification", {})
            is_valid = verification.get("is_valid", False)
            
            if is_valid:
                valid_item = {
                    "id": variant["variant_id"],
                    "data_source": data.get("data_source", ""),
                    "difficulty": data.get("difficulty", ""),
                    "transformation_type": "contradiction",
                    "original_question": data["original_question"],
                    "ground_truth": data.get("ground_truth", ""),
                    "original_condition": variant["original_condition"],
                    "contradicted_question": variant["contradicted_question"],
                    "contradicted_condition": variant.get("contradicted_condition", ""),
                    "unsolvable_reason": variant.get("unsolvable_reason", ""),
                    "verification": verification,
                    "original_id": data["id"],
                    "all_extracted_conditions": data.get("extracted_condition", []),
                    "num_conditions_extracted": data.get("num_conditions", 0)
                }
                valid_data.append(valid_item)
                valid_variants += 1
            else:
                reason = verification.get("failure_reason", "unknown")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    valid_data.sort(key=lambda x: x.get('original_id', 0))
    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)

    # 生成简略版 sample_valid.json (去掉详细的 verification 信息)
    sample_valid_data = []
    for item in valid_data:
        sample_item = {
            "id": item.get("id"),
            "data_source": item.get("data_source"),
            "difficulty": item.get("difficulty"),
            "transformation_type": item.get("transformation_type"),
            "original_question": item.get("original_question"),
            "ground_truth": item.get("ground_truth"),
            "original_condition": item.get("original_condition"),
            "contradicted_question": item.get("contradicted_question"),
            "contradicted_condition": item.get("contradicted_condition"),
            "unsolvable_reason": item.get("unsolvable_reason"),
            "all_extracted_conditions": item.get("all_extracted_conditions"),
            "num_conditions_extracted": item.get("num_conditions_extracted")
        }
        sample_valid_data.append(sample_item)

    sample_output_path = final_path.replace("_final.json", "_sample_valid.json")
    write_json(sample_output_path, sample_valid_data)
    logging.info(f"Sample valid data saved to: {sample_output_path}")

    print("\n" + "="*70)
    print("CONTRADICTION DATASET STATISTICS")
    print("="*70)
    print(f"Original problems: {total_original}")
    print(f"Total contradiction variants generated: {total_variants}")
    print(f"Valid contradiction variants: {valid_variants} ({valid_variants/total_variants*100:.1f}%)" if total_variants > 0 else "Valid: 0")
    
    if failure_reasons:
        print(f"\n📊 Failure Reason Distribution:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} ({count/total_variants*100:.1f}%)")
    
    gpt4o_prompt_rate = 2.5
    gpt4o_completion_rate = 10.0
    gpt4o_mini_prompt_rate = 0.15
    gpt4o_mini_completion_rate = 0.6
    
    print(f"\n💰 GPT-4o Token Usage:")
    print(f"  Prompt: {total_gpt4o_prompt:,}")
    print(f"  Completion: {total_gpt4o_completion:,}")
    print(f"  Cost ≈ ${total_gpt4o_prompt/1e6*gpt4o_prompt_rate + total_gpt4o_completion/1e6*gpt4o_completion_rate:.4f}")
    
    print(f"\n💰 GPT-4o-mini Token Usage:")
    print(f"  Prompt: {total_gpt4o_mini_prompt:,}")
    print(f"  Completion: {total_gpt4o_mini_completion:,}")
    print(f"  Cost ≈ ${total_gpt4o_mini_prompt/1e6*gpt4o_mini_prompt_rate + total_gpt4o_mini_completion/1e6*gpt4o_mini_completion_rate:.4f}")
    
    print(f"\n🖥️  Local Model (DeepSeek-R1-Distill-Qwen-7B) Token Usage:")
    print(f"  Prompt: {total_local_prompt:,}")
    print(f"  Completion: {total_local_completion:,}")
    
    print(f"\n🤖 DeepSeek-V3 Token Usage:")
    print(f"  Prompt: {total_deepseek_v3_prompt:,}")
    print(f"  Completion: {total_deepseek_v3_completion:,}")
    
    print(f"\n🎯 Heuristic Checks (free):")
    print(f"  Total heuristic validations: {total_heuristic_count:,}")

    print(f"\nOutput (full): {output_path}")
    print(f"Output (sample): {sample_output_path}")
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
    print("CONTRADICTION DATASET CONSTRUCTION (Clarified Version)")
    print("="*70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Prompt: {args.prompt_dir}")
    print(f"Model (extract): {args.model}")
    print(f"Model (analysis/rewrite): {args.analysis_model}")
    print(f"Model (vLLM sampling): {args.verify_model}")
    print(f"Model (extract reason): {args.extract_model}")
    print(f"Use Math ORM: {'✓ Enabled' if args.use_math_orm else '✗ Disabled (heuristic only)'}")
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
            print(f"\n[1/3] ✓ Conditions already extracted ({len(existing_conditions)} items), skipping...")
            dataset = existing_conditions
        else:
            print(f"\n[1/3] Extracting conditions (continuing from {len(existing_conditions)}/{len(dataset)})")
            process_with_jsonl_parallel(dataset, extract_path, extract_conditions, "Extracting conditions")
            dataset = read_json(extract_path)
    else:
        print("\n[1/3] Extracting conditions (parallel)")
        process_with_jsonl_parallel(dataset, extract_path, extract_conditions, "Extracting conditions")
        dataset = read_json(extract_path)
    
    # Step 2: Generate contradiction variants
    variants_path = os.path.join(output_dir, f"{args.dataset}_contradictions.json")
    if os.path.exists(variants_path) and not args.force:
        existing_variants = read_json(variants_path)
        if len(existing_variants) == len(dataset):
            print(f"\n[2/3] ✓ Contradictions already generated ({len(existing_variants)} items), skipping...")
            dataset = existing_variants
        else:
            print(f"\n[2/3] Generating contradictions (continuing from {len(existing_variants)}/{len(dataset)})")
            process_with_jsonl_parallel(dataset, variants_path, generate_contradiction_variants, "Generating contradictions")
            dataset = read_json(variants_path)
    else:
        print(f"\n[2/3] Generating contradictions (parallel)")
        process_with_jsonl_parallel(dataset, variants_path, generate_contradiction_variants, "Generating contradictions")
        dataset = read_json(variants_path)
    
    # Step 3: Verify contradictions with vLLM sampling
    final_path = os.path.join(output_dir, f"{args.dataset}_final.json")
    if os.path.exists(final_path) and not args.force:
        existing_final = read_json(final_path)
        if len(existing_final) == len(dataset):
            print(f"\n[3/3] ✓ Verification already complete ({len(existing_final)} items), skipping...")
        else:
            print(f"\n[3/3] Verifying contradictions with vLLM sampling (continuing from {len(existing_final)}/{len(dataset)})")
            print(f"  - Verify single condition change")
            print(f"  - Extract contradicted condition")
            print(f"  - vLLM sampling (n={args.max_attempts}) - all should fail")
            print(f"  - Extract unsolvable reason")
            process_with_jsonl_parallel(dataset, final_path, verify_contradiction_validity, "Verifying contradictions")
    else:
        print(f"\n[3/3] Verifying contradictions with vLLM sampling (parallel)")
        print(f"  - Verify single condition change")
        print(f"  - Extract contradicted condition")
        print(f"  - vLLM sampling (n={args.max_attempts}) - all should fail")
        print(f"  - Extract unsolvable reason")
        process_with_jsonl_parallel(dataset, final_path, verify_contradiction_validity, "Verifying contradictions")
    
    print("\n[4/3] Filtering valid data")
    filter_valid_data(final_path)
    
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()
