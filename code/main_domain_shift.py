#!/usr/bin/env python3
"""
Domain Shift Construction Pipeline
Type 2: 指代错误（条件与问题范围不匹配）
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
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Domain Shift Dataset Construction")
parser.add_argument("--model", default="gpt-4o", help="Model name")
parser.add_argument("--data_dir", default="./data/solve", help="Input directory")
parser.add_argument("--output_dir", default="./data/domain_shift", help="Output directory")
parser.add_argument("--prompt_dir", default="./prompt/domain_shift", help="Prompt directory")
parser.add_argument("--dataset", default="polaris", help="Dataset name")
parser.add_argument("--temperature", default=0.0, type=float, help="Temperature")
parser.add_argument("--test_mode", action='store_true', help="Test mode - first 2 items")
args = parser.parse_args()

# Load API config
try:
    model_options = json.load(open("./data/api_keys.json", "r"))
except FileNotFoundError:
    logging.error("api_keys.json not found!")
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

def extract_numbers(text):
    """Extract all numbers for comparison"""
    return sorted(re.findall(r'\b\d+\.?\d*\b', text))

# ============= API Functions =============

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    if model is None:
        model = args.model
    
    if model not in model_options:
        logging.error(f"Model {model} not found")
        return "", 0, 0
    
    model_name, key, url = random.choice(model_options[model])
    client = OpenAI(api_key=key, base_url=url)
    
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": input_prompt})
    
    prompt_text = (persona + "\n" if persona else "") + input_prompt
    prompt_tokens = count_tokens(prompt_text, model_name)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
                stream=False
            )
            
            response_text = completion.choices[0].message.content
            
            try:
                prompt_tokens = completion.usage.prompt_tokens
                completion_tokens = completion.usage.completion_tokens
            except:
                completion_tokens = count_tokens(response_text, model_name)
            
            return response_text, prompt_tokens, completion_tokens
            
        except Exception as e:
            logging.warning(f'API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
    
    return "", 0, 0

def parse_json_response(response, fallback=None):
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass
    return fallback if fallback is not None else {}

# ============= Step 1: Extract =============

def extract_conditions_and_question(data):
    """Step 1: Extract conditions and question"""
    prompt_path = os.path.join(args.prompt_dir, "extract.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        original_question=data["question"],
        ground_truth=data.get("ground_truth", "")
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at analyzing mathematical problem structure.",
        model=args.model,
        temperature=0.0
    )
    
    if "prompt_lengths" not in data:
        data["prompt_lengths"] = []
        data["completion_lengths"] = []
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Parse extraction
    extraction = parse_json_response(response, {
        "conditions": [],
        "question": data["question"]
    })
    
    data["extracted_conditions"] = extraction.get("conditions", [])
    data["extracted_question"] = extraction.get("question", "")
    data["original_numbers"] = extract_numbers(data["question"])
    
    logging.info(f"ID {data['id']}: Extracted {len(data['extracted_conditions'])} conditions")
    
    return data

# ============= Step 2: Identify Domain =============

def identify_question_domain(data):
    """Step 2: Identify current question domain"""
    prompt_path = os.path.join(args.prompt_dir, "identify_domain.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        original_question=data["question"],
        conditions=json.dumps(data.get("extracted_conditions", []), indent=2),
        question=data.get("extracted_question", "")
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at identifying problem domains.",
        model=args.model,
        temperature=0.0
    )
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Parse domain identification
    domain_info = parse_json_response(response, {
        "current_domain": "mathematical",
        "domain_shiftable": False
    })
    
    data["current_domain"] = domain_info.get("current_domain", "")
    data["domain_shiftable"] = domain_info.get("domain_shiftable", False)
    data["possible_target_domains"] = domain_info.get("possible_target_domains", [])
    
    logging.info(f"ID {data['id']}: Domain={data['current_domain']}, Shiftable={data['domain_shiftable']}")
    
    return data

# ============= Step 3: Change Domain =============

def change_question_domain(data):
    """Step 3: Change question to different domain"""
    
    # Skip if not shiftable
    if not data.get("domain_shiftable", False):
        logging.info(f"ID {data['id']}: Skipping - not domain shiftable")
        data["domain_shifted_question"] = ""
        return data
    
    prompt_path = os.path.join(args.prompt_dir, "change_domain.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        original_question=data["question"],
        conditions=json.dumps(data.get("extracted_conditions", []), indent=2),
        question=data.get("extracted_question", ""),
        current_domain=data.get("current_domain", "mathematical"),
        possible_domains=json.dumps(data.get("possible_target_domains", []))
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at transforming question domains.",
        model=args.model,
        temperature=0.0
    )
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Clean response
    if "Rewritten Problem:" in response:
        response = response.split("Rewritten Problem:")[-1].strip()
    
    response = response.replace("**", "").strip()
    
    data["domain_shifted_question"] = response
    
    logging.info(f"ID {data['id']}: Domain shifted question generated")
    
    return data

# ============= Step 4: Verify Preservation =============

def verify_conditions_preserved(data):
    """Step 4: Verify conditions are preserved"""
    
    if not data.get("domain_shifted_question"):
        data["preservation_verification"] = {"overall": 0.0, "recommendation": "reject"}
        return data
    
    prompt_path = os.path.join(args.prompt_dir, "verify_preservation.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        original_question=data["question"],
        original_conditions=json.dumps(data.get("extracted_conditions", []), indent=2),
        original_question_part=data.get("extracted_question", ""),
        rewritten_question=data.get("domain_shifted_question", "")
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at verifying problem transformations.",
        model=args.model,
        temperature=0.0
    )
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Parse verification
    verification = parse_json_response(response, {
        "overall": 0.5,
        "recommendation": "reject"
    })
    
    # Automated number check
    rewritten_numbers = extract_numbers(data.get("domain_shifted_question", ""))
    numbers_match = (data.get("original_numbers", []) == rewritten_numbers)
    
    verification["numbers_match_automated"] = numbers_match
    verification["original_numbers"] = data.get("original_numbers", [])
    verification["rewritten_numbers"] = rewritten_numbers
    
    data["preservation_verification"] = verification
    
    overall = verification.get("overall", 0)
    rec = verification.get("recommendation", "reject")
    logging.info(f"ID {data['id']}: Preservation - score={overall:.2f}, {rec}, numbers={numbers_match}")
    
    return data

# ============= Step 5: Verify Domain Mismatch =============

def verify_domain_mismatch(data):
    """Step 5: Verify domain mismatch makes it unsolvable"""
    
    if not data.get("domain_shifted_question"):
        data["mismatch_verification"] = {"overall": 0.0, "recommendation": "reject"}
        return data
    
    prompt_path = os.path.join(args.prompt_dir, "verify_mismatch.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        rewritten_question=data.get("domain_shifted_question", ""),
        original_domain=data.get("current_domain", "mathematical"),
        ground_truth=data.get("ground_truth", "")
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at verifying domain mismatches.",
        model=args.model,
        temperature=0.0
    )
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Parse verification
    verification = parse_json_response(response, {
        "overall": 0.5,
        "recommendation": "reject"
    })
    
    data["mismatch_verification"] = verification
    
    overall = verification.get("overall", 0)
    rec = verification.get("recommendation", "reject")
    logging.info(f"ID {data['id']}: Mismatch - score={overall:.2f}, {rec}")
    
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
                logging.error(f"Error processing {data['id']}: {e}")
                t.update(1)
                continue
    
    all_data = existing_data + read_jsonl(jsonl_path)[len(existing_data):]
    
    if all_data:
        write_json(output_path, all_data)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
    
    return len(all_data) == total_len

def filter_valid_data(final_path):
    """Filter valid domain-shifted problems"""
    dataset = read_json(final_path)
    valid_data = []
    
    total_prompt = sum(sum(d.get("prompt_lengths", [])) for d in dataset)
    total_completion = sum(sum(d.get("completion_lengths", [])) for d in dataset)
    
    for data in dataset:
        # Check if domain shift was generated
        if not data.get("domain_shifted_question"):
            continue
        
        # Get verifications
        preservation = data.get("preservation_verification", {})
        mismatch = data.get("mismatch_verification", {})
        
        # Strict checks
        numbers_match = preservation.get("numbers_match_automated", False)
        preservation_ok = (preservation.get("overall", 0) >= 0.75 and 
                          preservation.get("recommendation") == "accept" and
                          numbers_match)
        
        mismatch_ok = (mismatch.get("overall", 0) >= 0.70 and 
                      mismatch.get("recommendation") == "accept")
        
        if preservation_ok and mismatch_ok:
            valid_item = {
                "id": f"{data['id']}_domain",
                "data_source": data.get("data_source", ""),
                "difficulty": data.get("difficulty", ""),
                "transformation_type": "domain_shift",
                "original_question": data["question"],
                "ground_truth": data.get("ground_truth", ""),
                "extracted_conditions": data.get("extracted_conditions", []),
                "original_domain": data.get("current_domain", ""),
                "domain_shifted_question": data["domain_shifted_question"],
                "preservation_verification": preservation,
                "mismatch_verification": mismatch,
                "prompt_lengths": data.get("prompt_lengths", []),
                "completion_lengths": data.get("completion_lengths", [])
            }
            valid_data.append(valid_item)
    
    output_path = final_path.replace("_final.json", "_valid.json")
    write_json(output_path, valid_data)
    
    # Statistics
    total_valid_prompt = sum(sum(item["prompt_lengths"]) for item in valid_data)
    total_valid_completion = sum(sum(item["completion_lengths"]) for item in valid_data)
    
    print("\n" + "="*70)
    print("DOMAIN SHIFT DATASET STATISTICS")
    print("="*70)
    print(f"Original problems: {len(dataset)}")
    print(f"Domain shiftable: {sum(1 for d in dataset if d.get('domain_shiftable'))}")
    print(f"Valid domain-shifted problems: {len(valid_data)}")
    print(f"Success rate: {len(valid_data) / len(dataset) * 100:.2f}%")
    
    print(f"\nToken Usage (ALL):")
    print(f"  Prompt: {total_prompt:,}")
    print(f"  Completion: {total_completion:,}")
    print(f"  Total: {total_prompt + total_completion:,}")
    
    print(f"\nToken Usage (VALID):")
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
    input_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    
    if not os.path.exists(input_path):
        logging.error(f"Input not found: {input_path}")
        return
    
    dataset = read_json(input_path)
    
    if args.test_mode:
        dataset = dataset[:2]
        logging.info("TEST MODE: First 2 items")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("DOMAIN SHIFT CONSTRUCTION PIPELINE")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Items: {len(dataset)}")
    print("="*70)
    
    # Step 1: Extract
    print("\n[1/5] Extracting conditions & question")
    extract_path = os.path.join(args.output_dir, f"{args.dataset}_extract.json")
    process_with_jsonl(dataset, extract_path, extract_conditions_and_question, "Extracting")
    
    # Step 2: Identify Domain
    print("\n[2/5] Identifying question domain")
    dataset = read_json(extract_path)
    identify_path = os.path.join(args.output_dir, f"{args.dataset}_identify.json")
    process_with_jsonl(dataset, identify_path, identify_question_domain, "Identifying domain")
    
    # Step 3: Change Domain
    print("\n[3/5] Changing question domain")
    dataset = read_json(identify_path)
    change_path = os.path.join(args.output_dir, f"{args.dataset}_change.json")
    process_with_jsonl(dataset, change_path, change_question_domain, "Changing domain")
    
    # Step 4: Verify Preservation
    print("\n[4/5] Verifying conditions preserved")
    dataset = read_json(change_path)
    verify_pres_path = os.path.join(args.output_dir, f"{args.dataset}_verify_pres.json")
    process_with_jsonl(dataset, verify_pres_path, verify_conditions_preserved, "Verifying preservation")
    
    # Step 5: Verify Mismatch
    print("\n[5/5] Verifying domain mismatch")
    dataset = read_json(verify_pres_path)
    final_path = os.path.join(args.output_dir, f"{args.dataset}_final.json")
    process_with_jsonl(dataset, final_path, verify_domain_mismatch, "Verifying mismatch")
    
    # Filter
    print("\n[6/5] Filtering valid data")
    filter_valid_data(final_path)
    
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()