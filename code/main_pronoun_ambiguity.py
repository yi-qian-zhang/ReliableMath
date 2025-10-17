#!/usr/bin/env python3
"""
Pronoun Ambiguity Construction Pipeline
Type 3: 指代模糊（使用模糊代词或不明确的指代）
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
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Pronoun Ambiguity Dataset Construction")
parser.add_argument("--model", default="gpt-4o", help="Model name")
parser.add_argument("--data_dir", default="./data/solve", help="Input directory")
parser.add_argument("--output_dir", default="./data/pronoun_ambiguity", help="Output directory")
parser.add_argument("--prompt_dir", default="./prompt/pronoun_ambiguity", help="Prompt directory")
parser.add_argument("--dataset", default="polaris", help="Dataset name")
parser.add_argument("--temperature", default=0.0, type=float, help="Temperature")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 2 items")
parser.add_argument("--force", action='store_true', help="Force reprocess all items, delete existing intermediate files")
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
        data["extracted_conditions"] = []
        data["extracted_question"] = ""
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
    
    logging.info(f"ID {data['id']}: Extracted {len(data['extracted_conditions'])} conditions")
    
    return data

# ============= Step 2: Identify Entities =============

def identify_entities(data):
    """Step 2: Identify entities that can be ambiguated with pronouns"""
    prompt_path = os.path.join(args.prompt_dir, "identify_entities.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["identified_entities"] = []
        data["pronoun_applicable"] = False
        data["ambiguatable_entities"] = []
        data["suggested_pronouns"] = []
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
        persona="You are an expert at identifying entities in problems.",
        model=args.model,
        temperature=0.0
    )
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Parse entity identification
    entity_info = parse_json_response(response, {
        "entities": [],
        "pronoun_applicable": False
    })
    
    data["identified_entities"] = entity_info.get("entities", [])
    data["pronoun_applicable"] = entity_info.get("pronoun_applicable", False)
    data["ambiguatable_entities"] = entity_info.get("ambiguatable_entities", [])
    data["suggested_pronouns"] = entity_info.get("suggested_pronouns", [])
    
    logging.info(f"ID {data['id']}: Found {len(data['identified_entities'])} entities, Applicable={data['pronoun_applicable']}")
    
    return data

# ============= Step 3: Replace with Pronoun (FIXED) =============

def replace_with_pronoun(data):
    """Step 3: Replace specific entity reference with ambiguous pronoun"""
    
    if not data.get("pronoun_applicable", False):
        logging.info(f"ID {data['id']}: Skipping - not pronoun applicable")
        data["pronoun_ambiguous_question"] = ""
        return data
    
    prompt_path = os.path.join(args.prompt_dir, "replace_pronoun.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["pronoun_ambiguous_question"] = ""
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        original_question=data["question"],
        conditions=json.dumps(data.get("extracted_conditions", []), indent=2),
        question=data.get("extracted_question", ""),
        entities=json.dumps(data.get("identified_entities", []), indent=2),
        suggested_pronouns=json.dumps(data.get("suggested_pronouns", []))
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at creating pronoun ambiguity.",
        model=args.model,
        temperature=0.0
    )
    
    data["prompt_lengths"].append(prompt_tokens)
    data["completion_lengths"].append(completion_tokens)
    
    # Enhanced cleaning logic
    response = response.strip()
    
    # Remove common prefixes and labels
    prefixes_to_remove = [
        "Rewritten Problem:",
        "Rewritten Question:",
        "Rewritten:",
        "Problem:",
        "Question:",
        "**Rewritten Problem:**",
        "**Rewritten Question:**",
        "**Rewritten:**",
    ]
    
    for prefix in prefixes_to_remove:
        if prefix in response:
            # Take everything after the last occurrence of the prefix
            response = response.split(prefix)[-1].strip()
    
    # Remove markdown formatting
    response = response.replace("**", "").strip()
    
    # Remove leading/trailing quotes
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()
    if response.startswith("'") and response.endswith("'"):
        response = response[1:-1].strip()
    
    # Remove "Conditions: [...]" block if present at the start
    if response.lower().startswith("conditions:"):
        lines = response.split('\n')
        # Find where the actual problem starts (after the conditions list)
        in_conditions = True
        result_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip conditions block
            if in_conditions:
                if stripped and not any(stripped.startswith(x) for x in ['Conditions:', '[', ']', '"', '-', 'Rewritten']):
                    in_conditions = False
                    result_lines.append(line)
            else:
                result_lines.append(line)
        
        if result_lines:
            response = '\n'.join(result_lines).strip()
    
    # Final cleanup: remove any remaining "Rewritten Question:" that might be in the middle
    response = response.replace("Rewritten Question:", "").strip()
    
    data["pronoun_ambiguous_question"] = response
    
    logging.info(f"ID {data['id']}: Pronoun ambiguity created")
    
    return data

# ============= Step 4: Verify Preservation =============

def verify_conditions_preserved(data):
    """Step 4: Verify conditions are preserved"""
    
    if not data.get("pronoun_ambiguous_question"):
        data["preservation_verification"] = {"overall": 0.0, "recommendation": "reject"}
        return data
    
    prompt_path = os.path.join(args.prompt_dir, "verify_preservation.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["preservation_verification"] = {"overall": 0.0, "recommendation": "reject"}
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        original_question=data["question"],
        original_conditions=json.dumps(data.get("extracted_conditions", []), indent=2),
        original_question_part=data.get("extracted_question", ""),
        rewritten_question=data.get("pronoun_ambiguous_question", "")
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
    
    data["preservation_verification"] = verification
    
    overall = verification.get("overall", 0)
    rec = verification.get("recommendation", "reject")
    logging.info(f"ID {data['id']}: Preservation - score={overall:.2f}, {rec}")
    
    return data

# ============= Step 5: Verify Ambiguity =============

def verify_pronoun_ambiguity(data):
    """Step 5: Verify pronoun creates genuine ambiguity"""
    
    if not data.get("pronoun_ambiguous_question"):
        data["ambiguity_verification"] = {"overall": 0.0, "recommendation": "reject"}
        return data
    
    prompt_path = os.path.join(args.prompt_dir, "verify_ambiguity.txt")
    
    if not os.path.exists(prompt_path):
        logging.error(f"Prompt file not found: {prompt_path}")
        data["ambiguity_verification"] = {"overall": 0.0, "recommendation": "reject"}
        return data
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    input_prompt = prompt_template.format(
        rewritten_question=data.get("pronoun_ambiguous_question", ""),
        entities=json.dumps(data.get("identified_entities", []), indent=2),
        ground_truth=data.get("ground_truth", "")
    )
    
    response, prompt_tokens, completion_tokens = get_response_openai(
        input_prompt,
        persona="You are an expert at verifying pronoun ambiguity.",
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
    
    data["ambiguity_verification"] = verification
    
    overall = verification.get("overall", 0)
    rec = verification.get("recommendation", "reject")
    logging.info(f"ID {data['id']}: Ambiguity - score={overall:.2f}, {rec}")
    
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
    """Filter valid pronoun-ambiguous problems (FIXED)"""
    dataset = read_json(final_path)
    valid_data = []
    
    total_prompt = sum(sum(d.get("prompt_lengths", [])) for d in dataset)
    total_completion = sum(sum(d.get("completion_lengths", [])) for d in dataset)
    
    for data in dataset:
        if not data.get("pronoun_ambiguous_question"):
            continue
        
        preservation = data.get("preservation_verification", {})
        ambiguity = data.get("ambiguity_verification", {})
        
        preservation_score = preservation.get("overall", 0)
        ambiguity_score = ambiguity.get("overall", 0)
        
        # FIXED: Only check scores, ignore recommendation
        # This allows cases where the model thinks ambiguity is "too ambiguous" 
        # but that's actually what we want
        preservation_ok = preservation_score >= 0.75
        ambiguity_ok = ambiguity_score >= 0.70
        
        if preservation_ok and ambiguity_ok:
            valid_item = {
                "id": f"{data['id']}_pronoun",
                "data_source": data.get("data_source", ""),
                "difficulty": data.get("difficulty", ""),
                "transformation_type": "pronoun_ambiguity",
                "original_question": data["question"],
                "ground_truth": data.get("ground_truth", ""),
                "extracted_conditions": data.get("extracted_conditions", []),
                "identified_entities": data.get("identified_entities", []),
                "pronoun_ambiguous_question": data["pronoun_ambiguous_question"],
                "preservation_verification": preservation,
                "ambiguity_verification": ambiguity,
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
    print("PRONOUN AMBIGUITY DATASET STATISTICS")
    print("="*70)
    print(f"Original problems: {len(dataset)}")
    print(f"Pronoun applicable: {sum(1 for d in dataset if d.get('pronoun_applicable'))}")
    print(f"Valid pronoun-ambiguous problems: {len(valid_data)}")
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
    
    # Force cleanup if --force flag is set
    if args.force:
        logging.info("Force mode: Cleaning up existing intermediate files...")
        
        # Remove all JSON files for this dataset
        json_pattern = os.path.join(args.output_dir, f"{args.dataset}_*.json")
        for file in glob.glob(json_pattern):
            try:
                os.remove(file)
                logging.info(f"Removed: {file}")
            except Exception as e:
                logging.warning(f"Could not remove {file}: {e}")
        
        # Remove all JSONL files for this dataset
        jsonl_pattern = os.path.join(args.output_dir, f"{args.dataset}_*.jsonl")
        for file in glob.glob(jsonl_pattern):
            try:
                os.remove(file)
                logging.info(f"Removed: {file}")
            except Exception as e:
                logging.warning(f"Could not remove {file}: {e}")
        
        logging.info("Cleanup completed.")
    
    print("="*70)
    print("PRONOUN AMBIGUITY CONSTRUCTION PIPELINE")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Items: {len(dataset)}")
    if args.force:
        print(f"Mode: FORCE (reprocessing all)")
    print("="*70)
    
    # Step 1: Extract
    print("\n[1/5] Extracting conditions & question")
    extract_path = os.path.join(args.output_dir, f"{args.dataset}_extract.json")
    process_with_jsonl(dataset, extract_path, extract_conditions_and_question, "Extracting")
    
    # Step 2: Identify Entities
    print("\n[2/5] Identifying entities")
    dataset = read_json(extract_path)
    identify_path = os.path.join(args.output_dir, f"{args.dataset}_identify.json")
    process_with_jsonl(dataset, identify_path, identify_entities, "Identifying entities")
    
    # Step 3: Replace with Pronoun
    print("\n[3/5] Replacing with pronouns")
    dataset = read_json(identify_path)
    replace_path = os.path.join(args.output_dir, f"{args.dataset}_replace.json")
    process_with_jsonl(dataset, replace_path, replace_with_pronoun, "Replacing")
    
    # Step 4: Verify Preservation
    print("\n[4/5] Verifying conditions preserved")
    dataset = read_json(replace_path)
    verify_pres_path = os.path.join(args.output_dir, f"{args.dataset}_verify_pres.json")
    process_with_jsonl(dataset, verify_pres_path, verify_conditions_preserved, "Verifying preservation")
    
    # Step 5: Verify Ambiguity
    print("\n[5/5] Verifying pronoun ambiguity")
    dataset = read_json(verify_pres_path)
    final_path = os.path.join(args.output_dir, f"{args.dataset}_final.json")
    process_with_jsonl(dataset, final_path, verify_pronoun_ambiguity, "Verifying ambiguity")
    
    # Filter
    print("\n[6/5] Filtering valid data")
    filter_valid_data(final_path)
    
    print("\n✓ Pipeline completed!")

if __name__ == "__main__":
    construction_workflow()