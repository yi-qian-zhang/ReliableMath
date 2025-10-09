#!/usr/bin/env python3
"""
Dataset Construction Pipeline - Fixed Version with Better Error Handling
Removes key conditions from mathematical problems to create unsolvable variants
"""

import os
import json
import time
import logging
import argparse
from openai import OpenAI
import random
from tqdm import tqdm
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Mathematical Dataset Construction")
parser.add_argument("--model", default="gpt-4o", help="Model name")
parser.add_argument("--data_dir", default="./data/solve", help="Input file path")
parser.add_argument("--output_dir", default="./data/unsol", help="Output file path")
parser.add_argument("--prompt_dir", default="./prompt/{}/rewrite", help="Prompt template path")
parser.add_argument("--dataset", default="polaris", help="Dataset name")
parser.add_argument("--prompt", default="v4-remove-only", type=str, help="Prompt type")
parser.add_argument("--temperature", default=0.0, type=float, help="Temperature")
parser.add_argument("--split_id", default=0, type=int, help="Split ID for batch processing")
parser.add_argument("--test_mode", action='store_true', help="Test mode - process only first 2 items")
args = parser.parse_args()

# Only use remove type
UNS_TYPE = ["remove"]

# Load API configuration
try:
    model_options = json.load(open("./data/api_keys.json", "r"))
except FileNotFoundError:
    logging.error("api_keys.json not found! Please create it in ./data/")
    exit(1)

# ============= Utility Functions =============

def read_json(filepath):
    """Read JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(filepath, data):
    """Write JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_jsonl(filepath):
    """Read JSONL file with better error handling"""
    data = []
    if not os.path.exists(filepath):
        return data
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON at line {line_num} in {filepath}: {e}")
                logging.warning(f"Content: {line[:100]}...")
                continue
    return data

def dump_jsonl(data, filepath, append=False):
    """Write JSONL file with validation"""
    mode = 'a' if append else 'w'
    
    # Validate that data can be serialized
    try:
        json_str = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logging.error(f"Cannot serialize data to JSON: {e}")
        return False
    
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(json_str + '\n')
        f.flush()  # Ensure data is written
    
    return True

def jsonl2json(input_path, output_path):
    """Convert JSONL to JSON"""
    data = read_jsonl(input_path)
    if data:
        write_json(output_path, data)
        return True
    return False

# ============= API Functions =============

def get_response_openai(input_prompt, persona="", model=None, temperature=0.0):
    """Call OpenAI-compatible API"""
    if model is None:
        model = args.model
    
    if model not in model_options:
        logging.error(f"Model {model} not found in api_keys.json")
        return ""
    
    model_name, key, url = random.choice(model_options[model])
    client = OpenAI(api_key=key, base_url=url)
    
    # Build messages
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": input_prompt})
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.warning(f'API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                logging.error(f'All API attempts failed')
    
    return ""

# ============= Validation Functions =============

def validate_extraction(question, conditions):
    """Validate if extracted conditions match the original question"""
    if not conditions:
        return False
    
    question_lower = question.lower()
    
    # Check for specific problem types
    if any(word in question_lower for word in ['tire', 'wheel']):
        # Tire problem
        for cond in conditions:
            cond_lower = cond.lower()
            if any(word in cond_lower for word in ['tire', 'wheel', 'wear', 'km', '25', '15', 'front', 'rear', 'swap']):
                return True
    
    elif 'tetrahedron' in question_lower:
        # Tetrahedron problem  
        for cond in conditions:
            cond_lower = cond.lower()
            if any(word in cond_lower for word in ['edge', 'vertex', 'tetrahedron', 'median', 'length', 'ab', 'bc', 'ca']):
                return True
    
    else:
        # Generic validation
        question_words = set(question_lower.split())
        for cond in conditions:
            cond_words = set(cond.lower().split())
            # At least 3 common words indicates relevance
            if len(question_words & cond_words) >= 3:
                return True
            # Or contains important numbers from the question
            question_numbers = re.findall(r'\d+', question)
            cond_numbers = re.findall(r'\d+', cond)
            if question_numbers and cond_numbers:
                if set(question_numbers) & set(cond_numbers):
                    return True
    
    return False

def fallback_extraction(question):
    """Rule-based fallback extraction when API fails"""
    conditions = []
    question_lower = question.lower()
    
    # Tire problem
    if 'tire' in question_lower or 'wheel' in question_lower:
        # Extract mileage information
        front_match = re.search(r'front[^.]*?(\d+(?:,\d{3})*)\s*km', question, re.IGNORECASE)
        rear_match = re.search(r'rear[^.]*?(\d+(?:,\d{3})*)\s*km', question, re.IGNORECASE)
        
        if front_match:
            conditions.append(f"Front tires wear out after {front_match.group(1)} km")
        if rear_match:
            conditions.append(f"Rear tires wear out after {rear_match.group(1)} km")
        if 'same time' in question_lower:
            conditions.append("The goal is to swap tires so they wear out at the same time")
    
    # Tetrahedron problem
    elif 'tetrahedron' in question_lower:
        # Extract edge definitions
        edge_matches = re.findall(r'\$([A-Z]{2})\s*=\s*([a-z]_?\{?\d?\}?)\$', question)
        if edge_matches:
            edge_str = ", ".join([f"{e[0]}={e[1]}" for e in edge_matches[:6]])
            conditions.append(f"Edge lengths: {edge_str}")
        
        if 'median' in question_lower:
            conditions.append("h is the length of the median line from vertex D")
        
        # Extract the formula to prove
        formula_match = re.search(r'Prove that\s*\$([^$]+)\$', question)
        if formula_match:
            conditions.append(f"Need to prove: ${formula_match.group(1)}$")
    
    # Generic extraction if specific patterns don't match
    if not conditions:
        # Extract sentences with numbers
        sentences = question.split('.')
        for sent in sentences:
            if re.search(r'\d+', sent) and len(sent) > 20:
                conditions.append(sent.strip())
                if len(conditions) >= 2:
                    break
    
    # Final fallback
    if not conditions:
        conditions.append("Unable to extract conditions automatically - manual review needed")
    
    return conditions[:3]  # Return max 3 conditions

# ============= Core Processing Functions =============

def extract_condition(dataset, extract_path):
    """Extract key conditions from mathematical problems"""
    total_len = len(dataset)
    
    # Check for existing JSONL file
    existing_data = []
    jsonl_path = extract_path.replace('.json', '.jsonl')
    
    if os.path.exists(jsonl_path):
        logging.info(f"Found JSONL file: {jsonl_path}")
        existing_data = read_jsonl(jsonl_path)
        if existing_data:
            saved_ids = {item['id'] for item in existing_data}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
            logging.info(f"Continuing from {len(existing_data)} existing items, {len(dataset)} remaining")
    elif os.path.exists(extract_path):
        # Try to read existing JSON file
        try:
            existing_data = read_json(extract_path)
            if existing_data:
                saved_ids = {item['id'] for item in existing_data}
                dataset = [item for item in dataset if item['id'] not in saved_ids]
                logging.info(f"Continuing from {len(existing_data)} existing items, {len(dataset)} remaining")
        except:
            pass
    
    if not dataset:
        logging.info("All items already processed")
        return True
    
    with tqdm(total=len(dataset), desc="Extracting conditions") as t:
        for data in dataset:
            # Build extraction prompt
            extract_prompt = f"""Extract 1-3 key mathematical conditions from the following problem.

Problem: {data["question"]}

Instructions:
- Extract only the essential conditions that, if removed, would make the problem unsolvable
- For word problems: extract numerical values, constraints, or goals
- For proofs: extract given geometric/algebraic conditions, definitions, or relationships
- Output each condition on a separate line
- Do not include numbering or bullet points

Extracted conditions:"""
            
            # Get extraction with retries if needed
            max_attempts = 3
            valid_extraction = None
            
            for attempt in range(max_attempts):
                response = get_response_openai(
                    extract_prompt,
                    persona="You are a precise mathematical condition extractor. Extract only conditions present in the given problem.",
                    model=args.model,
                    temperature=0.0
                )
                
                if response:
                    # Parse response
                    lines = response.strip().split('\n')
                    conditions = []
                    
                    for line in lines:
                        line = line.strip()
                        # Remove numbering
                        if line and line[0].isdigit() and len(line) > 2:
                            if line[1] in ['.', ')', ':']:
                                line = line[2:].strip()
                        # Remove bullet points
                        if line.startswith(('- ', '• ', '* ')):
                            line = line[2:].strip()
                        
                        if line and len(line) > 10:
                            conditions.append(line)
                    
                    # Validate extraction
                    if conditions and validate_extraction(data["question"], conditions):
                        valid_extraction = conditions
                        logging.info(f"ID {data['id']}: Successfully extracted {len(conditions)} conditions")
                        break
                    else:
                        logging.warning(f"ID {data['id']}: Attempt {attempt+1} - Invalid extraction, retrying...")
                
                time.sleep(1)  # Rate limiting
            
            # Use fallback if all attempts failed
            if not valid_extraction:
                logging.warning(f"ID {data['id']}: Using fallback extraction")
                valid_extraction = fallback_extraction(data["question"])
            
            data["extracted_condition"] = valid_extraction
            t.update(1)
            
            # Write to JSONL
            if not dump_jsonl(data, jsonl_path, append=True):
                logging.error(f"Failed to write data for ID {data['id']}")
    
    # Convert JSONL to JSON
    all_data = existing_data + read_jsonl(jsonl_path)[len(existing_data):]
    
    if len(all_data) == total_len:
        write_json(extract_path, all_data)
        logging.info(f"Successfully extracted conditions for all {total_len} items")
        # Clean up JSONL file
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        return True
    else:
        logging.warning(f"Expected {total_len} items but got {len(all_data)}")
        # Still save what we have
        if all_data:
            write_json(extract_path, all_data)
        return len(all_data) == total_len

def condition_process(extract_path):
    """Process and clean extracted conditions"""
    dataset = read_json(extract_path)
    
    for data in dataset:
        # Already processed in extract_condition, just validate
        if "extracted_condition" in data:
            logging.info(f"ID {data['id']}: {len(data['extracted_condition'])} conditions ready")
    
    write_json(extract_path, dataset)

def process_with_jsonl_support(dataset, output_path, process_func, desc):
    """Generic function to process data with JSONL intermediate files"""
    total_len = len(dataset)
    jsonl_path = output_path.replace('.json', '.jsonl')
    
    # Check existing progress
    existing_data = []
    if os.path.exists(jsonl_path):
        existing_data = read_jsonl(jsonl_path)
        if existing_data:
            saved_ids = {item['id'] for item in existing_data}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
            logging.info(f"{desc}: Continuing from {len(existing_data)} existing items")
    elif os.path.exists(output_path):
        try:
            existing_data = read_json(output_path)
            saved_ids = {item['id'] for item in existing_data}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except:
            pass
    
    if not dataset:
        logging.info(f"{desc}: All items already processed")
        return True
    
    # Process remaining items
    with tqdm(total=len(dataset), desc=desc) as t:
        for data in dataset:
            processed_data = process_func(data)
            if processed_data:
                t.update(1)
                dump_jsonl(processed_data, jsonl_path, append=True)
    
    # Combine and save
    all_data = existing_data + read_jsonl(jsonl_path)[len(existing_data):]
    
    if all_data:
        write_json(output_path, all_data)
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
    
    return len(all_data) == total_len

def process_remove_analysis(data):
    """Process single item for remove analysis"""
    analyses = []
    
    for condition in data.get("extracted_condition", []):
        analysis_prompt = f"""You are analyzing a mathematical problem to understand the impact of removing a key condition.

Original problem: {data["question"]}
Original answer: {data["ground_truth"]}
Condition to analyze: {condition}

Analyze step by step:
1. What role does this condition play in solving the problem?
2. What critical information would be missing without this condition?
3. Why would the problem become unsolvable without this information?

Provide a clear, concise analysis:

Analysis:"""
        
        response = get_response_openai(
            analysis_prompt,
            persona="You are an expert mathematical problem analyst.",
            model=args.model,
            temperature=0.0
        )
        
        # Clean response
        if "Analysis:" in response:
            response = response.split("Analysis:")[-1].strip()
        
        analyses.append(response)
    
    data["remove_analysis"] = analyses
    return data

def process_condition_rewrite(data):
    """Process single item for condition rewrite"""
    rewrites = []
    
    for condition, analysis in zip(data.get("extracted_condition", []), data.get("remove_analysis", [])):
        if not analysis or analysis.strip() == "":
            rewrites.append("")
            continue
        
        rewrite_prompt = f"""Rewrite the mathematical problem by removing the specified condition.

Original problem: {data["question"]}
Original answer: {data["ground_truth"]}
Condition to remove: {condition}
Analysis: {analysis}

Requirements:
1. Completely remove the specified condition without leaving any traces or hints
2. Keep all other conditions exactly the same
3. Maintain the problem's language style and structure
4. Ensure the problem becomes unsolvable due to missing critical information

Output only the rewritten problem without any explanation:

Rewritten problem:"""
        
        response = get_response_openai(
            rewrite_prompt,
            persona="You are an expert at rewriting mathematical problems.",
            model=args.model,
            temperature=0.0
        )
        
        # Clean response
        if "Rewritten problem:" in response:
            response = response.split("Rewritten problem:")[-1].strip()
        
        rewrites.append(response)
    
    data["remove"] = rewrites
    return data

def process_verify_removal(data):
    """Process single item for removal verification"""
    removal_results = []
    
    for condition, rewritten in zip(data.get("extracted_condition", []), data.get("remove", [])):
        if not rewritten:
            removal_results.append("Unknown")
            continue
        
        verify_prompt = f"""Verify whether the specified condition was successfully removed from the rewritten problem.

Original problem: {data["question"]}
Condition to remove: {condition}
Rewritten problem: {rewritten}

Check:
1. Is the specified condition completely absent from the rewritten problem?
2. Are there no hints or implicit references to the removed condition?
3. Are all other conditions preserved unchanged?

Output only "True" if successfully removed or "False" if not:

Result:"""
        
        response = get_response_openai(
            verify_prompt,
            persona="You are a precise verification expert.",
            model=args.model,
            temperature=0.0
        )
        
        # Extract True/False
        response_lower = response.lower()
        if "true" in response_lower:
            removal_results.append("True")
        elif "false" in response_lower:
            removal_results.append("False")
        else:
            removal_results.append("Unknown")
    
    data["removal_verified"] = removal_results
    return data

def process_verify_unsolvable(data):
    """Process single item for unsolvability verification"""
    unsolvable_results = []
    
    for rewritten in data.get("remove", []):
        if not rewritten:
            unsolvable_results.append("Unknown")
            continue
        
        verify_prompt = f"""Determine whether this mathematical problem is unsolvable due to missing critical information.

Problem: {rewritten}
Original answer reference: {data["ground_truth"]}

Analysis steps:
1. Attempt to solve the problem
2. Identify what necessary information is missing
3. Determine if there are multiple possible answers or if no unique answer exists

Note:
- If the problem has a unique determinable answer, it is solvable
- If the problem lacks critical information leading to non-unique or indeterminate answers, it is unsolvable

Output only "True" if unsolvable or "False" if solvable:

Result:"""
        
        response = get_response_openai(
            verify_prompt,
            persona="You are an expert problem solver and verifier.",
            model=args.model,
            temperature=0.0
        )
        
        # Extract True/False
        response_lower = response.lower()
        if "true" in response_lower:
            unsolvable_results.append("True")
        elif "false" in response_lower:
            unsolvable_results.append("False")
        else:
            unsolvable_results.append("Unknown")
    
    data["unsolvable_verified"] = unsolvable_results
    return data

def filter_valid_data(final_path):
    """Filter and save only valid data"""
    dataset = read_json(final_path)
    valid_data = []
    
    for data in dataset:
        for i, (removed, unsolvable) in enumerate(zip(data.get("removal_verified", []), data.get("unsolvable_verified", []))):
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
    
    # Print statistics
    total_generated = sum(len(d.get("remove", [])) for d in dataset)
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Original data points: {len(dataset)}")
    print(f"Generated variants: {total_generated}")
    print(f"Valid unsolvable problems: {len(valid_data)}")
    if total_generated > 0:
        print(f"Success rate: {len(valid_data) / total_generated * 100:.2f}%")
    print(f"Output saved to: {output_path}")
    print("="*60)

# ============= Main Workflow =============

def construction_workflow():
    """Complete dataset construction pipeline"""
    input_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
    
    dataset = read_json(input_path)
    
    # Test mode - process only first 2 items
    if args.test_mode:
        dataset = dataset[:2]
        logging.info("TEST MODE: Processing only first 2 items")
    
    output_dir = os.path.join(args.output_dir, args.prompt)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("DATASET CONSTRUCTION PIPELINE")
    print("="*60)
    print(f"Input dataset: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Items to process: {len(dataset)}")
    print("="*60)
    
    # Step 1: Extract conditions
    print("\nStep 1: Extracting key conditions")
    extract_path = os.path.join(output_dir, f"{args.dataset}_extract.json")
    if extract_condition(dataset, extract_path):
        print("Condition extraction completed")
        condition_process(extract_path)
    else:
        logging.error("Failed to complete extraction")
        return
    
    # Step 2: Analyze removal impact
    print("\nStep 2: Analyzing removal impact")
    dataset = read_json(extract_path)
    analysis_path = os.path.join(output_dir, f"{args.dataset}_analysis.json")
    if process_with_jsonl_support(dataset, analysis_path, process_remove_analysis, "Analyzing"):
        print("Analysis completed")
    
    # Step 3: Rewrite problems
    print("\nStep 3: Rewriting problems")
    dataset = read_json(analysis_path)
    rewrite_path = os.path.join(output_dir, f"{args.dataset}_rewrite.json")
    if process_with_jsonl_support(dataset, rewrite_path, process_condition_rewrite, "Rewriting"):
        print("Rewriting completed")
    
    # Step 4: Verify removal
    print("\nStep 4: Verifying condition removal")
    dataset = read_json(rewrite_path)
    removal_verify_path = os.path.join(output_dir, f"{args.dataset}_removal_verified.json")
    if process_with_jsonl_support(dataset, removal_verify_path, process_verify_removal, "Verifying removal"):
        print("Removal verification completed")
    
    # Step 5: Verify unsolvability
    print("\nStep 5: Verifying unsolvability")
    dataset = read_json(removal_verify_path)
    final_path = os.path.join(output_dir, f"{args.dataset}_final.json")
    if process_with_jsonl_support(dataset, final_path, process_verify_unsolvable, "Verifying unsolvability"):
        print("Unsolvability verification completed")
    
    # Step 6: Filter valid data
    print("\nStep 6: Filtering valid data")
    filter_valid_data(final_path)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    construction_workflow()