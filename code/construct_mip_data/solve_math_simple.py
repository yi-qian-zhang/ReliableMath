import json
from openai import OpenAI
from tqdm import tqdm
import time
import re

# API configuration / API配置
API_KEY = "sk-HLA9kBx3vUpL6QVJE17b313aD1B34e53B9E660E932620a43"
BASE_URL = "https://api.ai-gaochao.cn/v1"
MODEL = "gpt-4o"

# File paths / 文件路径
INPUT_FILE = "/data/home/zyq/ReliableMath/data/solve/polaris_easy_20.json"
OUTPUT_FILE = "/data/home/zyq/ReliableMath/data/solve/polaris_easy_20_output.json"

# Initialize client / 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def extract_boxed_answer(text):
    """Extract answer from \\boxed{} with nested braces support / 从\\boxed{}中提取答案，支持嵌套大括号"""
    # Find \boxed{ position
    start_pattern = r'\\boxed\{'
    match = re.search(start_pattern, text)
    if not match:
        return ""
    
    # Start from the opening brace after \boxed
    start_pos = match.end() - 1  # Position of the opening {
    brace_count = 0
    pos = start_pos
    
    # Count braces to find the matching closing brace
    while pos < len(text):
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Found the matching closing brace
                return text[start_pos + 1:pos]
        pos += 1
    
    return ""

def call_gpt4o(question, max_retries=3):
    """Call GPT-4o API with retry mechanism"""
    prompt = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=8192
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        
        except Exception as e:
            print(f"\nAttempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait 2 seconds before retry / 等待2秒后重试
            else:
                return f"Error: {str(e)}"

# Load questions / 读取问题
print(f"Loading file: {INPUT_FILE}")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    questions = json.load(f)
print(f"Loaded {len(questions)} questions\n")

results = []

# Process each question / 处理每个问题
for item in tqdm(questions, desc="Processing"):
    question_id = item['id']
    question = item['question']
    ground_truth = item['ground_truth']
    
    # Call GPT-4o / 调用GPT-4o
    gpt_answer = call_gpt4o(question)
    
    # Extract boxed answer / 提取boxed答案
    boxed_answer = extract_boxed_answer(gpt_answer)
    
    # Save result / 保存结果
    results.append({
        'id': question_id,
        'question': question,
        'ground_truth': ground_truth,
        'gpt4o_answer': gpt_answer,
        'box': boxed_answer,
        'data_source': item.get('data_source', ''),
        'difficulty': item.get('difficulty', '')
    })
    
    # Sleep to avoid rate limiting / 暂停避免请求过快
    time.sleep(0.5)

# Save results / 保存结果
print(f"\nSaving results to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nCompleted! Processed {len(results)} questions")
print(f"Results saved to: {OUTPUT_FILE}")

# Statistics / 统计信息
successful = sum(1 for r in results if not r['gpt4o_answer'].startswith('Error'))
extracted = sum(1 for r in results if r['box'])
print(f"\nStatistics:")
print(f"  Success: {successful}/{len(results)}")
print(f"  Extracted: {extracted}/{len(results)}")
if successful < len(results):
    print(f"  Failed: {len(results) - successful}/{len(results)}")