#!/usr/bin/env python3
"""
Translate sample_valid.json to Chinese using local vLLM model
"""
import json
import argparse
import logging
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Translate sample_valid.json from English to Chinese")
parser.add_argument("--input", required=True, help="Input sample_valid.json file path")
parser.add_argument("--output", required=True, help="Output translated json file path")
parser.add_argument("--model_url", default="http://localhost:8716/v1", help="Local vLLM model URL")
parser.add_argument("--model_name", default="Qwen3-32B", help="Model name")
parser.add_argument("--threads", default=8, type=int, help="Number of parallel threads for translation")
args = parser.parse_args()

# 初始化 OpenAI 客户端
client = OpenAI(api_key="EMPTY", base_url=args.model_url)

# 翻译 prompt（英文 -> 中文）
TRANSLATION_PROMPT = """你是一位专业的翻译专家。请将下面的数学问题从英文翻译成中文。

要求：
1. 保持数学符号、数字和公式不变
2. 准确流畅地翻译自然语言
3. 保留原文的含义和语境
4. 只输出翻译后的文本，不要有任何解释

待翻译文本：
{text}

翻译："""

def translate_text(text):
    """翻译单个文本（英文->中文）"""
    if not text or not isinstance(text, str):
        return text

    prompt = TRANSLATION_PROMPT.format(text=text)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048
            )
            translation = completion.choices[0].message.content.strip()
            return translation
        except Exception as e:
            logging.warning(f"Translation failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

    logging.error(f"Failed to translate: {text[:100]}...")
    return text  # 返回原文

def translate_list(text_list):
    """翻译列表中的所有文本（英文->中文）"""
    if not text_list or not isinstance(text_list, list):
        return text_list

    translated = []
    for text in text_list:
        if isinstance(text, str):
            translated.append(translate_text(text))
        else:
            translated.append(text)
    return translated

def translate_item(item):
    """翻译单个数据项（英文->中文）

    翻译字段：
    - original_question
    - incomplete_question
    - all_extracted_conditions (列表)
    - removed_conditions (列表)
    - remaining_conditions (列表)

    其他字段（如 ground_truth, id, difficulty 等）保持不变
    """
    translated_item = item.copy()

    # 需要翻译的单个文本字段
    fields_to_translate = [
        "original_question",
        "incomplete_question"
    ]

    # 需要翻译的列表字段
    list_fields_to_translate = [
        "all_extracted_conditions",
        "removed_conditions",
        "remaining_conditions"
    ]

    # 翻译单个字段
    for field in fields_to_translate:
        if field in translated_item and translated_item[field]:
            logging.debug(f"Translating {field}...")
            translated_item[field] = translate_text(translated_item[field])

    # 翻译列表字段
    for field in list_fields_to_translate:
        if field in translated_item and translated_item[field]:
            logging.debug(f"Translating {field} list...")
            translated_item[field] = translate_list(translated_item[field])

    return translated_item

def main():
    logging.info(f"Loading input file: {args.input}")

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input}")
        return
    except Exception as e:
        logging.error(f"Failed to load input file: {e}")
        return

    if not isinstance(data, list):
        logging.error("Input data is not a list")
        return

    logging.info(f"Loaded {len(data)} items")
    logging.info(f"Target language: Chinese (中文)")
    logging.info(f"Model: {args.model_name} at {args.model_url}")
    logging.info(f"Threads: {args.threads}")

    # 测试连接
    try:
        test_translation = translate_text("Test")
        logging.info(f"✓ Connection test successful: 'Test' -> '{test_translation}'")
    except Exception as e:
        logging.error(f"Failed to connect to model: {e}")
        return

    # 多线程翻译所有数据
    logging.info(f"Starting parallel translation with {args.threads} threads...")

    def translate_with_index(idx_item):
        """翻译单个item并返回索引，保持顺序"""
        idx, item = idx_item
        try:
            translated_item = translate_item(item)
            return idx, translated_item
        except Exception as e:
            logging.error(f"Failed to translate item {idx}: {e}")
            return idx, item  # 失败时返回原数据

    # 创建 (index, item) 对
    indexed_data = list(enumerate(data))

    # 使用线程池并行翻译
    translated_results = [None] * len(data)  # 预分配列表保持顺序

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 提交所有任务
        futures = {executor.submit(translate_with_index, idx_item): idx_item[0]
                   for idx_item in indexed_data}

        # 使用 tqdm 显示进度
        with tqdm(total=len(data), desc="Translating to Chinese") as pbar:
            for future in as_completed(futures):
                try:
                    idx, translated_item = future.result()
                    translated_results[idx] = translated_item
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Translation task failed: {e}")
                    pbar.update(1)

    # 过滤掉 None（如果有的话）
    translated_data = [item for item in translated_results if item is not None]

    # 保存结果
    logging.info(f"Saving translated data to: {args.output}")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        logging.info(f"✓ Translation completed! Output: {args.output}")
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")

if __name__ == "__main__":
    main()
