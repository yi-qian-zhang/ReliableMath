# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import os
import json
import csv
import numpy as np
from tabulate import tabulate
import argparse
from openai import OpenAI
from utils import *
from datetime import datetime
import tiktoken
from transformers import AutoTokenizer

from metrics.rewards.math_reward import deepscaler_reward_fn, realmath_reward_fn

parser = argparse.ArgumentParser(description="Hallucination Generation")

parser.add_argument("--model", default="deepseek_r1", help="model name")
parser.add_argument("--data_dir", default="./data", help="data file path")
parser.add_argument("--output_dir", default="./exp/{}_T{}_{}", help="output file path")
parser.add_argument("--dataset", default="math", help="dataset name")
parser.add_argument("--solve", default="solve", type=str, help="unsolve type")
parser.add_argument("--prompt", default=0, type=int, help="using reliable instruction or not")
parser.add_argument("--task", default="solve", type=str, help="prompt type, instruction or few-shot")
parser.add_argument("--temperature", default=0.0, type=float, help="temperature")
parser.add_argument("--split_id", default=0, type=int, help="split id")
args = parser.parse_args()


def evaluate(input_dir, dataset, model, task, data_dir):
    input_path = os.path.join(input_dir, f"{dataset}.json")
    try:
        data_pool = read_json(input_path)
    except json.JSONDecodeError:
        data_pool = read_jsonl(input_path)

    data_path = os.path.join(data_dir, f"{task}/{dataset}.json")

    # assert len(read_json(data_path)) == len(data_pool), f"Data length mismatch: {len(read_json(data_path))} vs {len(data_pool)}"
    passes = 0

    # import pdb; pdb.set_trace()
    total = len(data_pool)
    total_scores = []

    if model in ["deepseek_r1_0528", "deepseek_r1", "deepseek_v3", "distill-32b", "distill-14b", "qwen-32b", "qwen-1.5b", "qwen-7b", "distill-1.5b", "distill-7b", "qwen3-235b", "qwen3-32b", "qwen3-14b"]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[model])
    elif model in ["gpt-4o", "o3-mini"]:
        encoding = tiktoken.encoding_for_model(tokenizer_dict[model])

    reliability_judge = {"correct": 0, "incorrect": 0, "unmatched": 0, "unknown": 0, "unsolvable": 0}
    token_count = 0

    for data in data_pool:
        # import pdb; pdb.set_trace()
        if model in ["qwen-1.5b", "qwen-7b", "distill-1.5b", "distill-7b"]:
            if isinstance(data["generation"], str):
                response_lst = [data["generation"]]
                reasoning_lst = ["" for _ in range(len(response_lst))]
            elif isinstance(data["generation"], list):
                response_lst = data["generation"]
                reasoning_lst = ["" for _ in range(len(response_lst))]
        elif model in ["deepseek_r1_0528", "gpt-4o", "o3-mini", "deepseek_r1", "deepseek_v3", "distill-32b", "distill-14b", "qwen-32b", "qwen3-235b", "qwen3-32b", "qwen3-14b"]:
            response_lst = [generation["answer"] for generation in data["generation"]]
            reasoning_lst = [generation["reasoning"] if generation["reasoning"] is not None else "" for generation in data["generation"]]

        # import pdb; pdb.set_trace()
        # select reward score based on data_source
        ground_truth = data["ground_truth"]
        score_lst = []
        for reasoning, response in zip(reasoning_lst, response_lst):
            score, judge, extract_answer = realmath_reward_fn(response, ground_truth, task)
            reliability_judge[judge] += 1
            score_lst.append(score)
            if model in ["deepseek_r1_0528", "deepseek_r1", "deepseek_v3", "distill-32b", "distill-14b", "qwen-32b", "qwen-1.5b", "qwen-7b", "distill-1.5b", "distill-7b", "qwen3-235b", "qwen3-32b", "qwen3-14b"]:
                tokens = tokenizer.encode(reasoning + response)
            elif model in ["gpt-4o"]:
                tokens = encoding.encode(reasoning + response)
            elif model in ["o3-mini"]:
                tokens = encoding.encode(response) if reasoning is not None else range(reasoning)
            else:
                tokens = []
            token_count += len(tokens)

        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        n_samples = len(response_lst)
        if max_score == 1:
            passes += 1
        data["judge"] = judge
        data["extract_answer"] = extract_answer

    write_jsonl(input_path, data_pool)

    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)
    avg_length = token_count / total

    # Save metrics to CSV
    score_path = os.path.join(input_dir, "pass.csv")
    
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    assert args.task in ["solve", "unsol"], f"Invalid task: {args.task}"

    if args.task == "solve":
        success = reliability_judge["correct"]
        refusal = reliability_judge["unknown"]
        failed = (reliability_judge["incorrect"] + reliability_judge["unmatched"] + reliability_judge["unsolvable"])
    elif args.task == "unsol":
        success = reliability_judge["unsolvable"]
        refusal = reliability_judge["unknown"]
        failed = (reliability_judge["correct"] + reliability_judge["incorrect"] + reliability_judge["unmatched"])

    precision = success / (success + failed + refusal) if (success + failed + refusal) > 0 else 0
    prudence = refusal / (success + failed + refusal) if (success + failed + refusal) > 0 else 0

    row_data = {
        'output_path': score_path,
        'dataset': dataset,
        'timestamp': formatted_time,
        'correct': reliability_judge["correct"],
        'incorrect': reliability_judge["incorrect"],
        'unmatched': reliability_judge["unmatched"],
        'unknown': reliability_judge["unknown"],
        'unsolvable': reliability_judge["unsolvable"],
        'accuracy': round(pass_at_1, 4),
        "success": success,
        "refusal": refusal,
        "failed": failed,
        'length': int(avg_length),
        "precision": round(precision, 4),
        "prudence": round(prudence, 4)
    }

    # Check if file exists
    file_exists = os.path.isfile(score_path)
    
    # Write to CSV
    with open(score_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]

    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))


if __name__ == '__main__':
    # Load the dataset
    input_dir = os.path.join(args.output_dir.format(args.model, args.temperature, prompts[args.prompt]), args.task)
    evaluate(input_dir, args.dataset, args.model, args.task, args.data_dir)
