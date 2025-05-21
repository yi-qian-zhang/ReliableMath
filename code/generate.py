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
Generate responses given a dataset of prompts
"""
import csv
import numpy as np
import os
import logging
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

from utils import *
from evaluate import evaluate

import random
import openai
import time
import argparse
from openai import OpenAI

import requests


parser = argparse.ArgumentParser(description="Hallucination Generation")

parser.add_argument("--model", default="deepseek_r1", help="model name")
parser.add_argument("--data_dir", default="./data", help="data file path")
parser.add_argument("--prompt_path", default="./prompt/math_instruction.json", help="instruction file path")
parser.add_argument("--output_dir", default="./exp/{}_T{}_{}/{}", help="output file path")
parser.add_argument("--dataset", default="train", help="dataset name")
parser.add_argument("--num_samples", default=1, type=int, help="number of samples to generate")
parser.add_argument("--task", default="solve", type=str, help="prompt type, instruction or few-shot")
parser.add_argument("--prompt", default=0, type=int, help="using reliable instruction or not")
parser.add_argument("--temperature", default=0.0, type=float, help="temperature")
parser.add_argument("--split_id", default=0, type=int, help="split id")
args = parser.parse_args()


model_options = json.load(open("./data/api_keys.json", "r")) # format: {model_name: [[model_name, key, url]]}
# model_options = {option: 0 for option in model_options}

# print(deepseek_options.keys())

def get_response_siliconflow(question, instruction):
    model_name, key, url = random.choice(model_options[args.model])
    max_len = 16384 if "distill" in args.model else 4096
    # import pdb; pdb.set_trace()
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": instruction["role"]}, 
            {"role": "user", "content": question + " " + instruction[prompts[args.prompt]]}
        ],
        "stream": False,
        "max_tokens": max_len,
        "stop": None,
        "temperature": 0.0,
        "frequency_penalty": 0.0,
        "response_format": {"type": "text"}
    }

    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers).json()
    # import pdb; pdb.set_trace()

    if "distill" in args.model:
        reasoning = response["choices"][0]["message"]["reasoning_content"]
    else:
        reasoning = None
    answer = response["choices"][0]["message"]["content"]
    # response = THOUGHT_DELIMITER_START + reasoning + THOUGHT_DELIMITER_END + answer
    response = {
        "reasoning": reasoning,
        "answer": answer
    }

    return response


def get_response_openai(question, instruction):
    # import pdb; pdb.set_trace()
    model_name, key, url = random.choice(model_options[args.model])
    client = OpenAI(api_key=key, base_url=url)

    # if args.model == "o3-mini":
    #     import pdb; pdb.set_trace()

    message = [
        {"role": "system", "content": instruction["role"]}, 
        {"role": "user", "content": question + " " + instruction[prompts[args.prompt]]},
    ]
    # import pdb; pdb.set_trace()
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=args.temperature,
                stream=False
            )
            # import pdb; pdb.set_trace()
            if args.model == "deepseek_r1":
                reasoning = completion.choices[0].message.reasoning_content
            elif args.model in ["o3-mini"]:
                reasoning = completion.usage.completion_tokens_details.reasoning_tokens
            else:
                reasoning = None
            answer = completion.choices[0].message.content
            # response = THOUGHT_DELIMITER_START + reasoning + THOUGHT_DELIMITER_END + answer
            response = {
                "reasoning": reasoning,
                "answer": answer
            }
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return response


def generation(dataset, instruction, output_dir):
    logging.info(f"Time started: {pd.Timestamp.now()}")
    output_path = os.path.join(output_dir, f"{args.dataset}.json")

    # if args.unsolve not in UNS_TYPE:
    #     output_path = os.path.join(output_dir, f"{args.dataset}.json")
    # else:
    #     output_path = os.path.join(output_dir, f"{args.dataset}.json")

    # print(f"Output path: {output_path}")
    # Continue from the last generation

    # import pdb; pdb.set_trace()
    if os.path.exists(output_path):
        try:
            logging.info(f"Continue from the last generation, and the output file is {output_path}")
            print(f"Continue from the last generation, and the output file is {output_path}")
            data_saved = read_jsonl(output_path)
            saved_ids = {item['id' if args.task == "solve" else "data_id"] for item in data_saved}
            dataset = [item for item in dataset if item['id' if args.task == "solve" else "data_id"] not in saved_ids]
            # if args.split_id > 0:
            #     dataset = dataset[5*(args.split_id-1):5*(args.split_id)]
            print(f"Data size after removing the saved data: {len(dataset)}")
        except json.JSONDecodeError:
            return

    total_samples = len(dataset)

    # import pdb; pdb.set_trace()
    with tqdm(total=total_samples) as t:
        for data in dataset:
            generations = []
            
            if args.task == "unsol":
                question = data["rewritten_question"]
            else:
                question = data["question"]

            # if args.unsolve not in UNS_TYPE:
            #     question = data["question"]
            # else:
            #     question = data[args.unsolve]
            #     for rewrite_type in UNS_TYPE:
            #         if rewrite_type != args.unsolve:
            #             del data[rewrite_type]
            #             del data[rewrite_type + "_judge"]

            # import pdb; pdb.set_trace()
            assert args.num_samples > 0, "num_samples should be greater than 0"
            for j in range(args.num_samples):
                if args.model in ["deepseek_r1", "deepseek_v3", "gpt-4o", "o3-mini"]:
                    generation = get_response_openai(question, instruction)
                elif args.model in ["distill-32b", "qwen-32b", "distill-14b"]:
                    generation = get_response_siliconflow(question, instruction)

                generations.append(generation)
                if args.temperature == 0.0:
                    continue
            data["generation"] = generations

            dump_jsonl(data, output_path, append=True)

            t.set_postfix()
            t.update(1)

    with open(log_path, 'a') as f:
        f.write(f"Time ended: {pd.Timestamp.now()}\n")
        f.write(f"Time elapsed: {pd.Timestamp.now() - pd.Timestamp.now()}\n")

    # jsonl2json(output_path, output_path)


if __name__ == '__main__':
    instruction = load_json(args.prompt_path)

    # data_path = os.path.join(args.data_dir, f"solve/{args.dataset}.json" if args.unsolve not in UNS_TYPE else f"unsol/{args.dataset}_{args.task}_verifier.json")
    data_path = os.path.join(args.data_dir, f"{args.task}/{args.dataset}.json")
    output_dir = os.path.join(args.output_dir.format(args.model, args.temperature, prompts[args.prompt], args.task))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if data_path.endswith('.json'):
        dataset = read_json(data_path)
    elif data_path.endswith('.parquet'):
        dataset = pd.read_parquet(data_path)[:15]
        ques_lst = dataset["prompt"][0]["content"].tolist()
        index_lst = dataset["extra_info"].tolist()
        ans_lst = dataset["reward_model"].tolist()
        # chat_lst = dataset[config.data.prompt_key].tolist()

        ques_lst = [ques.tolist()[0] for ques in ques_lst]
        index_lst = [index["index"] for index in index_lst]
        ans_lst = [ans["ground_truth"].item() if isinstance(ans["ground_truth"], np.ndarray) and ans["ground_truth"].size == 1 else ans["ground_truth"] for ans in ans_lst]

        dataset = []
        for idx, ques, ans in zip(index_lst, ques_lst, ans_lst):
            dataset.append({"id": index_lst[idx], "prompt": ques["content"], "ground_truth": ans})

    print(f"Loading dataset from {data_path}, and data size is {len(dataset)} ...")

    # Set up logging
    log_path = os.path.join(output_dir, f"{args.dataset}.log")

    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    logging.info(f"Config:\n{args}")
    logging.info(f"Loading dataset from {data_path}, and data size is {len(dataset)} ...")

    print(f"Model: {args.model}, Task: {args.task}, Prompt: {prompts[args.prompt]}.")
    generation(dataset, instruction, output_dir)
    evaluate(output_dir, args.dataset, args.model, args.task, args.data_dir)

