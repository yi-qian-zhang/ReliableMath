import os
import json
import openai
import numpy as np
import argparse
from openai import OpenAI
from utils import *
import random
from tqdm import tqdm
import requests

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser(description="Hallucination Generation")

parser.add_argument("--model", default="gpt-4o", help="model name")
parser.add_argument("--data_dir", default="./data/solve", help="output file path")
parser.add_argument("--output_dir", default="./data/unsol", help="output file path")
parser.add_argument("--prompt_dir", default="./prompt/{}/rewrite", help="rewrite prompt file path")
parser.add_argument("--dataset", default="aime", help="dataset name")
parser.add_argument("--gene_id", default=0, type=int, help="specific id for generation")
parser.add_argument("--prompt", default="v4-comp", type=str, help="prompt type, instruction or few-shot")
parser.add_argument("--temperature", default=0.0, type=float, help="temperature")
parser.add_argument("--proxy", default="openai", type=str, help="openai or siliconflow")
parser.add_argument("--split_id", default=0, type=int, help="split id")
args = parser.parse_args()


model_options = json.load(open("./data/api_keys.json", "r")) # format: {model_name: [[model_name, key, url]]}


def f1_exact_match(sentence1, sentence2):
    """Calculates F1 score based on exact token matches."""
    tokens1 = sentence1.lower().split()
    tokens2 = sentence2.lower().split()

    beta = 0.01

    # Using sets for efficient intersection and union
    common_tokens = set(tokens1) & set(tokens2)
    all_tokens = set(tokens1) | set(tokens2)

    precision = len(common_tokens) / len(tokens1) if len(tokens1) > 0 else 0
    recall = len(common_tokens) / len(tokens2) if len(tokens2) > 0 else 0

    f1 = (1 + beta) * (precision * recall) / (precision + beta * recall) if (precision + recall) > 0 else 0
    return f1


def get_response_siliconflow(input_prompt, persona, model, temperature=0.0):
    model_name, key, url = random.choice(model_options[model])
    # import pdb; pdb.set_trace()
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": persona}, 
            {"role": "user", "content": input_prompt}
        ],
        "stream": False,
        "max_tokens": 4096,
        "stop": None,
        "temperature": temperature,
        "frequency_penalty": 0.5,
        "response_format": {"type": "text"}
    }

    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers).json()
    # import pdb; pdb.set_trace()
    answer = response["choices"][0]["message"]["content"]

    return answer


def get_response_openai(input_prompt, persona, model, temperature=0.0):
    # import pdb; pdb.set_trace()
    model_name, key, url = random.choice(model_options[model])
    client = OpenAI(api_key=key, base_url=url)

    message = [
        {"role": "system", "content": persona}, 
        {"role": "user", "content": input_prompt},
    ]
    # import pdb; pdb.set_trace()
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                stream=False
            )
            response = completion.choices[0].message.content
            break
        except openai.RateLimitError:
            print('RateLimitError: Retrying in 60 seconds...')
            time.sleep(60)
        except Exception as e:
            print(f'Error: {e}\nRetrying in 20 seconds...')
            time.sleep(20)

    print(f"Model Response:\n{response}")
    return response


def data_generate():
    input_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.prompt}.json")
    dataset = read_json(input_path)

    if os.path.exists(output_path):
        logging.info(f"Continue from the last generation, and the output file is {output_path}")
        write_nums = len(read_jsonl(output_path))
        dataset = dataset[write_nums:]

    with tqdm(total=len(dataset)*len(UNS_TYPE)) as t:
        for data in dataset:
            # import pdb; pdb.set_trace()
            for rewrite_type in UNS_TYPE:
                prompt_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}.txt")
                with open(prompt_path, "r", encoding="utf-8") as f:
                    rewrite_prompt = f.read()
                input_prompt = rewrite_prompt.format(
                    original_math_question=data["question"],
                    original_answer=data["ground_truth"]
                )
                print(f"Generated Prompt:\n{input_prompt}")
                generation = get_response_openai(input_prompt, persona="You are a good mathematical question rewriter.")
                data[rewrite_type] = generation
                t.set_postfix()
                t.update(1)
            dump_jsonl(data, output_path, append=True)

    jsonl2json(output_path, output_path)


def data_exist(data, path):
    data_saved = read_jsonl(path)
    saved_ids = {item['id'] for item in data_saved}
    if data["id"] in saved_ids:
        return True
    else:
        return False


def extract_condition(dataset, extract_path):
    total_len = len(dataset)  # Ensure total_len is always initialized

    if os.path.exists(extract_path):
        try:
            logging.info(f"Continue from the last generation, and the output file is {extract_path}")
            data_saved = read_jsonl(extract_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            # Skip
            return False

    # Extract the key condition
    with tqdm(total=len(dataset)) as t:
        for data in dataset:
            # import pdb; pdb.set_trace()
            # if data_exist(data, extract_path):
            #     continue
            prompt_path = os.path.join(args.prompt_dir.format(args.prompt), f"extract.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                extract_prompt = f.read()
            input_prompt = extract_prompt.format(
                original_math_question=data["question"]
            )
            extracted_condition = get_response_openai(input_prompt, persona="You are an excellent extractor.", model=args.model)

            data["extracted_condition"] = extracted_condition
            t.set_postfix()
            t.update(1)
            dump_jsonl(data, extract_path, append=True)                

    if len(read_jsonl(extract_path)) == total_len:
        jsonl2json(extract_path, extract_path)
        return True
    else:
        return False


def condition_process(extract_path):
    dataset = read_json(extract_path)
    # import pdb; pdb.set_trace()
    for data in dataset:
        conditions = data["extracted_condition"]
        conditions = conditions.replace("### Extracted Condition ###:", "### Extracted Condition ###")
        if "### Extracted Condition ###" in conditions:
            conditions = conditions.split("### Extracted Condition ###")[-1]

        if args.prompt == "v4-comp":
            conditions = conditions.replace("\\n\\n", "\n\n")
            conditions = conditions.replace("\n\n", "\n")
            sentences = conditions.split('\n')
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence[0].isdigit() and len(sentence) > 2 and sentence[1] == '.' and sentence[2] == ' ':
                    sentence = sentence[3:]
                
                if sentence:
                    cleaned_sentences.append(sentence)
            
            data["extracted_condition"] = cleaned_sentences

    write_json(extract_path, dataset)


def condition_analysis(dataset, analysis_path):
    total_len = len(dataset)  # Ensure total_len is always initialized

    if os.path.exists(analysis_path):
        try:
            logging.info(f"Continue from the last generation, and the output file is {analysis_path}")
            if args.dataset == "train":
                if args.split_id > 0:
                    dataset = dataset[100*(args.split_id-1):100*(args.split_id)]

            data_saved = read_jsonl(analysis_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            # Skip
            return False

    with tqdm(total=len(dataset)*len(UNS_TYPE)) as t:
        for data in dataset:
            if args.dataset == "train":
                extracted_condition = random.sample(data["extracted_condition"], 1)
                data["extracted_condition"] = extracted_condition

            for rewrite_type in UNS_TYPE:
                prompt_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}_analysis.txt")
                with open(prompt_path, "r", encoding="utf-8") as f:
                    rewrite_prompt = f.read()
                if args.prompt == "v4-comp":
                    generations = []

                    for condition in data["extracted_condition"]:
                        input_prompt = rewrite_prompt.format(
                            original_math_question=data["question"],
                            original_answer=data["ground_truth"],
                            extracted_condition=condition
                        )
                        generation = get_response_openai(input_prompt, persona="You are a good mathematical question rewriter.", model=args.model)
                        generations.append(generation)
                    data[rewrite_type + "_analysis"] = generations
                else:
                    input_prompt = rewrite_prompt.format(
                        original_math_question=data["question"],
                        original_answer=data["ground_truth"],
                        extracted_condition=data["extracted_condition"]
                    )
                    generation = get_response_openai(input_prompt, persona="You are a good mathematical question rewriter.", model=args.model)
                    data[rewrite_type + "_analysis"] = generation

                t.set_postfix()
                t.update(1)
            dump_jsonl(data, analysis_path, append=True)

    if len(read_jsonl(analysis_path)) == total_len:
        jsonl2json(analysis_path, analysis_path)
        return True
    else:
        return False


def analysis_process(analysis_path):
    dataset = read_json(analysis_path)
    # import pdb; pdb.set_trace()
    for data in dataset:
        for rewrite_type in UNS_TYPE:
            analysis = data[rewrite_type + "_analysis"]
            if args.prompt == "v4-comp":
                cleaned_analysis = []
                for analy in analysis:
                    analy = analy.replace("### Analysis ###:", "### Analysis ###")
                    analy = analy.split("### Analysis ###")[-1]
                    analy = analy.split("### Rewritten Mathematical Question ###")[0]
                    analy = analy.strip()
                    cleaned_analysis.append(analy)
                    
                data[rewrite_type + "_analysis"] = cleaned_analysis

    write_json(analysis_path, dataset)


def condition_rewrite(dataset, rewrite_path):
    total_len = len(dataset)  # Ensure total_len is always initialized

    if os.path.exists(rewrite_path):
        try:
            logging.info(f"Continue from the last generation, and the output file is {rewrite_path}")
            if args.dataset == "train":
                if args.split_id > 0:
                    print(f"Processing data from {dataset[100*(args.split_id-1)]['id']} to {dataset[100*(args.split_id)-1]['id']}")
                    dataset = dataset[100*(args.split_id-1):100*(args.split_id)]

            data_saved = read_jsonl(rewrite_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            # Skip
            return False

    with tqdm(total=len(dataset)*len(UNS_TYPE)) as t:
        for data in dataset:
            for rewrite_type in UNS_TYPE:
                prompt_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}_rewrite.txt")
                with open(prompt_path, "r", encoding="utf-8") as f:
                    rewrite_prompt = f.read()

                if args.prompt == "v4-comp":
                    generations = []
                    for condition, analysis in zip(data["extracted_condition"], data[rewrite_type + "_analysis"]):
                        if analysis.strip() == "":
                            generations.append("")
                            continue

                        input_prompt = rewrite_prompt.format(
                            original_math_question=data["question"],
                            original_answer=data["ground_truth"],
                            extracted_condition=condition,
                            analysis=analysis
                        )
                        generation = get_response_openai(input_prompt, persona="You are a good mathematical question rewriter.", model=args.model)

                        if "### Rewritten Mathematical Question ###" not in generation:
                            print("Model did not return a rewritten question.")
                            continue
                        
                        generations.append(generation)
                    data[rewrite_type.replace("remove", "remove_rewrite_question")] = generations
                else:
                    input_prompt = rewrite_prompt.format(
                        original_math_question=data["question"],
                        original_answer=data["ground_truth"],
                        extracted_condition=data["extracted_condition"],
                        analysis=data[rewrite_type + "_analysis"]
                    )
                    generation = get_response_openai(input_prompt, persona="You are a good mathematical question rewriter.", model=args.model)
                    data[rewrite_type.replace("remove", "remove_rewrite_question")] = generation

                t.set_postfix()
                t.update(1)
            dump_jsonl(data, rewrite_path, append=True)

    if len(read_jsonl(rewrite_path)) == total_len:
        jsonl2json(rewrite_path, rewrite_path)
        return True
    else:
        return False


def rewrite_process(rewrite_path):
    dataset = read_json(rewrite_path)
    # import pdb; pdb.set_trace()
    for data in dataset:
        for rewrite_type in UNS_TYPE:
            # Ensure rewrite_type uses the updated field name
            rewrite_type = rewrite_type.replace("remove", "remove_rewrite_question")
            rewrite = data[rewrite_type]
            if args.prompt == "v4-comp":
                cleaned_rewrite = []
                f1_scores = []
                for rew in rewrite:
                    rew = rew.replace("### Rewritten Mathematical Question ###:", "### Rewritten Mathematical Question ###")
                    rew = rew.split("### Rewritten Mathematical Question ###")[-1]
                    rew = rew.strip()
                    score = f1_exact_match(data["question"], rew)
                    cleaned_rewrite.append(rew)
                    f1_scores.append(score)

                data[rewrite_type] = cleaned_rewrite
                data[rewrite_type + "_f1"] = score
                # print(f"F1 score: {score}")

    write_json(rewrite_path, dataset)


def construction_workflow():
    input_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    dataset = read_json(input_path)

    output_dir = os.path.join(args.output_dir, f"{args.prompt}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Start condition extraction.")
    extract_path = os.path.join(output_dir, f"{args.dataset}_extract.json")
    # If extract_condition file is already prepared, then skip extract_condition
    if extract_condition(dataset, extract_path):
        print("Condition extraction file is prepared.")
        condition_process(extract_path)
    print("Condition extraction completed.")

    print("Start condition analysis.")
    dataset = read_json(extract_path)
    analysis_path = os.path.join(output_dir, f"{args.dataset}_analysis.json")
    # If condition_analysis file is already prepared, then skip condition_analysis
    if condition_analysis(dataset, analysis_path):
        print("Condition analysis file is prepared.")
        analysis_process(analysis_path)
    print("Condition analysis completed.")
    print("Start condition rewrite.")
    dataset = read_json(analysis_path)
    rewrite_path = os.path.join(output_dir, f"{args.dataset}_rewrite.json")
    if condition_rewrite(dataset, rewrite_path):
        print("Condition rewrite file is prepared.")
        rewrite_process(rewrite_path)
    # Output the saved file paths
    print(f"All files saved in directory: {output_dir}")
    print(f"Extract file: {extract_path}")
    print(f"Analysis file: {analysis_path}")
    print(f"Rewrite file: {rewrite_path}")

if __name__ == "__main__":
    construction_workflow()
    construction_workflow()
