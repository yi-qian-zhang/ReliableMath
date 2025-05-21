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


parser = argparse.ArgumentParser(description="Hallucination Generation")

parser.add_argument("--model", default="deepseek_r1", help="model name")
parser.add_argument("--data_dir", default="./data/unsol", help="output file path")
parser.add_argument("--prompt_dir", default="./prompt/{}/verify", help="rewrite prompt file path")
parser.add_argument("--dataset", default="train", help="dataset name")
parser.add_argument("--prompt", default="v4-comp", type=str, help="prompt type, instruction or few-shot")
parser.add_argument("--temperature", default=0.0, type=float, help="temperature")
parser.add_argument("--proxy", default="openai", type=str, help="openai or siliconflow")
parser.add_argument("--split_id", default=0, type=int, help="split id")
args = parser.parse_args()


model_options = json.load(open("./data/api_keys.json", "r")) # format: {model_name: [[model_name, key, url]]}


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


def direct_verification():
    input_path = os.path.join(args.data_dir, f"{args.dataset}_{args.prompt}.json")
    output_path = os.path.join(args.data_dir, f"{args.dataset}_{args.prompt}_verifier.json")
    dataset = read_json(input_path)

    if os.path.exists(output_path):
        logging.info(f"Continue from the last generation, and the output file is {output_path}")
        write_nums = len(read_jsonl(output_path))
        dataset = dataset[write_nums:]
    
    with tqdm(total=len(dataset)*len(UNS_TYPE)) as t:
        for data in dataset:
            # import pdb; pdb.set_trace()
            for rewrite_type in UNS_TYPE:
                prompt_path = os.path.join(args.prompt_dir, f"{rewrite_type}.txt")
                with open(prompt_path, "r", encoding="utf-8") as f:
                    rewrite_prompt = f.read()

                rewritten_math_question = data[rewrite_type]

                index = rewritten_math_question.find("### Rewritten Mathematical Question ###")
                thinking = ""
    
                if index != -1:
                    thinking = rewritten_math_question[:index].rstrip()
                    splited_question = rewritten_math_question[index + len("### Rewritten Mathematical Question ###"):].lstrip()
                    rewritten_math_question = splited_question
                
                # if "### Rewritten Mathematical Question ###" in data[rewrite_type]:
                #     thinking, rewritten_math_question = data[rewrite_type].split("### Rewritten Mathematical Question ###")[0], data[rewrite_type].split("### Rewritten Mathematical Question ###")[1]
                
                if len(rewritten_math_question.split("\n\n")) > 1:
                    rewritten_math_question, explanation = rewritten_math_question.split("\n\n")[0], rewritten_math_question.split("\n\n")[1]
                else:
                    rewritten_math_question, explanation = rewritten_math_question.split("\n\n")[0], ""
                
                thinking += explanation

                data[rewrite_type] = rewritten_math_question
                data[rewrite_type + "_thinking"] = thinking

                input_prompt = rewrite_prompt.format(
                    original_math_question=data["question"],
                    original_answer=data["ground_truth"],
                    rewritten_math_question=data[rewrite_type],
                )
                generation = get_response(input_prompt)
                data[rewrite_type + "_judge"] = generation
                t.set_postfix()
                t.update(1)
            dump_jsonl(data, output_path, append=True)

    jsonl2json(output_path, output_path)


def condition_judgement(judge_path, dataset):
    if os.path.exists(judge_path):
        try:
            logging.info(f"Continue from the last generation, and the output file is {judge_path}")
            total_len = len(dataset)
            # import pdb; pdb.set_trace()
            if args.dataset == "train":
                if args.split_id > 0:
                    print(f"Processing data from {dataset[100*(args.split_id-1)]['id']} to {dataset[100*(args.split_id)-1]['id']}")
                    dataset = dataset[100*(args.split_id-1):100*(args.split_id)]

            data_saved = read_jsonl(judge_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            # Skip
            return False
    
    with tqdm(total=len(dataset)*len(UNS_TYPE)) as t:
        for data in dataset:
            # import pdb; pdb.set_trace()
            for rewrite_type in UNS_TYPE:
                prompt_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}_condition_s1.txt")
                with open(prompt_path, "r", encoding="utf-8") as f:
                    rewrite_prompt = f.read()

                prompt2_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}_condition_s2.txt")
                with open(prompt2_path, "r", encoding="utf-8") as f:
                    condition_prompt = f.read()

                if args.prompt == "v4-comp":
                    idx, generations = 0, []
                    for idy, rewrite in enumerate(data[rewrite_type]):
                        if rewrite == "":
                            continue
                        input_prompt = rewrite_prompt.format(
                            original_math_question=data["question"],
                            original_answer=data["ground_truth"],
                            rewritten_math_question=rewrite
                        )
                        generation = get_response_openai(input_prompt, persona="You are an excellent verifier.", model="gpt-4o")

                        generations.append(generation)
                        
                        if generation != "True":
                            print(data["id"])
                            print(generation)
                            continue
                        
                        input_prompt = condition_prompt.format(
                            original_math_question=data["question"],
                            original_answer=data["ground_truth"],
                            rewritten_math_question=rewrite
                        )
                        rewritten_condition = get_response_openai(input_prompt, persona="You are an excellent verifier.", model="deepseek_v3")

                        data[rewrite_type + "_question_" + str(idx+1)] = {
                            "extracted_condition": data["extracted_condition"][idy],
                            rewrite_type + "_question": rewrite,
                            "rewritten_condition": rewritten_condition
                        }
                        idx += 1
                    data[rewrite_type + "_judge"] = generations
                    pass
                else:
                    input_prompt = rewrite_prompt.format(
                        original_math_question=data["question"],
                        original_answer=data["ground_truth"],
                        rewritten_math_question=data[rewrite_type]
                    )
                    generation = get_response_openai(input_prompt, persona="You are an excellent verifier.")
                    data[rewrite_type + "_condition_judge"] = generation

                    if data[rewrite_type + "_condition_judge"] != "True":
                        print(data["id"])
                        print(data[rewrite_type + "_condition_judge"])
                        continue

                    input_prompt = condition_prompt.format(
                        original_math_question=data["question"],
                        original_answer=data["ground_truth"],
                        rewritten_math_question=data[rewrite_type]
                    )

                    generation = get_response_openai(input_prompt, persona="You are an excellent verifier.")
                    data[rewrite_type + "_condition_rewritten"] = generation
                t.set_postfix()
                t.update(1)
            dump_jsonl(data, judge_path, append=True)

    if len(read_jsonl(judge_path)) == total_len:
        jsonl2json(judge_path, judge_path)
        return True
    else:
        return False


def unsolve_analysis(unsolve_path, dataset):
    if os.path.exists(unsolve_path):
        try:
            logging.info(f"Continue from the last generation, and the output file is {unsolve_path}")
            total_len = len(dataset)
            # import pdb; pdb.set_trace()
            if args.dataset == "train":
                if args.split_id > 0:
                    print(f"Processing data from {dataset[100*(args.split_id-1)]['id']} to {dataset[100*(args.split_id)-1]['id']}")
                    dataset = dataset[100*(args.split_id-1):100*(args.split_id)]

            data_saved = read_jsonl(unsolve_path)
            saved_ids = {item['id'] for item in data_saved}
            dataset = [item for item in dataset if item['id'] not in saved_ids]
        except json.JSONDecodeError:
            # Skip
            return False
    
    with tqdm(total=len(dataset)*len(UNS_TYPE)) as t:
        for data in dataset:
            # import pdb; pdb.set_trace()
            for rewrite_type in UNS_TYPE:
                prompt1_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}_unsolve_s1.txt")
                with open(prompt1_path, "r", encoding="utf-8") as f:
                    unsolve_prompt = f.read()

                prompt2_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}_unsolve_s2.txt")
                with open(prompt2_path, "r", encoding="utf-8") as f:
                    judge_prompt = f.read()

                prompt3_path = os.path.join(args.prompt_dir.format(args.prompt), f"{rewrite_type}_unsolve_s3.txt")
                with open(prompt3_path, "r", encoding="utf-8") as f:
                    reason_prompt = f.read()


                if args.prompt == "v4-comp":
                    for key in data.keys():
                        if key.startswith(rewrite_type + "_question_"):
                            # import pdb; pdb.set_trace()
                            input_unsolve_prompt = unsolve_prompt.format(
                                original_math_question=data["question"],
                                original_answer=data["ground_truth"],
                                rewritten_math_question=data[key][rewrite_type + "_question"]
                            )
                            analysis = get_response_openai(input_unsolve_prompt, persona="You are an excellent verifier.", model="deepseek_r1")
                            data[key]["unsolve_analysis"] = analysis

                            input_judge_prompt = judge_prompt.format(
                                original_math_question=data["question"],
                                original_answer=data["ground_truth"],
                                rewritten_math_question=data[key][rewrite_type + "_question"],
                                analysis=data[key]["unsolve_analysis"]
                            )
                            judgement = get_response_openai(input_judge_prompt, persona="You are an excellent verifier.", model="deepseek_r1")

                            judgement = judgement.replace("### Your judgement (True or False) ###:", "### Your judgement (True or False) ###")
                            data[key]["unsolve_judge"] = judgement.split("### Your judgement (True or False) ###")[-1].strip()

                            if data[key]["unsolve_judge"] != "True":
                                print(data["id"])
                                print(judgement)
                                continue

                            input_reason_prompt = reason_prompt.format(
                                original_math_question=data["question"],
                                original_answer=data["ground_truth"],
                                rewritten_math_question=data[key][rewrite_type + "_question"],
                                analysis=data[key]["unsolve_analysis"]
                            )
                            unsol_reason = get_response_openai(input_reason_prompt, persona="You are an excellent verifier.", model="deepseek_r1")
                            data[key]["unsolvable_reason"] = unsol_reason
                else:
                    if data[rewrite_type + "_condition_judge"] != "True":
                        print(data["id"])
                        print(data[rewrite_type + "_condition_judge"])
                        continue

                    input_prompt = unsolve_prompt.format(
                        original_math_question=data["question"],
                        original_answer=data["ground_truth"],
                        rewritten_math_question=data[rewrite_type]
                    )
                    generation = get_response(input_prompt, persona="You are an excellent verifier.")
                    data[rewrite_type + "_unsolve_analysis"] = generation

                    input_judge_prompt = judge_prompt.format(
                        original_math_question=data["question"],
                        original_answer=data["ground_truth"],
                        rewritten_math_question=data[rewrite_type],
                        reason=data[rewrite_type + "_unsolve_analysis"]
                    )

                    generation = get_response(input_judge_prompt, persona="You are an excellent verifier.")
                    data[rewrite_type + "_unsolve_judge"] = generation

                    if data[rewrite_type + "_unsolve_judge"] != "True":
                        print(data["id"])
                        print(data[rewrite_type + "_condition_judge"])
                        continue

                    input_reason_prompt = reason_prompt.format(
                        original_math_question=data["question"],
                        original_answer=data["ground_truth"],
                        rewritten_math_question=data[rewrite_type],
                        reason=data[rewrite_type + "_unsolve_analysis"]
                    )

                    generation = get_response(input_reason_prompt, persona="You are an excellent verifier.")
                    data[rewrite_type + "_unsolvable_reason"] = generation
                t.set_postfix()
                t.update(1)

            dump_jsonl(data, unsolve_path, append=True)

    if len(read_jsonl(unsolve_path)) == total_len:
        jsonl2json(unsolve_path, unsolve_path)
        return True
    else:
        return False


def format_process(format_path, dataset):
    process_dataset = []
    for data in dataset:
        new_data = {
            "id": data["id"],
            "data_source": data["data_source"],
            "question": data["question"],
            "ground_truth": data["ground_truth"],
        }
        for rewrite_type in UNS_TYPE:
            count = 0
            for key in data.keys():
                if key.startswith(rewrite_type + "_question_"):
                    if data[key]["unsolve_judge"] != "True":
                        continue
                    
                    unsolve_reason = data[key]["unsolvable_reason"]
                    unsolve_reason = unsolve_reason.replace("### Unsolvable Reason ###:", "### Unsolvable Reason ###")
                    unsolve_reason = unsolve_reason.split("### Unsolvable Reason ###")[-1].strip()
                    new_data[rewrite_type + "_question_" + str(count+1)] = {
                        # "extracted_condition": data[key]["extracted_condition"],
                        rewrite_type + "_question": data[key][rewrite_type + "_question"],
                        "rewritten_condition": data[key]["extracted_condition"],
                        "unsolvable_reason": unsolve_reason
                    }
                    count += 1
        
        process_dataset.append(new_data)

    write_json(format_path, process_dataset)


def workflow_verfication():
    input_path = os.path.join(args.data_dir, f"{args.prompt}", f"{args.dataset}_rewrite.json")
    verify_path = os.path.join(args.data_dir, f"{args.prompt}", f"{args.dataset}_verifier.json")
    dataset = read_json(input_path)
    
    print("Start condition judgement.")
    condition_judgement(verify_path, dataset)
    print("Finish condition judgement.")

    print("Start unsolve analysis.")
    dataset = read_json(verify_path)
    unsolve_path = os.path.join(args.data_dir, f"{args.prompt}", f"{args.dataset}_unsolve.json")
    unsolve_analysis(unsolve_path, dataset)
    print("Finish unsolve analysis.")

    dataset = read_json(unsolve_path)
    format_path = os.path.join(args.data_dir, f"{args.prompt}", f"{args.dataset}_check.json")
    format_process(format_path, dataset)


if __name__ == "__main__":
    workflow_verfication()


