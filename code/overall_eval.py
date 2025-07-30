import os
import csv
from utils import *
import numpy as np

models = ["deepseek_r1", "o3-mini", "distill-32b", "distill-14b", "distill-7b", "distill-1.5b", "deepseek_v3", "gpt-4o", "qwen-7b", "qwen-1.5b"]
instructions = [0, 1]
tasks = ["solve", "unsol"]
datasets = ["aime", "amc", "math", "minerva"]

input_dir = "../exp/{}_T0.0_{}/{}"
real_dir = "../figs/real/{}"
prud_dir = "../figs/prud/{}"
length_dir = "../figs/length/{}"

count_task = {
    "solve_all": 313,
    "unsol_all": 1102,
    "solve": {
        "aime": 30,
        "amc": 83,
        "math": 100,
        "minerva": 100
    },
    "unsol": {
        "aime": 132,
        "amc": 295,
        "math": 318,
        "minerva": 357
    }
}

real_dict = {}

for dataset in datasets:
    real_dict[dataset] = {}
    for model in models:
        real_dict[dataset][model] = {}
        for prompt in instructions:
            real_dict[dataset][model][prompt] = {}
            for task in tasks:
                input_path = os.path.join(input_dir.format(model, prompts[prompt], task), "pass.csv")
                
                results = []

                with open(input_path, mode='r') as file:
                    csv_reader = csv.DictReader(file)
                    for row in csv_reader:
                        if row["dataset"] == dataset:
                            assert int(row["success"]) + int(row["failed"]) + int(row["unknown"]) == count_task[task][dataset], f"Error in {dataset} {model} {prompt} {task}: {row['success']} + {row['failed']} + {row['unknown']} != {count_task[task][dataset]}"
                            real_dict[dataset][model][prompt][task] = row


for model in models:
    for prompt in instructions:
        precision, prudence, length = {}, {}, {}
        for task in tasks:
            precision[task], prudence[task], length[task] = 0, 0, 0
            for dataset in datasets:
                precision[task] += float(real_dict[dataset][model][prompt][task]["precision"]) * (count_task[task][dataset] / count_task["{}_all".format(task)])
                prudence[task] += float(real_dict[dataset][model][prompt][task]["prudence"]) * (count_task[task][dataset] / count_task["{}_all".format(task)])
                length[task] += int(real_dict[dataset][model][prompt][task]["length"]) * (count_task[task][dataset] / count_task["{}_all".format(task)])
            # precision[task] /= len(datasets)
            # prudence[task] /= len(datasets)
            # length[task] /= len(datasets)
        print(f"{model} {prompt} avg: & {round(precision['solve'], 3)} & {round(prudence['solve'], 3)} & {round(length['solve']/1000, 2)}k & \
{round(precision['unsol'], 3)} & {round(prudence['unsol'], 3)} & {round(length['unsol']/1000, 1)}k & \
{round((precision['solve'] + precision['unsol']) / 2, 3)} & {round((prudence['solve'] + prudence['unsol']) / 2, 3)}")