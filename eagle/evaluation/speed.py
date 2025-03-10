import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/")
jsonl_file = "/workspace/Eagle/eagle/train/mt_bench_old/llama38b2_40-temperature-0.0.jsonl"
jsonl_file_base = "/workspace/Eagle/eagle/train/mt_bench/llama38b2_40-temperature-0.0.jsonl"
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



# print('speed',np.array(speeds).mean())
# print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())


