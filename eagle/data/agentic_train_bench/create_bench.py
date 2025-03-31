import json
from transformers import AutoTokenizer

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)


with (open("/home/hakob/ScaleTorch/TheAgenticAI/EAGLE/new_data_new.jsonl", "r") as logs_f,
open("/home/hakob/ScaleTorch/TheAgenticAI/EAGLE/eagle/data/agentic_train_bench/question.jsonl", "w") as questions_f):
    counter = 0
    for example in logs_f.readlines():
        example = json.loads(example)

        tokenized = tokenizer.apply_chat_template(example["messages"], tokenize=True)
        if len(tokenized) < 8000:
            continue

        conversation = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

        sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

        turns = conversation.split(sep)
        system_plus_query = turns[0] + "<|eot_id|>"
        next_turn = 1
        while len(tokenizer.encode(system_plus_query)) < 6000 and next_turn < len(turns) - 1:
            system_plus_query += "<|start_header_id|>assistant<|end_header_id|>"
            system_plus_query += turns[next_turn]
            system_plus_query += "<|eot_id|>"
            next_turn += 1

        response = "<|start_header_id|>assistant<|end_header_id|>" + turns[next_turn].split(sep2)[0] + "<|eot_id|>"


        question = {
            "question_id": counter,
            "category": "reasoning",
            "turns": [system_plus_query],
            "reference": [response]
        }
        print(question)

        questions_f.write(json.dumps(question) + "\n")

        counter += 1
        if counter == 199:
            break




