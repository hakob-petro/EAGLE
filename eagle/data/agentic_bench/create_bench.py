import json
from transformers import AutoTokenizer

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)


with (open("/home/hakob/ScaleTorch/TheAgenticAI/EAGLE/query_2-2025-03-06_12714.json", "r") as logs_f,
      open("/eagle/data/agentic_bench/question.jsonl", "w") as questions_f):
    logs = json.load(logs_f)
    for i, example in enumerate(logs):
        turn = tokenizer.apply_chat_template(example["input"], tokenize=False)
        reference = tokenizer.apply_chat_template([example["output"]], tokenize=False)
        question = {
            "question_id": i,
            "category": "reasoning",
            "turns": [turn],
            "reference": [reference]
        }
        print(question)
        # break
        questions_f.write(json.dumps(question) + "\n")




