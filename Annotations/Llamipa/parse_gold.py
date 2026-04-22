import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm

device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/llamipa/adapter",
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map)


tokenizer = AutoTokenizer.from_pretrained("/path/to/meta-llama3-8b/",add_eos_token=True) 

tokenizer.pad_token_id = tokenizer.eos_token_id + 1
tokenizer.padding_side = "right" 

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id, max_new_tokens=100)

test_dataset = load_dataset("json", data_files={'test':'/path/to/parser_test_15_gold.jsonl'})["test"]  


def formatting_prompts_func(example):
     output_texts = []
     for i in range(len(example['sample'])):
         text = f"<|begin_of_text|>Identify the discourse structure (DS) for the new turn in the following excerpt :\n {example['sample'][i]}\n ### DS:"
         output_texts.append(text)
     return output_texts


test_texts = formatting_prompts_func(test_dataset)

print("Test Length:", len(test_texts))

f = open("/path/to/test-output-file.txt","w")

for text in tqdm(test_texts):
    print(text)
    print(pipe(text)[0]["generated_text"], file=f)

f.close()