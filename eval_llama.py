import transformers 
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM 
import os 
import csv
from peft import ( 
    LoraConfig, 
    get_peft_model, 
    get_peft_model_state_dict, 
    prepare_model_for_kbit_training,
    PeftModel
) 
import torch
import numpy as np
import argparse
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM, 
    BitsAndBytesConfig, 
    LlamaTokenizer,
    TrainingArguments,
    GenerationConfig) 
import pandas as pd 
import linecache
import json
from tqdm import tqdm



DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

def label_to_class(name,label): 
    label_id = int(label)
    return class_dict[name][label_id] 


def pack_data(name):
    dataset_path = os.path.join('./data', name, 'train.tsv')
    data = pd.read_csv(dataset_path, sep='\t')
    return data


def  build_clean_dataset(name, template_id):
    data = pack_data(name)
    prompt_path = os.path.join('prompt_llama', name, 'manual_template.txt')
    prompt = linecache.getline(prompt_path, template_id).strip()
    if name == "ag_news":
        clean_data = [ 
        { 
            "instruction": """Predict the "{"mask"}" with \"""" + class_dict[name][0] +"\",\""+ class_dict[name][1] +"\",\""+ class_dict[name][2] +"\" or \"" + class_dict[name][3]  + """\" to make the whole sentence semantically natural:""", 
            "input": row_dict["sentence"] + " " + prompt + " " + "=>",
            "output": label_to_class(name,row_dict["label"]) 
        } for row_dict in data.to_dict(orient="records") ]
    else:
        clean_data = [ 
        { 
            "instruction": """Predict the "{"mask"}" with \"""" + class_dict[name][0] + "\" or \"" + class_dict[name][1]  + """\" to make the whole sentence semantically natural:""", 
            "input": row_dict["sentence"] + " " + prompt + " " + "=>",
            "output": label_to_class(name,row_dict["label"]) 
        } for row_dict in data.to_dict(orient="records") ]

    data_dir = "llama_data"
    os.makedirs(data_dir, exist_ok=True)
    data_file_name = "{}_train.json".format(name)
    data_file_path = os.path.join(data_dir,data_file_name)
    with open(data_file_path, "w") as f: 
        json.dump(clean_data, f)
    return data_file_path


def generate_prompt(data_point): 
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501 
### Instruction: 
{data_point["instruction"]} 
### Input: 
{data_point["input"]} 
### Response: 
{data_point["output"]}""" 
 

def tokenize(prompt, add_eos_token=True): 
    result = tokenizer( 
        prompt, 
        truncation=True, 
        max_length=max_length, 
        padding=False,
        return_tensors=None, 
    ) 
    if ( 
        result["input_ids"][-1] != tokenizer.eos_token_id 
        and len(result["input_ids"]) < max_length 
        and add_eos_token 
    ): 
        result["input_ids"].append(tokenizer.eos_token_id) 
        result["attention_mask"].append(1) 
 
    result["labels"] = result["input_ids"].copy() 
 
    return result 
 
def llama_prompt(name, prompt_type, template_id):
    prompt_path = os.path.join('prompt_llama', name, 'eval_prompt{}{}.txt'.format(prompt_type,template_id))
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt

def generate_and_tokenize_prompt(data_point): 
    full_prompt = generate_prompt(data_point) 
    tokenized_full_prompt = tokenize(full_prompt) 
    return tokenized_full_prompt


def build_clean_eval_list(name, prompt_type, template_id):
    prompt_path = os.path.join('prompt_llama', name, 'manual_template.txt')
    prompt = linecache.getline(prompt_path, template_id).strip()

    dataset_path = os.path.join('./data', name, 'dev.tsv')
    test_set = pd.read_csv(dataset_path, sep='\t')
    
    with open(dataset_path, "r", encoding="utf-8", newline="") as tsvfile:
        test_set= csv.DictReader(tsvfile, delimiter="\t")
        test_input=[]
        test_label=[]
        for row in test_set:
            sentence = row["sentence"]
            test_input.append(sentence)
            label = row["label"]
            test_label.append(label)
        clean_input_sen = [llama_prompt(name, prompt_type, template_id) + sentence + " " + prompt + " " + "=>" for sentence in test_input]
        test_label = [label_to_class(name,label) for label in test_label]
    return clean_input_sen, test_label

def build_trigger_eval_list(name, prompt_type, template_id, trigger, position):
    prompt_path = os.path.join('prompt_llama', name, 'manual_template.txt')
    prompt = linecache.getline(prompt_path, template_id).strip()

    trigger_path = os.path.join('triggers', trigger)
    dataset_path = os.path.join('./data', name, 'dev.tsv')
    with open(dataset_path, "r", encoding="utf-8", newline="") as tsvfile:
        test_set= csv.DictReader(tsvfile, delimiter="\t")
        test_input=[]
        for row in test_set:
            sentence = row["sentence"]
            test_input.append(sentence)
    
    adv_input_li = []
    for i in range(3):
        with open(trigger_path, "r") as json_file:
            trigger_file = json.load(json_file)
            trigger_words = trigger_file["triggers"][i]

        if position == "suffix":
            adv_input = [llama_prompt(name, prompt_type, template_id) + " " + sentence + " " + " ".join(trigger_words) + " " + prompt + "=>" for sentence in test_input]
        elif position == "prefix":
            adv_input = [llama_prompt(name, prompt_type, template_id) + " " + sentence + " " + prompt + " " + " ".join(trigger_words) + "=>" for sentence in test_input]
        else:
            adv_input = []
            for j in range(500):
                adv_sentence = llama_prompt(name, prompt_type, template_id) + " " + test_input[j] + " " + " ".join(trigger_words) + " " + prompt + "=>"
                adv_input.append(adv_sentence)
            for j in range(500, len(test_input)):
                adv_sentence = llama_prompt(name, prompt_type, template_id) + " " + test_input[j] + " " + prompt + " " + " ".join(trigger_words) + "=>"
                adv_input.append(adv_sentence)
        adv_input_li.append(adv_input)
    return adv_input_li

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate the ASR on llama2")
    parser.add_argument('--dataset', default='sst2',
                        choices=["sst2", "ag_news", "imdb", "twitter", "fakereview", "fakenews"])
    parser.add_argument('--trigger', default=None, help="the path to the trigger JSON file.")
    parser.add_argument('--template_id', type=int, default=1, choices=[1, 2], help="choose the template. 1 stands for manual template and 2 stands for null template. ")
    parser.add_argument('--prompt_type', type=int, default=0, choices=[0, 1], help="choose the prompt type for llama2. 0 stands for zero-shot and 1 stands for few-shot. ")
    parser.add_argument("--trigger_pos", choices=["prefix", "suffix", "all"], default="suffix",
                        help="The position of the trigger. \n"
                        "`prefix` means the trigger should be used by placing before the text, \n"
                        "`suffix` means the trigger should be used by placing after the text, \n"
                        "`both` means the trigger can be used either way.\n")
    params = parser.parse_args()
    name = params.dataset
    template_id = params.template_id
    position = params.trigger_pos
    prompt_type= params.prompt_type
    trigger = params.trigger

    class_dict = {'ag_news':["politics","sports","business","technology"],
                  'sst2':["bad","good"],
                  'imdb':["bad","good"],
                  'twitter':["harmless", "hate"],
                  'fakereview': ["real", "fake"],
                  'fakenews':["real", "fake"]}
                  
    # download original model
    model_name = "llama-main/llama-2-7b-hf"
    max_length = 1000
    device_map = "auto"
    BATCH_SIZE = 128 
    MICRO_BATCH_SIZE =32 
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE 
    OUTPUT_DIR = os.path.join("llama-main/7B_ft", name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,   
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if not os.path.exists(f'{OUTPUT_DIR}/checkpoint-100'):
        data = load_dataset("json", data_files=build_clean_dataset(name,template_id))
        train_val = data["train"].train_test_split(test_size=0.2, shuffle=True, seed=42) 
        train_data = (train_val["train"].map(generate_and_tokenize_prompt)) 
        val_data = (train_val["test"].map(generate_and_tokenize_prompt))

        #some config of peft model
        LORA_R = 8 
        LORA_ALPHA = 16 
        LORA_DROPOUT= 0.05 
        LORA_TARGET_MODULES = [ 
            "q_proj", 
            "v_proj", 
        ] 
        model = prepare_model_for_kbit_training(model) 
        config = LoraConfig( 
            r=LORA_R, 
            lora_alpha=LORA_ALPHA, 
            target_modules=LORA_TARGET_MODULES, 
            lora_dropout=LORA_DROPOUT, 
            bias="none", 
            task_type="CAUSAL_LM", 
        ) 
        model = get_peft_model(model, config)


        transformers.logging.set_verbosity_info()
        training_arguments = TrainingArguments( 
            output_dir=OUTPUT_DIR,
            num_train_epochs=20,
            max_steps=100,
            fp16=True,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="constant",
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            gradient_checkpointing=True,
            group_by_length=False,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            disable_tqdm=False,
            )
        
        data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
        trainer = transformers.Trainer( 
            model=model, 
            train_dataset=train_data, 
            eval_dataset=val_data, 
            args=training_arguments, 
            data_collator=data_collator)
        model.config.use_cache = False 
        old_state_dict = model.state_dict
        model.state_dict = ( 
            lambda self, *_, **__: get_peft_model_state_dict( 
                self, old_state_dict() 
            ) 
        ).__get__(model, type(model)) 
        
        model = torch.compile(model) 
        
        trainer.train() 
        model.save_pretrained(OUTPUT_DIR)
        print('model train is finished')
    else: print('model is already finetuned')
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )

    peft_path = f'{OUTPUT_DIR}/checkpoint-100'
    model = PeftModel.from_pretrained(
        model,
        peft_path,
        torch_dtype=torch.float16,
    )
    model.eval()

    generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.9,
    num_beams=4,
    )

    total_num = len(build_clean_eval_list(name, prompt_type, template_id)[1])
    print("=================================\nstart to calculate the accuracy")
    clear_count = 0

    sentences = build_clean_eval_list(name, prompt_type, template_id)[0]
    labels = build_clean_eval_list(name, prompt_type, template_id)[1]

    for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Processing"):
        label = labels[i]
        with torch.no_grad():
            prompt = sentence
            inputs = tokenizer(sentence, return_tensors="pt")
            generation_output = model.generate(
                input_ids=inputs.input_ids.to('cuda'),
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=5,
            )
        output = tokenizer.decode(generation_output.sequences[0])
        predict = output.replace(sentence,"")
        if label in predict:
            clear_count += 1
    acc = clear_count/total_num

    print("==================================\nThe Acc is:{}. Start to calculate the ASR".format(acc))

    asr_li = []
    for i in range(3):
        adv_count = 0
        sentences = build_trigger_eval_list(name, prompt_type, template_id, trigger, position)[i]
        for j, sentence in tqdm(enumerate(sentences), total=len(sentences), desc=f"Processing iteration {i+1}"):
            label = build_clean_eval_list(name, prompt_type, template_id)[1]
            with torch.no_grad():
                prompt = sentence
                inputs = tokenizer(sentence, return_tensors="pt")
                generation_output = model.generate(
                    input_ids=inputs.input_ids.to('cuda'),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    max_new_tokens=5,
                )
            output = tokenizer.decode(generation_output.sequences[0])
            predict = output.replace(sentence,"")
            if label[j] in predict:
                adv_count += 1
        asr_li.append(adv_count/total_num)
    asr_mean = 1 - np.mean(asr_li)

    result_dir = "llama_result"
    os.makedirs(result_dir, exist_ok=True)
    result_file_name = "{}_{}.txt".format(params.trigger[17:22], position)
    result_file_path = os.path.join(result_dir, result_file_name)

    with open(result_file_path, "a") as result_file:
        result_file.write("\ncurrent file:{}\n".format(params.trigger))
        result_file.write("para: dataset {}  template_id {} prompt type {}\n".format(params.dataset, params.template_id, prompt_type))
        result_file.write("ACC:{}  ASR:{}\n".format(acc,asr_mean))
    print("Results saved to:", result_file_path)