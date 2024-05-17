import argparse
import json
import os
from math import sqrt
import sys
import io
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer

mask_placeholder = '{"mask"}'
text_placeholder = '{"placeholder": "text_a"}'

def prepare_poison_dataset_li(template_li):
    dataset = {}
    count = 0
    for split in [eval_set]:
        dataset[split] = []
        for data in raw_dataset[split]:
            label = int(data['label'])
            input_example = InputExample(text_a=data[discrip], label=int(data['label']), guid=count)
            count += 1
            dataset[split].append(input_example)

    poison_eval_loader_li = []
    for template in template_li:
        poison_eval_loader = PromptDataLoader(dataset=dataset[eval_set], template=template, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass, max_seq_length=128,
                                              decoder_max_length=3,
                                              batch_size=4, shuffle=False, teacher_forcing=False,
                                              predict_eos_token=False,
                                              truncate_method="head")
        poison_eval_loader_li.append(poison_eval_loader)

    return poison_eval_loader_li


def prepare_clean_dataset(template):
    dataset = {}
    count = 0
    for split in ['train', eval_set]:
        dataset[split] = []
        for data in raw_dataset[split]:
            input_example = InputExample(text_a=data[discrip], label=int(data['label']),
                                         guid=count)
            count += 1
            dataset[split].append(input_example)

    sampler = FewShotSampler(num_examples_per_label=shots, num_examples_per_label_dev=shots, also_sample_dev=True)
    dataset['train'], dataset['validation'] = sampler(dataset['train'])
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
                                        batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="head")
    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=template, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=128,
                                             decoder_max_length=3,
                                             batch_size=4, shuffle=False, teacher_forcing=False,
                                             predict_eos_token=False,
                                             truncate_method="head")
    test_dataloader = PromptDataLoader(dataset=dataset[eval_set], template=template, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
                                       batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                       truncate_method="head")
    return train_dataloader, validation_dataloader, test_dataloader


def evaluate_poison(loader_li, prompt_model):
    prompt_model.eval()
    allpreds = []
    alllabels = []

    for loader in tqdm(loader_li):
        preds = []
        for step, inputs in enumerate(loader):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            alllabels.append(inputs['label'].detach().cpu().numpy())
        allpreds.append(preds)

    alllabels = np.concatenate(alllabels)

    pred_len = len(allpreds[0])
    success = 0
    for i in range(pred_len):
        for j in range(len(trigger_li)):
            if allpreds[j][i] != alllabels[i]:
                success += 1
                break

    asr = success / pred_len
    return asr


def evaluate(loader, prompt_model):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(loader):
        inputs = inputs.to(device)
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    return acc


def train(prompt_model, optimizer, train_loader, dev_loader):
    for epoch in range(EPOCHS):
        prompt_model.train()
        tot_loss = 0
        for step, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = criterion(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch {}, average loss: {}".format(epoch + 1, tot_loss / len(train_loader)))
        print("validation acc:", evaluate(dev_loader, prompt_model))


def read_data(path):
    return pd.read_csv(path, sep='\t').values.tolist()


def pack_data(name):
    train_path = os.path.join('./data', name, 'train.tsv')
    dev_path = os.path.join('./data', name, 'dev.tsv' if name != 'sst2' else 'test.tsv')
    train, eval = read_data(train_path), read_data(dev_path)
    data = {}
    data['train'] = [{'text': item[0], 'label': item[1]} for item in train]
    data['test'] = [{'text': item[0], 'label': item[1]} for item in eval]
    return data


def raw_dataset_dict(name):
    return pack_data(name)


def main():
    global plm

    train_loader, dev_loader, test_loader = prepare_clean_dataset(template)

    plm.load_state_dict(default_state_dict)
    print("Use standard language model.")

    plm = plm.to(device)
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=False)
    prompt_model = prompt_model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    train(prompt_model, optimizer, train_loader, dev_loader)
    acc = evaluate(test_loader, prompt_model)
    asr = evaluate_poison(poison_eval_loader_li, prompt_model)

    print("Acc: {}, ASR: {}".format(acc, asr))
    return acc, asr


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate ASR for Linkprompt")
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--dataset', default='ag_news',
                        choices=["ag_news", "imdb", "sst2", "twitter","fakereview", "fakenews"])
    parser.add_argument('--load_trigger', default=None, help="the path to the trigger JSON file.")
    parser.add_argument('--repeat', default=5, type=int, help="repeat evaluation to compute STD.")
    parser.add_argument('--bert_type', default='roberta-large',
                        choices=["bert-base-cased", "bert-large-cased", "roberta-base", "roberta-large"])
    parser.add_argument('--template_id', type=int, default=0, choices=[0, 1], help="choose the template. ")
    parser.add_argument('--gpu', default=0, type=int, help="gpu id.")

    params = parser.parse_args()

    device = torch.device("cuda:%d" % params.gpu)
    shots = params.shots
    dataset_name = params.dataset
    repeat_nums = params.repeat
    bert_type = params.bert_type
    template_id = params.template_id

    prompt_path = os.path.join('./prompt', dataset_name,
                               'manual_template.txt')
    prompt_li = []
    with open(prompt_path, 'r') as f:
        for line in f:
            prompt_li.append(line.strip())
    template_text = prompt_li[template_id]

    EPOCHS = 10
    raw_dataset = raw_dataset_dict(dataset_name)

    plm, tokenizer, model_config, WrapperClass = load_plm(bert_type.split('-')[0], bert_type)
    default_state_dict = plm.state_dict()

    verbalizer_dict = {'ag_news': ManualVerbalizer(tokenizer, num_classes=4,
                                                   label_words=[["politics"], ["sports"], ["business"],
                                                                ["technology"]]),
                       'sst2': ManualVerbalizer(tokenizer, classes=["negative", "positive"], num_classes=2,
                                                label_words={"negative": ["bad"],
                                                             "positive": ["good"], }),
                       'imdb': ManualVerbalizer(tokenizer, classes=["negative", "positive"], num_classes=2,
                                                label_words={"negative": ["bad"],
                                                             "positive": ["good"], }),
                       'twitter': ManualVerbalizer(tokenizer, num_classes=2,
                                                   label_words=[["harmless"], ["hate"]]),
                       'fakereview': ManualVerbalizer(tokenizer, num_classes=2,
                                                      label_words=[["real"], ["fake"]]),
                       'fakenews': ManualVerbalizer(tokenizer, num_classes=2,
                                                    label_words=[["real"],
                                                                 ["fake"]])}

    eval_set = 'test'
    discrip = 'text'
    verbalizer = verbalizer_dict[dataset_name]

    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

    tmp_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta/")
    with open(params.load_trigger, encoding='utf-8') as f:
        trigger_li = [tmp_tokenizer.convert_tokens_to_string(trigger_tokens)
                        for trigger_tokens in (json.load(f))["triggers"]]

    template_trigger_li = []
    for trigger in trigger_li:
        mask_index = template_text.find(mask_placeholder)
        text_index = template_text.find(text_placeholder)
        template_text_trigger = template_text.replace(text_placeholder, text_placeholder + " " + trigger)
        print(template_text_trigger)
        template_trigger = ManualTemplate(tokenizer=tokenizer, text=template_text_trigger)
        template_trigger_li.append(template_trigger)
    poison_eval_loader_li = prepare_poison_dataset_li(template_trigger_li)
    criterion = torch.nn.CrossEntropyLoss()

    acc_li = []
    asr_li = []
    for _ in range(repeat_nums):
        acc, asr = main()
        acc_li.append(acc * 100)
        asr_li.append(asr * 100)

    mean_acc, var_acc = np.mean(acc_li), np.var(acc_li)
    mean_asr, var_asr = np.mean(asr_li), np.var(asr_li)
    print("Acc: {}, Std: {}".format(mean_acc, sqrt(var_acc)))
    print("ASR: {}, Std: {}".format(mean_asr, sqrt(var_asr)))

    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    result_file_name = "results_{}.txt".format(params.dataset)
    result_file_path = os.path.join(result_dir, result_file_name)

    with open(result_file_path, "a") as result_file:
        result_file.write("para: dataset {}  template_id {} \n".format(params.dataset, params.template_id))
        result_file.write("current file:{}\n".format(params.load_trigger))
        result_file.write("Acc: {}, Std: {}\n".format(mean_acc, sqrt(var_acc)))
        result_file.write("ASR: {}, Std: {}\n".format(mean_asr, sqrt(var_asr)))

    print("Results saved to:", result_file_path)


