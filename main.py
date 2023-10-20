import os
import argparse
import torch
import re

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm

from utils.tag_map import get_tag_map, get_keys

MAX_TEXT_LENGTH = 512
MODEL_PATH = "./flan-t5-xl" # backbone model path 
DATA_TYPE = "ACE2005" # dataset type
tag_map = get_tag_map(DATA_TYPE.split("/")[-1])
KEYS = get_keys(DATA_TYPE.split("/")[-1])


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_train_dataset(file_name):
    cur_dataset = load_dataset("json", data_files=[file_name], split="train").shuffle(seed=2023)
    cur_dataset = cur_dataset.map(
        prepare_features,
        batched=True,
        # num_proc=100
    )
    return cur_dataset

def get_val_dataset(file_name):
    cur_dataset = load_dataset("json", data_files=[file_name], split="train")
    cur_dataset = cur_dataset.map(
        prepare_features,
        batched=True,
        # num_proc=100
    )
    return cur_dataset

def prepare_features(example):
    global MODEL_PATH, DATA_TYPE
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    features = {
        "source_ids": [], 
        "source_mask": [], 
        "target_ids": [], 
        "cls_label": []
    }
    total = len(example["input"])

    for i in range(total):
        # features["raw_index"].append(example["raw_index"][i])

        source_text = example['input'][i]
        source = tokenizer(
            source_text,
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
        )
        source_raw = tokenizer(source_text)
        if len(source_raw["input_ids"]) > MAX_TEXT_LENGTH:
            raise Exception()

        # if len(source_raw["input_ids"]) > MAX_TEXT_LENGTH:
        #     raise Exception()

        target_text = example['target'][i]
        target = tokenizer(
            target_text,
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
        )

        # if len(target_raw["input_ids"]) > MAX_TEXT_LENGTH:
        #     raise Exception()

        features["source_ids"].append(source["input_ids"])
        features["source_mask"].append(source["attention_mask"])
        features["target_ids"].append(target["input_ids"])

        labels = set([x["name"] for x in example["labels"][i]])
        features["cls_label"].append([1 if key in labels else 0 for key in get_keys(DATA_TYPE)])
    return features

def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids

def train_collate_fn(batch):
    max_len = max([len(d['source_ids']) for d in batch])
    max_target_len = max([len(d['target_ids']) for d in batch])

    source_ids, source_mask, target_ids, cls_label = [], [], [], []
    for item in batch:
        source_ids.append(pad_to_maxlen(item['source_ids'], max_len=max_len))
        source_mask.append(pad_to_maxlen(item['source_mask'], max_len=max_len))
        target_ids.append(pad_to_maxlen(item['target_ids'], max_len=max_target_len))
        cls_label.append(item['cls_label'])
        # raw_index.append(item['raw_index'])

    source_ids = torch.tensor(source_ids, dtype=torch.long)
    source_mask = torch.tensor(source_mask, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)
    cls_label = torch.tensor(cls_label, dtype=torch.long)

    return {
        "source_ids": source_ids,
        "source_mask": source_mask,
        "target_ids": target_ids,
        "cls_label": cls_label
    }

def test_collate_fn(batch):
    max_len = max([len(d['source_ids']) for d in batch])
    max_target_len = MAX_TEXT_LENGTH

    source_ids, source_mask, target_ids, cls_label = [], [], [], []
    for item in batch:
        source_ids.append(pad_to_maxlen(item['source_ids'], max_len=max_len))
        source_mask.append(pad_to_maxlen(item['source_mask'], max_len=max_len))
        target_ids.append(pad_to_maxlen(item['target_ids'], max_len=max_target_len))
        # raw_index.append(item['raw_index'])

    source_ids = torch.tensor(source_ids, dtype=torch.long)
    source_mask = torch.tensor(source_mask, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    return {
        "source_ids": source_ids,
        "source_mask": source_mask,
        "target_ids": target_ids,
    }

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    mean_state = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return mean_state

def compute_cls_loss(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    cls_loss = 0.1 * (neg_loss + pos_loss).mean(dim=-1)
    return cls_loss

def train(args, epoch, tokenizer, model, progress_bar, train_dataloader, optimizer, lr_scheduler, accelerator):
    model.train()
    accelerator.wait_for_everyone()
    for index, data in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        labels = data["target_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        input_ids = data["source_ids"]
        attention_mask = data["source_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = outputs["loss"]

        cls_embedding = average_pool(outputs["encoder_last_hidden_state"], attention_mask)
        cls_loss = compute_cls_loss(model.linear_layer(cls_embedding), data["cls_label"])
        if index % 100 == 0 and index != 0 and accelerator.is_local_main_process:
            print(
                index, "epoch:" + str(epoch) + "-loss1:" + str(loss) + "-cls_loss:" + str(cls_loss)
            )
        
        accelerator.backward(loss + cls_loss)
        # accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

def validate(args, tokenizer, model, val_dataloader, accelerator):
    progress_bar = tqdm(range(len(val_dataloader)), disable=not accelerator.is_local_main_process)
    accelerator.wait_for_everyone()
    nb_correct, nb_pred, nb_label = 0, 0, 0
    inputs, outputs, targets = [], [], []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            y = data['target_ids'].to(accelerator.device)
            ids = data['source_ids'].to(accelerator.device)
            mask = data['source_mask'].to(accelerator.device)

            generated_ids = accelerator.unwrap_model(model).generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=MAX_TEXT_LENGTH // 2,
                synced_gpus=True
            )

            generated_ids = accelerator.pad_across_processes(generated_ids, dim=1)
            generated_ids = accelerator.gather_for_metrics(generated_ids)
            y = accelerator.gather_for_metrics(y)
            ids = accelerator.pad_across_processes(ids, dim=1)
            ids = accelerator.gather_for_metrics(ids)

            if accelerator.is_local_main_process:
                input = tokenizer.batch_decode(ids, skip_special_tokens=True)
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                target = tokenizer.batch_decode(y, skip_special_tokens=True)

                inputs.extend(input)
                outputs.extend(output)
                targets.extend(target)
            progress_bar.update(1)
    
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        for i, o, t in tqdm(zip(inputs, outputs, targets)):
            # For ToNER, use this:
            preds = split2pair(o, i)
            targets = split2pair(t, i)

            # For ToNER-EXP, use this:
            # preds = split2pair_exp(o, i)
            # targets = split2pair_exp(t, i)

            nb_label += len(targets)
            nb_pred += len(preds)
            for k, v in preds:
                if (k, v) in targets:
                    nb_correct += 1
    
    accelerator.wait_for_everyone()
    precision = nb_correct / nb_pred if nb_pred else 0
    recall = nb_correct / nb_label if nb_label else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if accelerator.is_local_main_process:
        print(f"result: nb_pred: {nb_pred} nb_label: {nb_label} nb_correct: {nb_correct}\n")
        print("result: precision: {:.5f} recall: {:.5f} f1: {:.5f}\n".format(precision, recall, f1))
    return f1

def split2pair(answer, text):
    res = []
    answer = answer[1:-1]
    left, right = 0, 0
    while right < len(answer):
        if answer[right] == "(":
            left = right + 1
        elif answer[right] == ")":
            comma_index = answer[left:right].find(',')
            t = [answer[left:right][:comma_index].strip(" "), answer[left:right][comma_index+1:].strip(" ")]
            res.append((t[0], t[1]))
        right += 1
    return list(set(res))

def split2pair_exp(answer, text):
    end = answer.find("Explanation: ")
    answer = answer[8:end-1]
    res = []
    answer = answer[1:-1]
    left, right = 0, 0
    while right < len(answer):
        if answer[right] == "(":
            left = right + 1
        elif answer[right] == ")":
            comma_index = answer[left:right].find(',')
            t = [answer[left:right][:comma_index].strip(" "), answer[left:right][comma_index+1:].strip(" ")]
            res.append((t[0], t[1]))
        right += 1
    return list(set(res))

def save_model(args, model, accelerator, tokenizer):
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        print("[Saving Model]...\n")
        os.makedirs(args.output_path, exist_ok=True)
        accelerator.unwrap_model(model).config.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), os.path.join(args.output_path, "pytorch_model.bin"))

    accelerator.wait_for_everyone()
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--data_type", type=str, default=DATA_TYPE)
    parser.add_argument("--output_path", type=str, default="./{}/{}/".format(DATA_TYPE, MODEL_PATH.split("/")[-1]))
    parser.add_argument("--train_data_path", type=str, default=f"./data/{DATA_TYPE}/train.json")
    parser.add_argument("--val_data_path", type=str, default=f"./data/{DATA_TYPE}/test.json")
    
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4396)

    args = parser.parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(split_batches=True)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.resize_token_embeddings(len(tokenizer))
    device = accelerator.device
    model.to(device)
    print("Trans to: ", device)

    linear_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, model.config.hidden_size),
        nn.Tanh(),
        nn.Linear(model.config.hidden_size, len(get_keys(args.data_type)))
    )
    linear_layer.to(device)
    model.add_module("linear_layer", linear_layer)


    train_dataset = get_train_dataset(args.train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn)
    total_train_steps = int((len(train_dataset) * args.epoch) / (args.batch_size))

    val_dataset = get_val_dataset(args.val_data_path)
    val_dataloader = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, collate_fn=test_collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_train_steps), num_training_steps=total_train_steps
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    val_dataloader = accelerator.prepare(val_dataloader)

    progress_bar = tqdm(range(total_train_steps), disable=not accelerator.is_local_main_process)

    best_f1 = 0
    for epoch in range(args.epoch):
        train(args, epoch, tokenizer, model, progress_bar, train_dataloader, optimizer, lr_scheduler, accelerator)

        accelerator.wait_for_everyone()
        if epoch < 10: continue
        if accelerator.is_local_main_process:
            print("Validation...")
        f1 = validate(args, tokenizer, model, val_dataloader, accelerator)
        if f1 > best_f1:
            best_f1 = f1
            accelerator.print('[Saving Model]...')
            save_model(args, model, accelerator, tokenizer)

