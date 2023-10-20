import json
import random
import os
import copy
import torch
import pickle

import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from utils.nest import get_key_map, get_keys, get_entity_type_desc
from collections import defaultdict

# For nest NER dataset helper function.
# data: {'tokens': [], 'entities': [{'start': start, 'end': end, 'type': type}]}

random.seed(7777)
rng = random.Random(7777)

recall_model = SentenceTransformer(
    "your_matching_model",
    device="cuda"
)

def get_topk(model, input, keys, topk=5, threshold=0.0):
    # 输入input文本（句子）和一系列实体标签词列表keys，输出keys中余弦相似度topk
    embedding = model.encode([input] + keys, convert_to_tensor=True, normalize_embeddings=True)
    input_embedding = embedding[0]
    keys_embedding = embedding[1:]
    similarities = F.cosine_similarity(
        input_embedding.unsqueeze(0),
        keys_embedding
    )
    topk_scores, topk_indices = torch.topk(similarities, topk)
    return [keys[topk_indices[i]] for i in range(len(topk_indices)) if topk_scores[i] >= threshold], [x for x in topk_scores if x >= threshold]

def add_sample_id(raw_json, raw_index):
    raw_json["raw_index"] = raw_index
    return raw_json

def ner_t5_data_format(fout, item, schemas):
    global recall_model

    entity_desc = [x + ": " + get_entity_type_desc('ACE2004')[x] for x in schemas]
    query, score = get_topk(recall_model, item["text"], entity_desc, len(entity_desc), 0.6)
    query = [x.split(": ")[0] for x in query]
    input = "List all named entities of the type [{}] and give explanations.\nText: {}\nEntities of type [{}] may exist in text".format(", ".join(schemas), item["text"], ", ".join(query))
    target = []
    for label in item["label"]:
        target.append("({}, {})".format(label["name"], label["value"]))

    target = ", ".join(target)
    target = "[" + target + "]"
    target = "Entity: " + target + "\n" + "Explanation: " + item["explanation"]
    item = {
        "input": input,
        "target": target,
        "labels": item["label"],
        "raw_index": item["raw_index"]
    }
    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

def auxiliary_t5_data_format(fout, item, schemas):
    input = "List all entity types in the text of the type [{}].\nText: {}".format(", ".join(schemas), item["text"])

    target = []
    for label in item["label"]:
        target.append("{}".format(label["name"]))
    target = list(set(target))
    target = ", ".join(target)
    target = "[" + target + "]"

    item = {
        "input": input,
        "target": target,
        "labels": item["label"],
        "raw_index": item["raw_index"]
    }
    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

def process_2_t5_type(data_path):
    sample_id = 0
    
    data_split = ["train", "validation", 'test'] # train need explanation for ToNER-EXP
    out_files = ["train_recall_0.6_cls_exp.json", "valid_recall_0.6_cls_exp.json", "test_recall_0.6_cls_exp.json"]
    schemas = get_keys(data_type)
    dataset = load_from_disk(data_path)

    for split, fn_out in zip(data_split, out_files):
        file_name = os.path.join(data_path, fn_out)
        try:
            with open(os.path.join(data_path, split), "r") as f:
                split_dataset = json.load(f)
        except:
            split_dataset = dataset[split]
        file_name = os.path.join(data_path, fn_out)
        f_out = open(file_name, 'w')
        pbar = tqdm(total=len(split_dataset))
        for data in split_dataset:
            label = set()
            for entity in data['entities']:
                label.add((get_key_map(data_type)[entity['type']], data['sentence'][entity['start']:entity['end']]))
            label = [{"name": x[0], "value": x[1]} for x in label]
            item = {"text": data['sentence'], "label": label, "explanation": data["explanation"] if "explanation" in data else ""}
            item = add_sample_id(item, sample_id)
            ner_t5_data_format(fout=f_out, item=item, schemas=schemas)
            p = random.random()
            if p < 0.5 and split == data_split[0]:
                auxiliary_t5_data_format(f_out, item, schemas)
            sample_id += 1
            pbar.update(1)
        f_out.close()
        pbar.close()


if __name__ == "__main__":
    data_type = "ACE2005"
    data_path = './data/{}'.format(data_type)

    process_2_t5_type(data_path)
