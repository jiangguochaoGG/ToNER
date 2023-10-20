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
from utils.flat import get_tag_map, get_entity_type_desc
from collections import defaultdict

# For flat NER dataset helper function.
# data: {'tokens': [], 'tags' | 'ner_tags': []}

random.seed(7777)
rng = random.Random(7777)
data_type = 'conll2003' # dataset name

recall_model = SentenceTransformer(
    "your_matching_model",
    device="cuda"
)

def get_topk(model, input, keys, topk=5, threshold=0.0):
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

def ner_t5_data_format(fout, item, schemas, mode='train'):
    global recall_model
    entity_desc = [x + ": " + get_entity_type_desc(data_type)[x] for x in schemas]
    query, score = get_topk(recall_model, item["text"], entity_desc, len(entity_desc), 0.95)
    query = [x.split(": ")[0] for x in query]
    input = "List all named entities of the type [{}].\nText: {}\nEntities of type [{}] may exist in text".format(", ".join(schemas), item["text"], ", ".join(query))
    target = []
    for label in item["label"]:
        target.append("({}, {})".format(label["name"], label["value"]))

    target = ", ".join(target)
    target = "[" + target + "]"

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

def process_2_t5_type(data_path, tag_map):
    global pos, neg
    sample_id = 0
    
    data_split = ['train', "validation", 'test'] # input_data_split
    out_files = ['train_recall_0.95_cls.json', "valid_recall_0.95_cls.json", 'test_recall_0.95_cls.json'] # output data name
    try:
        schemas = list(set([tag_map[x][tag_map[x].index("-") + 1:].lower() for x in tag_map if x != 0]))
    except:
        schemas = list(set([tag_map[x].lower() for x in tag_map if x != 0]))

    dataset = load_from_disk(data_path)

    for split, fn_out in zip(data_split, out_files):
        split_dataset = dataset[split]
        file_name = os.path.join(data_path, fn_out)
        f_out = open(file_name, 'w')
        pbar = tqdm(total=len(split_dataset))
        for data in split_dataset:
            label = set()
            begin = 0
            try:
                tags = data["ner_tags"]
            except:
                tags = data["tags"]
            while begin < len(tags):
                while begin < len(tags) and tags[begin] == 0: begin += 1
                if begin < len(tags):
                    end = begin + 1
                    if end < len(tags):
                        try:
                            name = tag_map[tags[begin]][tag_map[tags[begin]].index("-") + 1:].lower()
                        except:
                            name = tag_map[tags[begin]].lower()
                        try:
                            end_name = tag_map[tags[end]][tag_map[tags[end]].index("-") + 1:].lower()
                        except:
                            end_name = tag_map[tags[end]].lower()
                        while end < len(tags) and tags[end] != 0 and end_name == name:
                            end += 1
                    entity = " ".join(data["tokens"][begin:end])
                    label.add((name, entity))
                    begin = end
            label = [{"name": x[0], "value": x[1]} for x in label]
            item = {"text": " ".join(data["tokens"]), "label": label}
            item = add_sample_id(item, sample_id)
            ner_t5_data_format(fout=f_out, item=item, schemas=schemas, mode=split)
            p = random.random()
            if p < 0.5 and (split == 'train' or split == 'processed_conll2003_with_explanation.json'):
                auxiliary_t5_data_format(f_out, item, schemas)
            sample_id += 1
            pbar.update(1)
        f_out.close()
        pbar.close()


if __name__ == "__main__":
    data_path = './data/{}'.format(data_type)
    tag_map = get_tag_map(data_type)

    process_2_t5_type(data_path, tag_map)
