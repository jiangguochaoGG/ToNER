import random
import torch

import torch.nn.functional as F

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import Dataset, DatasetDict, load_from_disk
from uniem.finetuner import FineTuner
from collections import defaultdict

from utils.nest import get_key_map, get_keys, get_entity_type_desc

MODEL_PATH = "./gte-large" # thenlper/gte-large
DATA_PATH = "./data/ACE2005"
random.seed(7777)

def process(raw_data, mode="train", data_type="WNUT2017"):
    key_map = get_key_map(data_type)
    schemas = get_keys(data_type)
    entity_type_desc = get_entity_type_desc(data_type)

    data = []
    for line in raw_data:
        sentence = " ".join(line["tokens"])
        pos_name, neg_name = [], []
        try:
            labels = set([key_map[x['type']] for x in line['entities']])
        except:
            labels = set([x['type'] for x in line['entities']])
        for name in labels:
            if name not in pos_name:
                pos_name.append(name)
        for name in schemas:
            if name not in pos_name and name not in neg_name:
                neg_name.append(name)
        if mode == "train":
            # TripletRecord
            for pos in pos_name:
                for neg in neg_name:
                    data.append({
                        "text": sentence,
                        "text_pos": pos + ": " + entity_type_desc[pos],
                        "text_neg": neg + ": " + entity_type_desc[neg]
                    })
        else:
            data.append({
                "input": sentence,
                "label": labels
            })
    return data

def load_train_data(data_path):
    dataset = {}
    data_type = data_path.split("/")[-1]
    for key in ["train", "validation"]:
        raw_data = []
        train_dataset = load_from_disk(data_path)[key]
        for data in train_dataset:
            raw_data.append(data)
        dataset[key] = Dataset.from_list(process(raw_data, "train", data_type))
    return DatasetDict(dataset)

def get_topk(model, input, keys, topk):
    embedding = model.encode([input] + keys, convert_to_tensor=True, normalize_embeddings=True)
    input_embedding = embedding[0]
    keys_embedding = embedding[1:]
    similarities = F.cosine_similarity(
        input_embedding.unsqueeze(0),
        keys_embedding
    )
    _, topk_indices = torch.topk(similarities, topk)
    return [keys[i] for i in topk_indices]

if __name__ == "__main__":
    finetuner = FineTuner.from_pretrained(MODEL_PATH, dataset=load_train_data(DATA_PATH))
    finetuner.run(
        epochs=1, 
        output_dir=f"./tmp/",
        batch_size=32,
        shuffle=True
    )