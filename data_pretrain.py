import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM
from torch import nn
import json


class PretrainDataset(Dataset):
    def __init__(self, tokenizer, kb, max_length):
        self.tokenizer = tokenizer
        self.kb = kb
        self.or_token = "[or]"
        self.max_length = max_length

    def __len__(self):
        return len(self.kb.keys())

    def __getitem__(self, index):
        max_length = self.max_length - 2
        keys = list(self.kb.keys())
        key = keys[index]
        entities = self.kb[key]
        random.shuffle(entities)
        entity_combined_text = []
        mask_index_list = []
        label_tokens = []

        for entity in entities:
            entity_token = self.tokenizer.tokenize(entity)
            if len(entity_combined_text) + len(entity_token) <= max_length:
                mask_index = random.choice(list(range(len(entity_token))))
                label_tokens.append(entity_token[mask_index])
                mask_index_list.append(len(entity_combined_text) + mask_index)
                entity_token[mask_index] = self.tokenizer.mask_token
                entity_combined_text += (entity_token + [self.or_token])
        entity_combined_text = [self.tokenizer.cls_token] + entity_combined_text + [self.tokenizer.sep_token]
        mask_index_list = [index + 1 for index in mask_index_list]
        input_ids = self.tokenizer.convert_tokens_to_ids(entity_combined_text)
        attention_mask = [1] * len(input_ids)
        label_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)
        weights = [1] * len(label_ids)
        return {"input_ids": input_ids, "mask_index": mask_index_list, "label_ids": label_ids,
                "attention_mask": attention_mask, "weights": weights}


def collate_fn(batch):
    def pad_sequence(inputs, pad_value):
        max_length = max([len(input) for input in inputs])
        outputs = [(input + [pad_value] * max_length)[:max_length] for input in inputs]
        return outputs
    def pad_index(inputs):
        max_length = max([len(input) for input in inputs])
        outputs = [get_useless_index(input,max_length) for input in inputs]
        for output in outputs:
            assert len(output) == max_length
        return outputs
    def get_useless_index(input,max_len):
        while(len(input) < max_len):
            random_index = random.choice(range(max_index))
            if random_index not in input:
                input.append(random_index)
        return input



    input_ids = [b["input_ids"] for b in batch]

    attention_mask = [b["attention_mask"] for b in batch]
    mask_index = [b["mask_index"] for b in batch]

    max_index = max(index for mask_in in mask_index for index in mask_in)
    label_ids = [b["label_ids"] for b in batch]
    weights = [b["weights"] for b in batch]
    assert sum([len(label_id) for label_id in label_ids]) == sum([sum(weight) for weight in weights])
    input_ids = torch.tensor(pad_sequence(input_ids, 0)).long()
    attention_mask = torch.tensor(pad_sequence(attention_mask, 0)).long()
    mask_index = torch.tensor(pad_index(mask_index)).long()
    label_ids = torch.tensor([label for label_id in label_ids for label in label_id ]).long()

    weights = torch.tensor(pad_sequence(weights, 0)).long()
    assert label_ids.shape[0] == torch.sum(weights)
    return input_ids,attention_mask,mask_index,label_ids,weights

def load_entities(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = json.loads(f.read())
    return data

def make_single_loader(data_set, bsz, shuffle, coll_fn=None):
    if coll_fn is not None:
        loader = DataLoader(data_set, bsz, shuffle=shuffle, collate_fn=coll_fn)
    else:
        loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader

def get_mention_loader(tokenizer,kb,args):
    sample_set = PretrainDataset(tokenizer,kb,args.max_length)
    return make_single_loader(sample_set,args.batch,True,coll_fn=collate_fn)