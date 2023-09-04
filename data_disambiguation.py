import random
import json
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm



class PromptDataset(Dataset):
    def __init__(self,mentions, kb, tokenizer,is_test, args):
        self.mentions = mentions
        self.kb = kb
        self.tokenizer = tokenizer
        self.cand_num = args.cand_num
        self.max_ent_len = args.max_ent_len
        self.max_text_len = args.max_text_len
        self.max_length = args.max_len
        self.or_token = " [or] "
        self.is_test = is_test


    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        data = self.mentions[index]

        text = data["text"]
        choice_label = [self.tokenizer.convert_tokens_to_ids(f"[{i}]")  for i in range(self.cand_num)]
        mention_data = data["mention_data"]
        mention = mention_data["mention"]
        pattern = f"which of the following options is the same as {mention}?"
        # pattern = f"which of the following options is the same as [E1] {mention} [\E1]?"
        # pattern = " ".join([f"[{i}]" for i in range(self.cand_num)]) + f" {mention}?"
        # pattern = mention
        # pattern = ""

        candidates = mention_data["candidates"][:self.cand_num]
        labels = mention_data["labels"][:self.cand_num]
        can_la = [(can, la) for can, la in zip(candidates, labels)]
        mention = mention_data["mention"]
        mention_token = self.tokenizer.tokenize(mention)
        if not self.is_test:
            random.shuffle(can_la)

        labels = [c_l[1] for c_l in can_la]

        max_half_text = self.max_text_len // 2 - 1
        max_ent_len = self.max_ent_len - 1

        text_tokens = self.tokenizer.tokenize(text)
        pattern_tokens = self.tokenizer.tokenize(pattern)
        men_start = text_tokens.index("[E1]")
        men_end = text_tokens.index("[/E1]")
        text_tokens = [self.tokenizer.cls_token] + text_tokens[
                                                   max(0, men_start - max_half_text):men_end + max_half_text] \
                      + pattern_tokens
        # text_tokens = text_tokens + pattern_tokens + [self.tokenizer.sep_token]
        ans_pos = len(text_tokens)
        text_tokens += ([self.tokenizer.mask_token] + [self.tokenizer.sep_token])

        for i in range(len(labels)):
            entity_id, _ = can_la[i]
            entity_names = self.kb[entity_id]

            entity_names = sorted(entity_names, key=lambda x: self.longest_subsequence(mention_token, x), reverse=True)
            entity_names = self.or_token.join(entity_names)

            entity_tokens = self.tokenizer.tokenize(entity_names)
            entity_tokens = [f"[{i}]"] + entity_tokens[:max_ent_len] + [self.tokenizer.sep_token]

            text_tokens += entity_tokens

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        attention_masks = [1] * len(text_token_ids)
        attention_masks = self.pad_values(attention_masks, 0, self.max_length)
        text_token_ids = self.pad_values(text_token_ids, self.tokenizer.pad_token_id, self.max_length)
        text_token_ids = torch.tensor(text_token_ids).long()
        attention_masks = torch.tensor(attention_masks).long()
        labels = torch.tensor(labels).float()
        choice_label = torch.tensor(choice_label).long()
        ans_pos = torch.tensor(ans_pos).long()

        return text_token_ids, attention_masks, ans_pos, choice_label , labels

    def pad_values(self, tokens, value, max_len):
        return (tokens + [value] * max_len)[:max_len]



    def longest_subsequence(self, mention_token, entity_name):
        entity_token = self.tokenizer.tokenize(entity_name)
        men_len = len(mention_token)
        en_len = len(entity_token)
        dp = [[0] * (en_len + 1) for _ in range(men_len + 1)]
        for i in range(men_len):
            for j in range(en_len):
                if mention_token[i] == entity_token[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[men_len][en_len] / en_len




def generate_samples(batch):
    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        input_ids += b["input_ids"]
        attention_masks += b["attention_masks"]
        labels += b["labels"]
    input_ids = torch.tensor(input_ids).long()
    attention_masks = torch.tensor(attention_masks).long()
    labels = torch.tensor(labels).float()
    return input_ids, attention_masks, labels


def load_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


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




def get_prompt_mention_loader(samples, kb, tokenizer, shuffle, is_test, args):
    samples_set = PromptDataset(samples, kb, tokenizer, is_test,args)
    return make_single_loader(samples_set, args.batch, shuffle)






def save_prompt_predict_test(model, samples, kb, tokenizer, device, args):
    samples_set = PromptDataset(samples, kb, tokenizer, True, args)
    sample_loader = make_single_loader(samples_set, 1, False)
    data_loader = tqdm(sample_loader)
    for step, batch in enumerate(data_loader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, ans_pos, choice_label, label = batch
        score = model(input_ids, attention_mask, ans_pos, choice_label, label, "val").tolist()
        samples[step]["mention_data"]["score"] = score
    with open(args.dataset + "test_res.json", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in samples)