import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import faiss
from utils import sample_range_excluding


class EntitySet(Dataset):
    def __init__(self, entities):
        self.entities = entities

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        entity = self.entities[index]
        entity_token_ids = torch.tensor(entity['text_ids']).long()
        entity_masks = torch.tensor(entity['text_masks']).long()
        return entity_token_ids, entity_masks


class MentionSet(Dataset):
    def __init__(self, mentions, max_len, tokenizer):
        self.mentions = mentions
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        mention = self.mentions[index]
        mention_ids_raw = mention["text"]
        mention_ids = (mention_ids_raw + [self.tokenizer.pad_token_id] * (
                self.max_len - len(mention_ids_raw)))[:self.max_len]
        mention_masks = ([1] * len(mention_ids_raw) + [0] * (
                self.max_len - len(mention_ids_raw)))[:self.max_len]
        mention_token_ids = torch.tensor(mention_ids).long()
        mention_masks = torch.tensor(mention_masks).long()
        return mention_token_ids, mention_masks


class RetrievalSet(Dataset):
    def __init__(self, mentions, entities, labels, max_len,
                 tokenizer, candidates,
                 num_cands, rands_ratio, type_loss):
        self.mentions = mentions
        self.candidates = candidates
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.labels = labels
        self.num_cands = num_cands
        self.rands_ratio = rands_ratio
        self.all_entity_token_ids = np.array([e['text_ids'] for e in entities])
        self.all_entity_masks = np.array([e['text_masks'] for e in entities])
        self.all_entity_kb_ids = [e["id"].split("_")[0] for e in entities]
        self.entities = entities
        self.type_loss = type_loss

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        mention = self.mentions[index]
        mention_ids_raw = mention["text"]
        mention_ids = (mention_ids_raw + [self.tokenizer.pad_token_id] * (
                self.max_len - len(mention_ids_raw)))[:self.max_len]
        mention_masks = ([1] * len(mention_ids_raw) + [0] * (
                self.max_len - len(mention_ids_raw)))[:self.max_len]
        mention_token_ids = torch.tensor(mention_ids).long()
        mention_masks = torch.tensor(mention_masks).long()

        cand_ids = []
        label = self.labels[index]
        labels = self.get_golden_labels(label)[:self.num_cands]
        if len(labels) == 0:
            labels = [-1]
        else:
            labels = list(set(labels))
        cand_ids += labels
        num_pos = len(labels)
        # assert num_pos >= 0
        num_neg = self.num_cands - num_pos
        assert num_neg >= 0
        num_rands = int(self.rands_ratio * num_neg)
        num_hards = num_neg - num_rands
        # non-hard and non-label for random negatives
        if self.candidates is not None:
            rand_cands = sample_range_excluding(len(self.entities), num_rands,
                                                set(labels).union(set(
                                                    self.candidates[index])))
        else:
            rand_cands = sample_range_excluding(len(self.entities), num_rands,
                                                set(labels))
        cand_ids += rand_cands
        # process hard negatives
        if self.candidates is not None:
            # hard negatives
            hard_negs = random.sample(list(set(self.candidates[index]) - set(
                labels)), num_hards)
            cand_ids += hard_negs

        passage_labels = torch.tensor([1] * num_pos + [0] * num_neg).long()
        candidate_token_ids = self.all_entity_token_ids[cand_ids].tolist()
        candidate_masks = self.all_entity_masks[cand_ids].tolist()
        assert passage_labels.size(0) == self.num_cands
        candidate_token_ids = torch.tensor(candidate_token_ids).long()
        assert candidate_token_ids.size(0) == self.num_cands
        candidate_masks = torch.tensor(candidate_masks).long()
        return mention_token_ids, mention_masks, candidate_token_ids, \
               candidate_masks, passage_labels

    def get_golden_labels(self, entity_id):
        golden_ids = []
        for i, id in enumerate(self.all_entity_kb_ids):
            if len(set(entity_id.split("|")).intersection(set(id.split("|")))) > 0:
                golden_ids.append(i)
        return golden_ids


def load_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def load_entities(data_path):
    return pd.read_pickle(data_path)


def write_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in data)


def make_single_loader(data_set, bsz, shuffle):
    loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader


def get_mention_loader(samples, max_len, tokenizer, mention_bz):
    sample_set = MentionSet(samples, max_len, tokenizer)
    return make_single_loader(sample_set, mention_bz, False)


def get_entity_loader(entity_samples, entity_bz):
    entity_sample_set = EntitySet(entity_samples)
    return make_single_loader(entity_sample_set, entity_bz, False)


def get_entity_map(entities):
    entity_map = {}

    for i, e in enumerate(entities):
        entity_map[e["id"]] = i

    assert len(entity_map) == len(entities)
    return entity_map


def get_labels(samples, all_entity_map):
    labels = [sample["entity_id"] for sample in samples]
    # labels = [all_entity_map[label] for label in labels]
    return labels


def get_embeddings(loader, model, is_mention, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks = batch
            k1, k2 = ("mention_token_ids", "mention_masks") if is_mention else \
                ("entity_token_ids", "entity_masks")
            kwargs = {k1: input_ids, k2: input_masks}
            j = 0 if is_mention else 2
            embed = model(**kwargs)[j].detach()
            embeddings.append(embed.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    model.train()
    return embeddings


def get_hard_negative(mention_embeddings, all_entity_embeds, k,
                      max_num_postives,
                      use_gpu_index=False):
    index = faiss.IndexFlatIP(all_entity_embeds.shape[1])
    if use_gpu_index:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(all_entity_embeds)
    scores, hard_indices = index.search(mention_embeddings,
                                        k + max_num_postives)
    del mention_embeddings
    del index
    return hard_indices, scores


def get_loader_from_candidates(samples, entities, labels, max_len,
                               tokenizer, candidates,
                               num_cands, rands_ratio, type_loss,
                               shuffle, bsz):
    data_set = RetrievalSet(samples, entities, labels,
                            max_len, tokenizer, candidates,
                            num_cands, rands_ratio, type_loss)
    loader = make_single_loader(data_set, bsz, shuffle)
    return loader


def check_candidates(candidates, label):
    can_label = []
    for candidate in candidates:
        can_label.append((int(len(set(candidate.split("|")).intersection(set(label.split("|")))) > 0)))
    return can_label


def save_candidates(samples, candidates, entity_map, labels, out_file, part):
    assert len(samples) == len(candidates)

    res = []
    entity_ids = list(entity_map.keys())
    for i in range(len(samples)):
        label = labels[i]
        # text = samples[i]["text"]
        oringinal_data = samples[i]["original_data"]
        text = oringinal_data["text"]
        mention_data = oringinal_data["mention_data"][0]
        mention = mention_data["mention"]
        golden = mention_data["kb_id"]
        m_candidates = candidates[i].tolist()
        m_candidates = [entity_ids[j] for j in m_candidates]
        candidates_labels = check_candidates(m_candidates, label)
        if part == "train":
            if not any(candidates_labels):
                if label not in entity_ids:
                    label = get_hit_label(entity_ids, label)[0]

                else:
                    label = label
                m_candidates[-1] = label
                candidates_labels[-1] = 1
                assert any(candidates_labels)

            res.append({"text": text, "mention_data": {"mention": mention, "kb_id": golden, "candidates": m_candidates,
                                                       "labels": candidates_labels}})

        else:
            res.append({"text": text, "mention_data": {"mention": mention, "kb_id": golden, "candidates": m_candidates,
                                                       "labels": candidates_labels}})

    write_data(res, out_file)


def get_hit_label(entities, label):
    hit_labels = []
    for entity in entities:
        if len(set(entity.split("|")).intersection(label.split("|"))) > 0:
            hit_labels.append(entity)
    random.shuffle(hit_labels)
    return hit_labels
