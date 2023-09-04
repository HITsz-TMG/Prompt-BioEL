import argparse
import json

import pandas as pd

from transformers import BertTokenizer



def read_kb(kb_path):
    with open(kb_path, encoding="utf-8") as f:
        data = json.loads(f.read())
    return data


def write_kb(data, path):
    pd.to_pickle(data, path)


def read_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def write_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d) + "\n" for d in data)


def process_kb(kb, args):
    tokenized_kb = []
    for key, value in kb.items():
        field = {}

        value = f" {tokenizer.sep_token} ".join(value)
        entity_dic = tokenizer.encode_plus(value,
                                           padding="max_length",
                                           max_length=args.max_ent_len,
                                           truncation=True)
        field["id"] = key
        field["text_ids"] = entity_dic["input_ids"]
        field["text_masks"] = entity_dic["attention_mask"]
        tokenized_kb.append(field)
        assert len(field["text_ids"]) == args.max_ent_len
    print("save kb...")
    write_kb(tokenized_kb, args.dataset + args.tokenized_kb)


def process_data(data_list, dtype, args):
    content_length = args.content_length
    max_pair_length = content_length // 2
    res = []

    start_token_id = tokenizer.convert_tokens_to_ids("[E1]")
    end_token_id = tokenizer.convert_tokens_to_ids("[/E1]")

    for data in data_list:
        text = data["text"]
        mention_data = data["mention_data"]
        for mention_info in mention_data:
            kb_id = mention_info["kb_id"]
            text_tokens = tokenizer.tokenize(text)
            text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
            start = text_ids.index(start_token_id)
            end = text_ids.index(end_token_id)
            text_ids = text_ids[max(0, start - max_pair_length):end + max_pair_length][:content_length - 2]
            text_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
            assert start_token_id in text_ids
            assert end_token_id in text_ids
            assert len(text_ids) <= content_length

            res.append({
                "entity_id": kb_id,
                "text": text_ids,
                "original_data": data
            })

    if dtype == "train":
        write_data(res, args.dataset + args.train_save_path)
    if dtype == "dev":
        write_data(res, args.dataset + args.dev_save_path)
    if dtype == "test":
        write_data(res, args.dataset + args.test_save_path)


def main(args):
    print("process kb...")
    process_kb(read_kb(args.dataset + args.kb_path),args)

    print("process train")
    train_data = read_data(args.dataset + args.train_data)
    print(len(train_data))
    process_data(train_data, "train", args)

    print("process dev")
    dev_data = read_data(args.dataset + args.dev_data)
    print(len(dev_data))
    process_data(dev_data, "dev", args)

    print("process test")
    test_data = read_data(args.dataset + args.test_data)
    print(len(test_data))
    process_data(test_data, "test", args)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    special_tokens = ["[E1]", "[/E1]", '[or]',"[NIL]"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/ncbi-disease/")
    parser.add_argument("--kb_path", default="entity_kb.json")
    parser.add_argument("--tokenized_kb", default="tokenized_kb.pkl")

    parser.add_argument("--max_ent_len", default=128,type=int)
    parser.add_argument("--content_length", default=256,type=int)
    parser.add_argument("--change_kb",default=True)

    parser.add_argument("--train_data", default="train.json")
    parser.add_argument("--dev_data", default="dev.json")
    parser.add_argument("--test_data", default="test.json")

    parser.add_argument("--train_save_path", default="disambiguation_input/train.json")
    parser.add_argument("--dev_save_path", default="disambiguation_input/dev.json")
    parser.add_argument("--test_save_path", default="disambiguation_input/test.json")

    args = parser.parse_args()
    main(args)
