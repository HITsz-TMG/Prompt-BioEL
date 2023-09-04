import argparse
import os
import random
import time
from datetime import datetime

# import numpy as np
# import torch
from tqdm import tqdm
from torch import nn

from data_retriver import *
from utils import Logger
from retriver import DualEncoder, SimpleEncoder
from transformers import BertTokenizer, BertModel, \
    get_linear_schedule_with_warmup, get_constant_schedule
from torch.optim import AdamW


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


def load_model(is_init, device, type_loss, tokenizer, args):
    if args.use_Dual_encoder:
        ctxt_bert = BertModel.from_pretrained(args.pretrained_model)
        cand_bert = BertModel.from_pretrained(args.pretrained_model)

        if is_init:
            model = DualEncoder(ctxt_bert, cand_bert, type_loss)
            model.entity_encoder.resize_token_embeddings(tokenizer.vocab_size + 10)
            model.mention_encoder.resize_token_embeddings(tokenizer.vocab_size + 10)
        else:
            state_dict = torch.load(args.model) if device.type == 'cuda' else \
                torch.load(args.model, map_location=torch.device('cpu'))
            model = DualEncoder(ctxt_bert, cand_bert, type_loss)
            model.entity_encoder.resize_token_embeddings(tokenizer.vocab_size + 10)
            model.mention_encoder.resize_token_embeddings(tokenizer.vocab_size + 10)
            model.load_state_dict(state_dict['sd'])
    else:
        bert = BertModel.from_pretrained(args.pretrained_model)
        if is_init:
            model = SimpleEncoder(bert, type_loss)
            model.encoder.resize_token_embeddings(tokenizer.vocab_size + 10)
        else:
            state_dict = torch.load(args.model) if device.type == 'cuda' else \
                torch.load(args.model, map_location=torch.device('cpu'))
            model = SimpleEncoder(bert, type_loss)
            model.encoder.resize_token_embeddings(tokenizer.vocab_size + 10)
            model.load_state_dict(state_dict['sd'])

    return model


def check_intersection(label, pre):
    label = set(label.split("|"))
    pre = set(pre.split("|"))
    return len(label.intersection(pre)) > 0


def evaluate(scores_k, top_k, labels, entity_map):
    nb_samples = len(labels)
    entities = list(entity_map.keys())
    num_hit = 0
    assert len(labels) == top_k.shape[0]
    for i in range(len(labels)):
        label = labels[i]
        pred = top_k[i]
        pred = [entities[j].split("_")[0] for j in pred]
        num_hit += any([check_intersection(label, p) for p in pred])
    return num_hit / nb_samples, 0, 0


def generate(samples_train, samples_val, samples_test, args):
    set_seeds(args)

    logger = Logger(args.model + '.log', on=True)
    logger.log(str(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.log(f'Using device: {str(device)}', force=True)
    entities = load_entities(args.dataset + args.kb_path)
    logger.log('number of entities {:d}'.format(len(entities)))

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    special_tokens = ["[E1]", "[/E1]", '[c]', "[NIL]"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    train_men_loader = get_mention_loader(samples_train, args.max_len, tokenizer, args.mention_bsz)
    val_men_loader = get_mention_loader(samples_val, args.max_len, tokenizer, args.mention_bsz)
    test_men_loader = get_mention_loader(samples_test, args.max_len, tokenizer, args.mention_bsz)
    entity_loader = get_entity_loader(entities, args.entity_bsz)

    entity_map = get_entity_map(entities)

    train_labels = get_labels(samples_train, entity_map)
    val_labels = get_labels(samples_val, entity_map)
    test_labels = get_labels(samples_test, entity_map)




    model = load_model(False, device, args.type_loss, tokenizer, args)

    model.to(device)
    save_optimal_result(samples_train, model, entity_loader, train_men_loader, entity_map, device, "train",
                        train_labels, args)
    save_optimal_result(samples_val, model, entity_loader, val_men_loader, entity_map, device, "dev",
                        val_labels, args)
    save_optimal_result(samples_test, model, entity_loader, test_men_loader, entity_map, device, "test",
                        test_labels, args)


def save_optimal_result(samples, model, entity_loader, mention_loader, entity_map,
                        device, data_type, labels, args):
    logger = Logger(args.model + '.log', on=True)
    all_cands_embeds = get_embeddings(entity_loader, model, False, device)
    mention_embeds = get_embeddings(mention_loader, model, True, device)

    if data_type == "train":
        print("save train...")
        top_k, scores_k = get_hard_negative(mention_embeds, all_cands_embeds,
                                            args.dev_cand, 0, args.use_gpu_index)
        save_candidates(samples, top_k, entity_map, labels,
                        args.dataset + args.disambiguation_train_output_file, data_type)
    elif data_type == "dev":
        print("save dev...")
        top_k, scores_k = get_hard_negative(mention_embeds, all_cands_embeds,
                                            args.dev_cand, 0, args.use_gpu_index)
        eval_result = evaluate(scores_k, top_k, labels, entity_map)
        logger.log(f"dev evaluate: recall@{args.dev_cand}={eval_result[0]}")
        save_candidates(samples, top_k, entity_map, labels,
                        args.dataset + args.disambiguation_dev_output_file, data_type)
    else:
        print("save test...")
        top_k, scores_k = get_hard_negative(mention_embeds, all_cands_embeds,
                                            1, 0, args.use_gpu_index)
        eval_result = evaluate(scores_k, top_k, labels, entity_map)

        logger.log(f"test evaluate: recall@1={eval_result[0]}")
        top_k, scores_k = get_hard_negative(mention_embeds, all_cands_embeds,
                                            args.dev_cand, 0, args.use_gpu_index)
        eval_result = evaluate(scores_k, top_k, labels, entity_map)
        logger.log(f"test evaluate: recall@{args.dev_cand}={eval_result[0]}")
        save_candidates(samples, top_k, entity_map, labels,
                        args.dataset + args.disambiguation_test_output_file, data_type)


def main(args):
    train_data = load_data(args.dataset + args.train_data)
    dev_data = load_data(args.dataset + args.dev_data)
    test_data = load_data(args.dataset + args.test_data)

    generate(train_data, dev_data, test_data, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default="dataset/ncbi-disease/")
    parser.add_argument('--model',
                        default="model_retriever/ncbi_retriever.pt",
                        help='model path')
    parser.add_argument("--pretrained_model",
                        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    parser.add_argument('--type_loss', type=str,
                        default="sum_log_nce",
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of multi-label loss ?')

    parser.add_argument('--max_len', type=int, default=256,
                        help='max length of the mention input ')
    parser.add_argument("--use_Dual_encoder", default=False)
    parser.add_argument("--train_data", default="disambiguation_input/train.json")
    parser.add_argument("--dev_data", default="disambiguation_input/dev.json")
    parser.add_argument("--test_data", default="disambiguation_input/test.json")
    parser.add_argument("--disambiguation_dev_output_file", default="disambiguation_output/dev.json")
    parser.add_argument("--disambiguation_test_output_file", default="disambiguation_output/test.json")
    parser.add_argument("--disambiguation_train_output_file", default="disambiguation_output/train.json")
    parser.add_argument('--kb_path', type=str,
                        default="tokenized_kb.pkl",
                        help='the knowledge base directory')
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument('--B', type=int, default=2,
                        help='the batch size per gpu')

    parser.add_argument("--dev_cand", default=6, type=int)

    parser.add_argument('--gpus', default='3', type=str,
                        help='GPUs separated by comma [%(default)s]')

    parser.add_argument('--mention_bsz', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--entity_bsz', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--use_gpu_index', default=True,
                        help='use gpu index?')

    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior
    main(args)
