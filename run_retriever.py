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
from transformers import BertTokenizer, BertModel ,\
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


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)

    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = 0

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


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


def train(samples_train, samples_val, samples_test, args):
    set_seeds(args)
    best_val_perf = float('-inf')
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

    max_num_positives = args.k - args.num_cands

    model = load_model(True, device, args.type_loss, tokenizer, args)

    num_train_samples = len(samples_train)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, num_train_samples)

    if args.resume_training:
        cpt = torch.load(args.model) if device.type == 'cuda' \
            else torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(cpt['sd'])
        optimizer.load_state_dict(cpt['opt_sd'])
        scheduler.load_state_dict(cpt['scheduler_sd'])
        best_val_perf = cpt['perf']
    model.to(device)
    args.n_gpu = torch.cuda.device_count()
    dp = args.n_gpu > 1
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)

    train_men_loader = get_mention_loader(samples_train, args.max_len, tokenizer, args.mention_bsz)
    val_men_loader = get_mention_loader(samples_val, args.max_len, tokenizer, args.mention_bsz)
    test_men_loader = get_mention_loader(samples_test, args.max_len, tokenizer, args.mention_bsz)
    entity_loader = get_entity_loader(entities, args.entity_bsz)

    entity_map = get_entity_map(entities)

    train_labels = get_labels(samples_train, entity_map)
    val_labels = get_labels(samples_val, entity_map)
    test_labels = get_labels(samples_test, entity_map)
    model.train()
    effective_bsz = args.B * args.gradient_accumulation_steps
    # train
    logger.log('***** train *****')
    logger.log('# train samples: {:d}'.format(num_train_samples))
    logger.log('# val samples: {:d}'.format(len(samples_val)))

    logger.log('# test samples: {:d}'.format(len(samples_test)))
    logger.log('# epochs: {:d}'.format(args.epochs))
    logger.log(' batch size : {:d}'.format(args.B))
    logger.log(' gradient accumulation steps {:d}'
               ''.format(args.gradient_accumulation_steps))
    logger.log(
        ' effective training batch size with accumulation: {:d}'
        ''.format(effective_bsz))
    logger.log(' # training steps: {:d}'.format(num_train_steps))
    logger.log(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.log(' learning rate: {:g}'.format(args.lr))
    logger.log(' # parameters: {:d}'.format(count_parameters(model)))

    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    start_epoch = 1

    if args.resume_training:
        step_num = cpt['step_num']
        tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
        start_epoch = cpt['epoch'] + 1
    model.zero_grad()
    all_cands_embeds = None


    logger.log('get candidates embeddings')
    if args.resume_training or args.epochs == 0:
            # we store candidates embeddings after each epoch
        all_cands_embeds = np.load(args.cands_embeds_path)
    elif args.rands_ratio != 1.0 and args.epochs != 0:
        all_cands_embeds = get_embeddings(entity_loader, model, False, device)

    for epoch in range(start_epoch, args.epochs + 1):

        logger.log('\nEpoch {:d}'.format(epoch))

        epoch_start_time = datetime.now()

        if args.rands_ratio == 1.0:
            logger.log('no need to mine hard negatives')
            candidates = None
        else:
            mention_embeds = get_embeddings(train_men_loader, model, True, device)
            logger.log('mining hard negatives')
            mining_start_time = datetime.now()
            candidates = get_hard_negative(mention_embeds, all_cands_embeds,
                                           args.num_cands,
                                           max_num_positives,
                                           args.use_gpu_index)[0]
            mining_time = strtime(mining_start_time)
            logger.log('mining time for epoch {:3d} '
                       'are {:s}'.format(epoch, mining_time))
        train_loader = get_loader_from_candidates(samples_train, entities,
                                                  train_labels, args.max_len,
                                                  tokenizer, candidates,
                                                  args.num_cands,
                                                  args.rands_ratio,
                                                  args.type_loss,
                                                  True, args.B)
        epoch_train_start_time = datetime.now()
        train_loader  = tqdm(train_loader)
        for step, batch in enumerate(train_loader):
            model.train()
            bsz = batch[0].size(0)
            batch = tuple(t.to(device) for t in batch)
            loss = model(*batch)[0]
            if dp:
                loss = loss.sum() / bsz
            else:
                loss /= bsz
            loss_avg = loss / args.gradient_accumulation_steps

            loss_avg.backward()
            tr_loss += loss_avg.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step_num += 1

                if step_num % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    logger.log('Step {:10d}/{:d} | Epoch {:3d} | '
                               'Batch {:5d}/{:5d} | '
                               'Average Loss {:8.4f}'
                               ''.format(step_num, num_train_steps,
                                         epoch, step + 1,
                                         len(train_loader), avg_loss))
                    logging_loss = tr_loss
        logger.log('training time for epoch {:3d} '
                   'is {:s}'.format(epoch, strtime(epoch_train_start_time)))
        all_cands_embeds = get_embeddings(entity_loader, model, False, device)
        all_mention_embeds = get_embeddings(val_men_loader, model, True, device)
        top_k, scores_k = get_hard_negative(all_mention_embeds,
                                            all_cands_embeds, args.dev_cand,
                                            0, args.use_gpu_index)
        eval_result = evaluate(scores_k, top_k, val_labels, entity_map)

        logger.log('Done with epoch {:3d} | train loss {:8.4f} | '
                   'validation hard recall {:8.4f}'
                   '|validation LRAP {:8.4f} | validation recall {:8.4f}|'
                   ' epoch time {} '.format(
            epoch,
            tr_loss / step_num,
            eval_result[0],
            eval_result[1],
            eval_result[2],
            strtime(epoch_start_time)
        ))
        save_model = (eval_result[0] >= best_val_perf)
        if save_model:
            current_best = eval_result[0]
            logger.log('------- new best val perf: {:g} --> {:g} '
                       ''.format(best_val_perf, current_best))
            best_val_perf = current_best
            torch.save({'opt': args,
                        'sd': model.module.state_dict() if dp else model.state_dict(),
                        'perf': best_val_perf, 'epoch': epoch,
                        'opt_sd': optimizer.state_dict(),
                        'scheduler_sd': scheduler.state_dict(),
                        'tr_loss': tr_loss, 'step_num': step_num,
                        'logging_loss': logging_loss},
                       args.model)
            np.save(args.cands_embeds_path, all_cands_embeds)

        else:
            logger.log('')

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


    train(train_data, dev_data, test_data, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default="dataset/ncbi-disease/")
    parser.add_argument('--model',
                        default="model_retriever/ncbi_new_retriever.pt",
                        help='model path')
    parser.add_argument("--pretrained_model",
                        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument('--resume_training',
                        type=bool,
                        # action='store_true',
                        default=False,
                        help='resume training from checkpoint?')
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

    parser.add_argument('--B', type=int, default=2,
                        help='the batch size per gpu')
    parser.add_argument('--lr', type=float, default=2e-6,
                        help='the learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='the number of training epochs')
    parser.add_argument('--k', type=int, default=100,
                        help='recall@k when evaluate')
    parser.add_argument("--dev_cand", default=6,type=int)


    parser.add_argument('--warmup_proportion', type=float, default=0.2,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='num gradient accumulation steps [%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers [%(default)d]')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='num logging steps [%(default)d]')
    parser.add_argument('--gpus', default='0', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--rands_ratio', default=0.9, type=float,
                        help='the ratio of random candidates and hard')
    parser.add_argument('--num_cands', default=32, type=int,
                        help='the total number of candidates')
    parser.add_argument('--mention_bsz', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--entity_bsz', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--use_gpu_index', default=True,
                        help='use gpu index?')
    parser.add_argument("--update_can_embedding",default=True,type=bool)
    parser.add_argument('--cands_embeds_path', type=str,
                        default="dataset/candidates_embeds/candidate_embeds.npy",
                        help='the directory of candidates embeddings')
    parser.add_argument('--use_cached_embeds', action='store_true',
                        help='use cached candidates embeddings ?')

    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior
    main(args)
