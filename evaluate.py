import argparse

from disambiguation import *
from data_disambiguation import *
from utils import *
from loss import *
from datetime import datetime
from torch.optim import AdamW
from tqdm import tqdm
from pretrain import MaskLMEncoder


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


def get_pretrained_model(pretrained_model, tokenizer, device, args):
    model = MaskLMEncoder(pretrained_model, tokenizer, device)
    state_dict = torch.load(args.pretrained_model_path) if device.type == 'cuda' else \
        torch.load(args.model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["sd"])
    return model.model.bert


def load_model(is_init, device, type_loss, args):
    model = PromptEncoder(args.pretrained_model, device, type_loss)
    # model = CaEncoder(args.pretrained_model, device, type_loss)
    if args.use_pretrained_model:
        model.model = get_pretrained_model(args.pretrained_model, args.tokenizer, device, args)
    if not is_init:
        state_dict = torch.load(args.model) if device.type == 'cuda' else \
            torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['sd'])
    return model


def get_hit_scores(indices, labels):
    hit = 0
    nums = len(labels)
    for i in range(nums):
        indice = indices[i]
        label = labels[i]
        hit += any([label[index] for index in indice])
    return hit / nums


def evaluate(model, data_loader, device):
    data_loader = tqdm(data_loader)
    scores = []
    labels = []

    for step, batch in enumerate(data_loader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, ans_pos, choice_label, label = batch
        score = model(input_ids, attention_mask, ans_pos, choice_label, label, "val")
        scores += score.tolist()
        labels += label.tolist()

    scores = torch.tensor(scores)

    top1_indices = torch.topk(scores, k=1).indices.tolist()
    hit1 = get_hit_scores(top1_indices, labels)
    top5_indices = torch.topk(scores, k=5).indices.tolist()
    hit5 = get_hit_scores(top5_indices, labels)

    return hit1, hit5


def eval(samples_test, args):
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
    special_tokens = ["[E1]", "[/E1]", '[or]', "[NIL]"]
    sel_tokens = [f"[{i}]" for i in range(args.cand_num)]
    special_tokens += sel_tokens
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.tokenizer = tokenizer

    test_loader = get_prompt_mention_loader(samples_test, entities, tokenizer, False, True, args)

    model = load_model(False, device, args.type_loss, args)
    model.to(device)

    hit1, hit5 = evaluate(model, test_loader, device)
    logger.log(f"test acc @1: {hit1},   test acc @5: {hit5}")


def main(args):
    test_data = load_data(args.dataset + args.test_data)

    eval(test_data, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default="dataset/bc5cdr/")
    parser.add_argument("--model",
                        default="model_disambiguation/bc5cdr_disambiguation_prompt_pretrain.pt")
    parser.add_argument("--pretrained_model",
                        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--use_pretrained_model" ,
                        action="store_true")
    parser.add_argument("--pretrained_model_path",
                        default="model_pretrain/bc5cdr_pretrain.pt")

    parser.add_argument("--type_loss", type=str,
                        default="sum_log_nce",
                        choices=["log_sum", "sum_log", "sum_log_nce",
                                 "max_min", "bce_loss"])
    parser.add_argument("--max_len", default=512)
    parser.add_argument("--max_ent_len", default=32)
    parser.add_argument("--max_text_len", default=256)

    parser.add_argument("--test_data", default="disambiguation_output/test.json")
    parser.add_argument("--kb_path", default="entity_kb.json")

    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--cand_num", default=6)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--gpus", default="4")
    parser.add_argument("--logging_steps", default=100)
    args = parser.parse_args()
    # Set environment variables before all else.
    # Sets torch.cuda behavior
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
