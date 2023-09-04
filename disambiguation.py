import torch
from torch import nn
from torch.nn import Linear
from transformers import BertModel, BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertForQuestionAnswering
from loss import MultiLabelLoss


class PromptEncoder(nn.Module):
    def __init__(self, pretrained_model, device, type_loss):
        super(PromptEncoder, self).__init__()
        self.device = device
        self.loss_fnc = torch.nn.functional.binary_cross_entropy_with_logits
        bert_config = BertConfig.from_pretrained(pretrained_model)
        bert_config.vocab_size = bert_config.vocab_size + 20
        self.model = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size + 20)
        self.cls = BertOnlyMLMHead(bert_config)

    def forward(self, input_ids, attention_mask, ans_pos, choice_label, labels, op="train"):
        outputs = self.model(input_ids, attention_mask)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        batch_num, seq_len, vocab_size = prediction_scores.shape
        ans_pos = ans_pos.view(batch_num, -1)
        ans_position_labels = (torch.zeros(batch_num, seq_len).to(self.device)).scatter(
            1, ans_pos, torch.ones(batch_num, seq_len).to(self.device))

        ans_position_labels = ans_position_labels.unsqueeze(2).repeat(1, 1, vocab_size)
        ans_position_mask = ans_position_labels.ge(0.5)
        ans_logits = torch.masked_select(prediction_scores, ans_position_mask).view(batch_num, vocab_size)

        logits_positons = (torch.zeros(batch_num, vocab_size).to(self.device)).scatter(
            1, choice_label, torch.ones(batch_num, vocab_size).to(self.device))
        logits_mask = logits_positons.ge(0.5)
        choice_logits = torch.masked_select(ans_logits, logits_mask).view(batch_num, -1)
        if op == "train":
            return self.loss_fnc(choice_logits, labels)
        else:
            return choice_logits
