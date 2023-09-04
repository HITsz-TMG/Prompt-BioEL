from transformers import BertForMaskedLM
import torch
from torch import nn


class MaskLMEncoder(nn.Module):
    def __init__(self, pretrained_model, tokenizer, device):
        super(MaskLMEncoder, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(pretrained_model)
        self.tokenizer = tokenizer
        self.model.bert.resize_token_embeddings(self.tokenizer.vocab_size + 20)
        self.device = device

    def forward(self, input_ids, attention_mask, mask_index, label_ids, weights):
        batch, seq_len = input_ids.shape

        lm_outputs = self.model(input_ids, attention_mask)
        logits = lm_outputs.logits


        hidden_size = logits.shape[-1]
        position_labels = (torch.zeros(batch, seq_len).to(self.device)).scatter(
            1, mask_index, torch.ones(batch,seq_len).to(self.device))
        position_labels = position_labels.unsqueeze(2).repeat(1, 1, hidden_size)
        position_mask = position_labels.ge(0.5)
        position_logits = torch.masked_select(logits, position_mask).view(batch, -1, hidden_size)
        weights = weights.unsqueeze(2).repeat(1,1,hidden_size)
        weights_mask = weights.ge(0.5)
        position_logits = torch.masked_select(position_logits, weights_mask).view(-1, hidden_size)

        loss = torch.nn.functional.cross_entropy(input=position_logits,
                                                 target=label_ids,
                                               )
        return loss
