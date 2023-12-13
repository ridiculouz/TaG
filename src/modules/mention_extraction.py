import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

tag = {"O": 0, "B": 1, "I": 2}

class SequenceTagger(nn.Module):
    def __init__(self, hidden_dim: int, num_class: int=3, dropout_rate: float=0.3):
        """
        follow the BIO setting.
        """
        super(SequenceTagger, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_class)
        self.activate_func = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.lossf = nn.CrossEntropyLoss()
        
    # input size: [batch_size, doc_len, hidden_dim]
    def forward(self, input: torch.Tensor):
        logits = self.classifier(self.dropout(self.activate_func(input)))
        return logits

    def compute_loss(self, input: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        # labels here: sequence tensor
        logits = self.forward(input) # [batch_size, doc_len, num_class]
        # delete padding & resize
        logits = logits[attention_mask == 1]
        labels = labels[attention_mask == 1]
        # compute cross entropy loss
        loss = self.lossf(logits, labels)
        return loss

    def inference(self, input, attention_mask, strategy='discard'):
        """
        return a list of list of tuple, containing predicted span results.
        
        @return: [[(start_i, end_i), ...], ...]
        """
        # current support:
        #   discard: discard invalid "I" tokens
 
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        pred_logit = self.forward(input)
        pred_class = torch.max(pred_logit, dim=-1)[1]
        pred_class = pred_class.cpu().numpy().astype(np.int32)
        pred_spans = [[] for i in range(len(seq_len))]
        for i, l_i in enumerate(seq_len):
            flag = False    # whether in a mention or not
            start = -1      # start position of mention
            for j in range(1, l_i):
                if pred_class[i][j] == 0:   # O case
                    if flag:
                        pred_spans[i].append((start, j))
                        flag = False
                elif pred_class[i][j] == 1: # B case
                    if flag:
                        pred_spans[i].append((start, j))
                    else:
                        flag = True
                    start = j
                elif pred_class[i][j] == 2: # I case
                    pass
                else:
                    raise ValueError("Unexpected predictions in ME.")
        return pred_spans