from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from .table_filler import BaseTableFiller
from .loss_func import ATLoss

# implementation of different aggregation strategy
MAJORITY_VOTE_TH = 0.5

def majority_vote(probs: torch.Tensor) -> torch.Tensor:
    # probs has been converted to 0/1 already.
    num_voters = probs.size(0)
    probs = probs.sum(0) / num_voters
    preds = probs > MAJORITY_VOTE_TH
    preds = torch.nonzero(preds).squeeze(-1).cpu().tolist()
    return preds

def one_vote(probs: torch.Tensor) -> torch.Tensor:
    preds = probs.sum(0)
    preds = torch.nonzero(preds).squeeze(-1).cpu().tolist()
    return preds

def mean_vote(probs: torch.Tensor) -> torch.Tensor:
    num_voters = probs.size(0)
    probs = probs.sum(0) / num_voters
    return probs.unsqueeze(0)

def softmax_vote():
    return

class RelationExtractionTableFiller(BaseTableFiller):
    def __init__(self, hidden_dim: int, num_class: int, block_dim: int=64, sample_rate: float=0.0, beta: float=1.0,\
                    max_pred: int=-1, aggr_func: Callable=mean_vote):
        """
        Optional strategy:
        """
        super().__init__(hidden_dim, hidden_dim, block_dim, num_class, sample_rate, ATLoss(beta))
        self.max_pred = max_pred
        self.aggr_func = aggr_func

    def inference(self, head_embed: torch.Tensor=None, tail_embed: torch.Tensor=None,\
                    span_len=None, batch_hts=None, batch_clusters=None, logits: torch.Tensor=None):
        """
        batch_cluster is a n-d list, containing CR clusters for each table.
        """
        offset = 0
        predictions = []
        if logits is None:
            logits = self.forward(head_embed, tail_embed).to(dtype=torch.float64)
        for l, hts, clusters in zip(span_len, batch_hts, batch_clusters):
            preds = logits[offset:offset+len(hts)]
            # preds = torch.sigmoid(preds)       # convert logits to probabilities
            offset += len(hts)

            indices = [[ht_i[0] for ht_i in hts], [ht_i[1] for ht_i in hts]]
            indices = [torch.tensor(x).to(preds.device) for x in indices]
            table = torch.zeros(l, l, self.num_class).to(preds)
            table.index_put_(indices, preds)
            predictions.append(self.__decode_single_table(table, clusters))
        return predictions

    def get_label(self, logits):
        # input: [n, num_class], return 0/1 value w/ same size.
        th_logits = logits[:, 0].unsqueeze(-1)
        mask = (logits > th_logits)
        if self.max_pred > 0:
            top_v, _ = torch.topk(logits, self.max_pred, dim=-1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(-1)) & mask
        output = torch.zeros_like(logits).to(logits)
        output[mask] = 1.0
        output[:, 0] = (output.sum(-1) == 0.).to(output)
        return output

    def __decode_single_table(self, table: torch.Tensor, clusters: list):
        num_ent = len(clusters)
        device = table.device
        clusters = [torch.LongTensor(c).to(device) for c in clusters]
        triples = []
        for i in range(num_ent):
            for j in range(num_ent):
                if i == j:
                    continue
                probs = torch.index_select(table, dim=0, index=clusters[i])
                probs = torch.index_select(probs, dim=1, index=clusters[j])
                probs = probs.view(-1, probs.size(-1))
                if self.aggr_func in [majority_vote, one_vote]:
                    probs = self.get_label(probs)
                    preds = self.aggr_func(probs)
                elif self.aggr_func in [mean_vote]:
                    probs = self.aggr_func(probs)
                    preds = self.get_label(probs).squeeze(0)
                    preds = torch.nonzero(preds).squeeze(-1).cpu().tolist()
                else:
                    raise ValueError("Unimplemented aggregation function.")
                for r in preds:
                    if r == 0:
                        continue
                    triples.append({"h": i, "r": r, "t": j})
        return triples