import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

def get_node_embed(sequence_output: torch.Tensor=None, batch_span_pos=None, strategy='marker'):
    span_embs = []
    for i, span_pos in enumerate(batch_span_pos):
        for span in span_pos:
            if strategy == 'marker':
                emb = sequence_output[i, span[0]]
                span_embs.append(emb)
            else:
                raise ValueError("Unimplemented strategy.")
    span_embs = torch.stack(span_embs, dim=0)
    return span_embs

def convert_node_to_table(nodes: torch.Tensor=None, span_len=None):
    offset = 0
    hss, tss = [], []
    for l in span_len:
        x = nodes[offset: offset + l]
        hs = x.repeat(1, l).view(l*l, -1)
        ts = x.repeat(l, 1)
        hss.append(hs)
        tss.append(ts)
        offset += l
    hss = torch.cat(hss, dim=0)
    tss = torch.cat(tss, dim=0)
    return hss, tss

def form_table_input(sequence_output: torch.Tensor=None, span_pos=None, strategy='max-pooling'):
    """
    input:          span_pos is a [batch[span pos[]]]
                    hts is a [batch[ht pair]]
    return a tuple: node_embeds: tensor, node_len: list
         satisfied: \sum{node_len} = len(node_embed)

    """
    span_len = [len(row) for row in span_pos]
    span_embed = []
    for i, row in enumerate(span_pos):
        for span in row:
            emb = sequence_output[i, span[0]:span[1], :]
            if strategy == 'max-pooling':
                emb = torch.max(emb, dim=0)[0]
                span_embed.append(emb)
            if strategy == 'marker':
                emb = emb[0]
                span_embed.append(emb)
            else:
                raise ValueError("Unimplemented strategy.")
    span_embed = torch.stack(span_embed, dim=0)
    return span_embed, span_len

def get_hrt(sequence_output: torch.Tensor=None, attention: torch.Tensor=None,\
            batch_span_pos=None, batch_hts=None, strategy='marker'):
    """
    span_pos and hts are in batch format.
    attention shape: (batch_size, num_heads, sequence_length, sequence_length).
    """
    hss, tss, rss = [], [], []
    for i, (span_pos, hts) in enumerate(zip(batch_span_pos, batch_hts)):
        span_embs, span_atts = [], []
        for span in span_pos:
            if strategy == 'marker':
                emb = sequence_output[i, span[0]]
                att = attention[i, :, span[0]]
            else:
                raise ValueError("Unimplemented strategy.")
            span_embs.append(emb)
            span_atts.append(att)
        span_embs = torch.stack(span_embs, dim=0)   # [n_s, d]
        span_atts = torch.stack(span_atts, dim=0)   # [n_s, num_heads, seq_len]

        hts = torch.LongTensor(hts).to(sequence_output.device)
        hs = torch.index_select(span_embs, 0, hts[:, 0])
        ts = torch.index_select(span_embs, 0, hts[:, 1])

        h_att = torch.index_select(span_atts, 0, hts[:, 0])
        t_att = torch.index_select(span_atts, 0, hts[:, 1])
        ht_att = (h_att * t_att).mean(1)
        ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
        rs = contract("ld,rl->rd", sequence_output[i], ht_att)
        hss.append(hs)
        tss.append(ts)
        rss.append(rs)
    hss = torch.cat(hss, dim=0)
    tss = torch.cat(tss, dim=0)
    rss = torch.cat(rss, dim=0)
    hss = torch.cat([hss, rss], dim=-1)
    tss = torch.cat([tss, rss], dim=-1)
    return hss, tss

def gen_hts(span_len: int):
    return [[i, j] for i in range(span_len) for j in range(span_len)]

class BaseTableFiller(nn.Module):
    def __init__(self, hidden_dim: int, emb_dim:int, block_dim: int, num_class: int, sample_rate:float, lossf: nn.Module):
        super().__init__()
        # validation: input can be chunked into blocks
        assert emb_dim % block_dim == 0, "emb_dim must be multiple of block_dim."

        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.block_dim = block_dim
        self.num_block = hidden_dim // block_dim
        self.num_class = num_class
        self.head_extractor = nn.Linear(2 * hidden_dim, emb_dim)                    # add a linear transform + nonlinear activation layer
        self.tail_extractor = nn.Linear(2 * hidden_dim, emb_dim)                    # before bilinear layer
        self.linear = nn.Linear(2 * emb_dim, num_class, bias=False)
        self.bilinear = nn.Linear(emb_dim * block_dim, num_class)    # size: [(d^2/k), num_class], bias is included
        if sample_rate == 0:
            self.sampler = None
        else:
            self.sampler = nn.Dropout(p=sample_rate)
        self.lossf = lossf

    def forward(self, head_embed: torch.Tensor=None, tail_embed: torch.Tensor=None):
        """
        input: nodes' representations, size: [n, d].
        embeds should be concat from batch before !!
        """
        hs = torch.tanh(self.head_extractor(head_embed))
        ts = torch.tanh(self.tail_extractor(tail_embed))

        linear_logits = self.linear(torch.cat([hs, ts], dim=-1))

        hs = hs.view(-1, self.num_block, self.block_dim)
        ts = hs.view(-1, self.num_block, self.block_dim)
        bl = (hs.unsqueeze(3) * ts.unsqueeze(2)).view(-1, self.emb_dim * self.block_dim)
        bilinear_logits = self.bilinear(bl)
        logits = bilinear_logits + linear_logits
        return logits

    def compute_loss(self, head_embed: torch.Tensor, tail_embed: torch.Tensor, labels: torch.Tensor, return_logit: bool=False):
        """
        all tensor with size [\sum{n^2}]
        """
        logits = self.forward(head_embed, tail_embed)

        sample_logits = logits
        sample_labels = labels
        # sample negative data
        if self.sampler is not None:
            if len(labels.size()) == 2:
                neg_mask = (labels[:, 1:].sum(dim=-1) == 0).float()
            else:
                neg_mask = (labels == 0).float()
            pos_mask = 1 - neg_mask
            mask = pos_mask + self.sampler(neg_mask)
            sample_logits = logits[mask == 1]
            sample_labels = labels[mask == 1]

        if isinstance(self.lossf, nn.BCEWithLogitsLoss):
            sample_labels = sample_labels.float()
            sample_logits = sample_logits.squeeze(-1)
        # if BCELoss:   labels = [n^2]
        # else:         labels = [n^2, num_class]
        loss = self.lossf(sample_logits, sample_labels)
        
        if return_logit:
            return loss, logits
        return loss
