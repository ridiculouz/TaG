import torch
import torch.nn as nn

from modules.table_filler import BaseTableFiller, get_hrt, gen_hts, get_node_embed, convert_node_to_table
from modules.mention_extraction import SequenceTagger
from modules.coreference_resolution import CoreferenceResolutionTableFiller
from modules.relation_extraction import RelationExtractionTableFiller
from modules.graph import RelGCN
from long_seq import process_long_input, process_multiple_segments

num_ner = 6
        
class MEModel(nn.Module):
    """
    Solely compute mention spans, and store in advance.
    """
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config # include custom config (args), basically extended from BertConfig
        self.bert = encoder
        
        self.hidden_size = config.hidden_size
        self.tagger = SequenceTagger(self.hidden_size)

    def encode(self, input_ids, attention_mask):
        """
        Reference: ATLOP code
        """ 
        # start tokens: config.bos_token_id
        # end_tokens: config.eos_token_id
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output

    def compute_loss(self, input_ids=None, attention_mask=None, label=None):
        sequence_output = self.encode(input_ids, attention_mask)
        loss1 = self.tagger.compute_loss(sequence_output, attention_mask, label)
        return loss1

    def inference(self, input_ids=None, attention_mask=None):
        sequence_output = self.encode(input_ids, attention_mask)
        preds = self.tagger.inference(sequence_output, attention_mask)
        return preds

class JointTableModel(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()        
        self.config = config
        self.bert = encoder

        self.hidden_size = config.hidden_size
        self.CR = CoreferenceResolutionTableFiller(self.hidden_size)
        self.RE = RelationExtractionTableFiller(self.hidden_size, config.num_class, beta=config.beta)
        self.alpha = config.alpha
        self.beta = config.beta

    def encode(self, input_ids, attention_mask):
        """
        Reference: ATLOP code
        """ 
        # start tokens: config.bos_token_id
        # end_tokens: config.eos_token_id
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output, attention

    def forward(self, input_ids=None, attention_mask=None):
        # return self.compute_loss(input_ids, attention_mask, mr_labels)
        return

    def compute_loss(self, input_ids=None, attention_mask=None, spans=None, 
                     hts=None, cr_label=None, re_label=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, ts = get_hrt(sequence_output, attention, spans, hts)
        cr_loss = self.CR.compute_loss(hs, ts, cr_label)
        re_loss = self.RE.compute_loss(hs, ts, re_label)
        return self.alpha * cr_loss + re_loss

    def inference(self, input_ids=None, attention_mask=None, spans=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        span_len = [len(span) for span in spans]
        hts = [gen_hts(l) for l in span_len]
        hs, ts = get_hrt(sequence_output, attention, spans, hts)
        cr_predictions = self.CR.inference(hs, ts, span_len, hts)
        re_predictions = self.RE.inference(hs, ts, span_len, hts, cr_predictions)
        outputs = {'cr_predictions': cr_predictions, 're_predictions': re_predictions}
        return outputs

class Table2Graph(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()        
        self.config = config
        self.bert = encoder

        self.hidden_size = config.hidden_size

        self.CRTablePredictor = BaseTableFiller(hidden_dim=self.hidden_size, emb_dim=self.hidden_size,
                                                block_dim=64, num_class=1, sample_rate=0, lossf=nn.BCEWithLogitsLoss())
        self.RETablePredictor = BaseTableFiller(hidden_dim=self.hidden_size, emb_dim=self.hidden_size,
                                                block_dim=64, num_class=1, sample_rate=0, lossf=nn.BCEWithLogitsLoss())
        self.RGCN = RelGCN(self.hidden_size, self.hidden_size, self.hidden_size, num_rel=3, num_layer=config.num_gcn_layers)

        self.CR = CoreferenceResolutionTableFiller(self.hidden_size)
        self.RE = RelationExtractionTableFiller(self.hidden_size, config.num_class, beta=config.beta)
        self.alpha = config.alpha
        self.beta = config.beta
        self.rho = config.rho

    def encode(self, input_ids, attention_mask):
        """
        Reference: ATLOP code
        """ 
        # start tokens: config.bos_token_id
        # end_tokens: config.eos_token_id
        func = process_multiple_segments if self.config.dataset == 'dwie' else process_long_input
        sequence_output, attention = func(self.config, self.bert, input_ids, attention_mask)
        return sequence_output, attention

    def forward(self, input_ids=None, attention_mask=None, spans=None, 
                    syntax_graph=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        span_len = [len(span) for span in spans]
        hts = [gen_hts(l) for l in span_len]
        hs, ts = get_hrt(sequence_output, attention, spans, hts)
        rs = hs[:, self.hidden_size:]   # get context embedding

        cr_table = self.CRTablePredictor.forward(hs, ts)
        re_table = self.RETablePredictor.forward(hs, ts)

        # convert logits to tabel in batch form
        offset = 0
        cr_adj, re_adj = [], [] # store sub table first
        for l in span_len:
            cr_sub = cr_table[offset: offset + l*l].view(l, l)
            re_sub = re_table[offset: offset + l*l].view(l, l)
            cr_sub = torch.softmax(cr_sub, dim=-1)
            re_sub = torch.softmax(re_sub, dim=-1)
            cr_adj.append(cr_sub)
            re_adj.append(re_sub)
            offset += l*l
        cr_adj = torch.block_diag(*cr_adj)
        re_adj = torch.block_diag(*re_adj)
        sg_adj = syntax_graph

        adjacency_list = [cr_adj, re_adj, sg_adj]
        nodes = get_node_embed(sequence_output, spans)

        nodes = self.RGCN(nodes, adjacency_list)

        hs, ts = convert_node_to_table(nodes, span_len)
        hs = torch.cat([hs, rs], dim=-1)
        ts = torch.cat([ts, rs], dim=-1)

        cr_logits = self.CR.forward(hs, ts)
        re_logits = self.RE.forward(hs, ts)

        return cr_logits, re_logits

    def compute_loss(self, input_ids=None, attention_mask=None, spans=None, 
                     hts=None, cr_label=None, re_label=None,
                     cr_table_label=None, re_table_label=None,
                     syntax_graph=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, ts = get_hrt(sequence_output, attention, spans, hts)
        rs = hs[:, self.hidden_size:]   # get context embedding

        # graph structure prediction & compute auxiliary loss
        cr_table_loss, cr_table = self.CRTablePredictor.compute_loss(hs, ts, cr_table_label, return_logit=True)
        re_table_loss, re_table = self.RETablePredictor.compute_loss(hs, ts, re_table_label, return_logit=True)

        # convert logits to table in batch form
        span_len = [len(span) for span in spans]
        offset = 0
        cr_adj, re_adj = [], [] # store sub table first
        for l in span_len:
            cr_sub = cr_table[offset: offset + l*l].view(l, l)
            re_sub = re_table[offset: offset + l*l].view(l, l)
            cr_sub = torch.softmax(cr_sub, dim=-1)
            re_sub = torch.softmax(re_sub, dim=-1)
            cr_adj.append(cr_sub)
            re_adj.append(re_sub)
            offset += l*l
        cr_adj = torch.block_diag(*cr_adj)
        re_adj = torch.block_diag(*re_adj)
        sg_adj = syntax_graph

        adjacency_list = [cr_adj, re_adj, sg_adj]
        nodes = get_node_embed(sequence_output, spans)

        nodes = self.RGCN(nodes, adjacency_list)

        hs, ts = convert_node_to_table(nodes, span_len)
        hs = torch.cat([hs, rs], dim=-1)
        ts = torch.cat([ts, rs], dim=-1)

        cr_loss = self.CR.compute_loss(hs, ts, cr_label)
        re_loss = self.RE.compute_loss(hs, ts, re_label)
        return cr_loss + re_loss + self.alpha * cr_table_loss + self.alpha * re_table_loss

    def inference(self, input_ids=None, attention_mask=None, spans=None, 
                    syntax_graph=None):
        span_len = [len(span) for span in spans]
        hts = [gen_hts(l) for l in span_len]

        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "spans": spans,
                  "syntax_graph": syntax_graph}

        cr_logits, re_logits = self.forward(**inputs) 
        cr_logits = cr_logits.to(dtype=torch.float64)
        cr_logits = torch.sigmoid(cr_logits)
        re_logits = re_logits.to(dtype=torch.float64)
        ####################################################
        # Levenstein decoding
        ####################################################
        re_labels = self.RE.get_label(re_logits)[:, 1:].bool()
        lev_reg = []
        offset = 0
        for batch_idx, l in enumerate(span_len):
            square = re_labels[offset:offset+l*l].view(l, l, -1)
            levmat = torch.zeros(l, l).to(re_labels.device)
            for i in range(l):
                for j in range(l):
                    lev_ij = (square[i, :]^square[j, :]).float().sum() + (square[:, i]^square[:, j]).float().sum()
                    levmat[i][j] = lev_ij
            levmat = torch.sigmoid(levmat)
            lev_reg.append(levmat.view(-1, 1))
            offset += l*l
        lev_reg = torch.cat(lev_reg, dim=0).to(dtype=torch.float64)
        cr_logits = cr_logits - self.rho * lev_reg
        ####################################################
        # decoding cr
        ####################################################
        cr_predictions = self.CR.inference(span_len=span_len, batch_hts=hts, logits=cr_logits)
        ####################################################
        # decoding re
        ####################################################
        re_predictions = self.RE.inference(span_len=span_len, batch_hts=hts, batch_clusters=cr_predictions, logits=re_logits)
        
        outputs = {'cr_predictions': cr_predictions, 're_predictions': re_predictions}
        return outputs