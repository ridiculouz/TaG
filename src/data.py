# implementation of data reader of different datasets
from itertools import accumulate
import json
from tqdm import tqdm
import torch
import numpy as np
import os

from modules.mention_extraction import tag

# find project abs dir
root_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(root_dir)

docred_rel2id = json.load(open(os.path.join(root_dir, "data/docred/rel2id.json"), "r"))
docred_id2rel = {v: k for k, v in docred_rel2id.items()}

def read_dataset(tokenizer, split='train_annotated', dataset='docred', task='me'):
    if task == 'me':
        return read_me_data(tokenizer=tokenizer, split=split, dataset=dataset)
    if task == 'gc':
        return read_gc_data(tokenizer=tokenizer, split=split, dataset=dataset)
    raise ValueError("Unknown task type.")

def read_me_data(tokenizer, split='train_annotated', dataset='docred'):
    i_line = 0
    features = []
    bias_mentions, bias_entities = 0, 0    # count the mention that been discard
    cnt_mention = 0
    data_path = 'data/{}/{}.json'.format(dataset, split)
    data = json.load(open(os.path.join(root_dir, data_path), 'r', encoding='utf-8'))
    if dataset == 'docred' or dataset == 're-docred':
        rel2id = docred_rel2id
    else:
        raise ValueError("Unknown dataset.")

    offset = 1  # for BERT/RoBERTa/Longformer, all follows [CLS] sent [SEP] format
    for sample in tqdm(data, desc='Data'):
        i_line += 1
        sents = []
        doc_map = []
        entities = sample['vertexSet']
        spans = []
        for i_s, sent in enumerate(sample['sents']):
            sent_map = {}    # map the old index to the new index
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            sent_map[i_t + 1] = len(sents)
            doc_map.append(sent_map)
        label = [tag["O"] for i in range(len(sents))]
        entity_len = []
        for e in entities:
            flag_entity = False
            e_new = set()   # first: remove duplicate mentions for single entity
            for m in e:
                e_new.add((m["sent_id"], m["pos"][0], m["pos"][1]))
            entity_len.append(len(e_new))  # save entity lens to later construct cr_label
            cnt_mention += len(e_new)
            for m in e_new:
                flag_mention = False
                start, end = doc_map[m[0]][m[1]], doc_map[m[0]][m[2]]
                # add this span to mr_spans
                spans.append((start + offset, end + offset))
                for j in range(start, end):
                    if label[j] != 0:
                        bias_mentions += 1
                        flag_mention = True
                        break
                if flag_mention:
                    flag_entity = True
                    continue
                label[start] = tag["B"]
                for j in range(start + 1, end):
                    label[j] = tag["I"]
            if flag_entity:
                bias_entities += 1

        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        label = [tag["O"]] + label + [tag["O"]]
        # collect re_triples to find useful mentions
        accumulate_entity_len = [0] + [i for i in accumulate(entity_len)]
        re_triples = []
        if split != "test":
            for hrt in sample["labels"]:
                h = hrt["h"]
                t = hrt["t"]
                r = rel2id[hrt["r"]]
                re_triples.append({"h": h, "r": r, "t": t})
        # collect useful mention: mentions of entities in relation triples
        useful_entity = set()
        useful_mention = []
        for hrt in re_triples:
            useful_entity.add(hrt["h"])
            useful_entity.add(hrt["t"])
        for e in useful_entity:
            for i in range(accumulate_entity_len[e], accumulate_entity_len[e + 1]):
                useful_mention.append(spans[i])
        feature = {"input_ids": input_ids, "label": label, "spans": spans, 
                    "doc_map": doc_map, "doc_len": len(input_ids), "useful_mention": useful_mention}
        features.append(feature)

    print("# of documents:\t\t{}".format(i_line))
    print("# of mention bias:\t{}".format(bias_mentions))
    print("# of entity bias:\t{}".format(bias_entities))
    print("# of mentions:\t\t{}".format(cnt_mention))
    return features

def read_gc_data(tokenizer, split='train_annotated', dataset='docred'):
    i_line = 0
    features = []
    data_path = f'data/{dataset}/{split}_gc.json'
    data = json.load(open(os.path.join(root_dir, data_path), 'r', encoding='utf-8'))

    offset = 1  # for BERT/RoBERTa/Longformer, all follows [CLS] sent [SEP] format
    for sample in tqdm(data, desc='Data'):
        i_line += 1
        sents = []
        doc_map = []
        span_start, span_end = [], []
        raw_spans = [((s[0][0], s[0][1]), (s[1][0], s[1][1])) for s in sample['spans']]
        for s in raw_spans:
            span_start.append(s[0])
            span_end.append(s[1])
        # get doc_map, input_ids
        for i_s, sent in enumerate(sample['sents']):
            sent_map = {}    # map the old index to the new index
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in span_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t + 1) in span_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            sent_map[i_t + 1] = len(sents)
            doc_map.append(sent_map)
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        # get spans
        spans = []
        for rs in raw_spans:
            start = doc_map[rs[0][0]][rs[0][1]]
            end = doc_map[rs[1][0]][rs[1][1]]
            spans.append((start + offset, end + offset))
        # get syntax graph
        syntax_graph = torch.zeros(len(spans), len(spans))
        for i in range(len(spans)):
            for j in range(len(spans)):
                if raw_spans[i][0][0] == raw_spans[j][0][0]:
                    syntax_graph[i][j] = 1
        syntax_graph[syntax_graph == 0] = -1e30
        syntax_graph = torch.softmax(syntax_graph, dim=-1)
        syntax_graph = syntax_graph.cpu().tolist()
        # get clusters and relations
        # if mention not in predictions, add -1 as index
        clusters, relations = [], []
        cr_label, re_label, re_table_label = None, None, None
        hts = None
        if 'labels' in sample:
            for hrt in sample['labels']:
                h = hrt['h']
                t = hrt['t']
                r = docred_rel2id[hrt['r']]
                relations.append({'h': h, 'r': r, 't': t})
        entities = sample['vertexSet']
        for e in entities:
            cluster = []
            for m in e:
                ms = ((m['sent_id'], m['pos'][0]), (m['sent_id'], m['pos'][1]))
                if ms not in raw_spans:
                    cluster.append(-1)
                else:
                    cluster.append(raw_spans.index(ms))
            # cluster = list(set(cluster))
            clusters.append(cluster)
        # construct label, by utilizing mapping
        if 'train' in split:
            # construct ht_to_r dict
            ht_to_r = dict()
            for hrt in relations:
                h, r, t = hrt['h'], hrt['r'], hrt['t']
                if (h, t) in ht_to_r:
                    ht_to_r[(h, t)].append(r)
                else:
                    ht_to_r[(h, t)] = [r]
            # construct label
            entity_len = sample['entity_len']
            accumulate_entity_len = [0] + [i for i in accumulate(entity_len)]

            hts, cr_label, re_label, re_table_label = [], [], [], []
            for h_e in range(len(entities)):
                for t_e in range(len(entities)):
                    cr_flag = -1
                    re_table_flag = 0
                    if (h_e, t_e) in ht_to_r or (t_e, h_e) in ht_to_r:
                        re_table_flag = 1
                    if h_e == t_e:
                        cr_flag = 1
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                            for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                hts.append([h_m, t_m])
                                cr_label.append(cr_flag)
                                re_label.append(relation)
                                re_table_label.append(re_table_flag)
                        continue
                    # add negative cr_label
                    cr_flag = 0
                    if (h_e, t_e) in ht_to_r:
                        relation = [0] * len(docred_rel2id)
                        for r in ht_to_r[(h_e, t_e)]:
                            relation[r] = 1
                        for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                            for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                hts.append([h_m, t_m])
                                cr_label.append(cr_flag)
                                re_label.append(relation)
                                re_table_label.append(re_table_flag)
                    else:
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        for h_m in range(accumulate_entity_len[h_e], accumulate_entity_len[h_e + 1]):
                            for t_m in range(accumulate_entity_len[t_e], accumulate_entity_len[t_e + 1]):
                                hts.append([h_m, t_m])
                                cr_label.append(cr_flag)
                                re_label.append(relation)
                                re_table_label.append(re_table_flag)

            tmp = list(zip(hts, re_label, cr_label))
            tmp = sorted(tmp, key=lambda x: 100*x[0][0]+x[0][1])
            hts, re_label, cr_label = zip(*tmp)
            hts, re_label, cr_label = list(hts), list(re_label), list(cr_label)
            # make sure all data are in table format, i.e. can be view() to transfer to table
            assert len(hts) == len(re_label) == len(cr_label) == len(spans)*len(spans), "{} must match {}.".format(len(hts), len(spans)*len(spans))      
        
        feature = {"input_ids": input_ids, "spans": spans, "hts": hts,
                   "cr_label": cr_label, "cr_clusters": clusters,
                   "re_label": re_label, "re_triples": relations, "re_table_label": re_table_label,
                   "syntax_graph": syntax_graph,    # contain self loop
                   "vertexSet": entities,
                   "title": sample["title"]}
        features.append(feature)

    print("# of documents:\t\t{}.".format(i_line))
    return features
