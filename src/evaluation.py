# implementation of evaluation metrics
import os
import os.path
import json
from metrics import muc, b_cubed, ceafe
from data import docred_id2rel

def get_span_mapping(pred: list, gold: list):
    """
    Input in sample format. Get the mapping dict from pred to gold.
    Span format: (start, end)
    """
    span_mapping = {}
    tp = 0
    for i_p, ps in enumerate(pred):
        for i_g, gs in enumerate(gold):
            if ps == gs:
                span_mapping[i_p] = i_g
                tp += 1
                break
        if i_p not in span_mapping:
            span_mapping[i_p] = -1
    return tp, span_mapping

def get_cluster_mapping(pred: list, gold: list):
    """
    Input in sample format. Get the mapping dict from pred to gold.
    Pred has already been converted by span_mapping.
    Cluster format: [...set of span index...]
    """
    cluster_mapping = {}
    tp = 0
    pred = [set(cluster) for cluster in pred]
    gold = [set(cluster) for cluster in gold]
    for i_p, pc in enumerate(pred):
        for i_g, gc in enumerate(gold):
            if pc == gc:
                cluster_mapping[i_p] = i_g
                tp += 1
                break
        if i_p not in cluster_mapping:
            cluster_mapping[i_p] = -1
    return tp, cluster_mapping

def get_relation_mapping(pred: list, gold: list):
    """
    Input in sample format. Get the mapping dict from pred to gold.
    Pred has already been converted by cluster_mapping.
    Relation format: {'h': h, 'r': r, 't': t}
    """
    relation_mapping = {}
    tp = 0
    for i_p, pr in enumerate(pred):
        for i_g, gr in enumerate(gold):
            if pr == gr:
                relation_mapping[i_p] = i_g
                tp += 1
                break
        if i_p not in relation_mapping:
            relation_mapping[i_p] = -1
    return tp, relation_mapping

def compute_me_f1(pred: list, gold: list):
    """
    Input in batch format.
    """
    gold_cnt, pred_cnt, tp_cnt = 0, 0, 0
    for g, p in zip(gold, pred):
        g, p = list(set(g)), list(set(p))
        gold_cnt += len(g)
        pred_cnt += len(p)
        tp_cnt += get_span_mapping(p, g)[0]
    precision = tp_cnt / (pred_cnt + 1e-7)
    recall = tp_cnt / gold_cnt
    f1 = (precision * recall * 2) / (precision + recall + 1e-7) 
    return f1, precision, recall

def compute_cr_f1(span_mappings=None, pred_clusters: list=None, gold_clusters: list=None, metric=None):
    """
    Input in batch format. Including span_mappings.
    """
    gold_cnt, pred_cnt, tp_cnt = 0, 0, 0
    if span_mappings is None: # pred_spans == gold_spans
        if metric is None:
            for p_clusters, g_clusters in zip(pred_clusters, gold_clusters):  
                gold_cnt += len(g_clusters)
                pred_cnt += len(p_clusters)
                tp_cnt += get_cluster_mapping(p_clusters, g_clusters)[0]     
            precision = tp_cnt / (pred_cnt + 1e-7)
            recall = tp_cnt / gold_cnt    
        elif metric in [muc, b_cubed]:
            p_num, p_den, r_num, r_den = 0, 0, 0, 0
            for p_clusters, g_clusters in zip(pred_clusters, gold_clusters):  
                pn, pd = metric(p_clusters, g_clusters)
                rn, rd = metric(g_clusters, p_clusters)
                p_num += pn
                p_den += pd
                r_num += rn
                r_den += rd
            precision = p_num / (p_den + 1e-7)
            recall = r_num / (r_den + 1e-7)
        elif metric == ceafe:
            p_num, p_den, r_num, r_den = 0, 0, 0, 0
            for p_clusters, g_clusters in zip(pred_clusters, gold_clusters):  
                pn, pd, rn, rd = metric(p_clusters, g_clusters)
                p_num += pn
                p_den += pd
                r_num += rn
                r_den += rd
            pn, pd, rn, rd = metric(pred_clusters, gold_clusters)
            precision = pn / (pd + 1e-7)
            recall = rn / (rd + 1e-7)
        else:
            raise ValueError("Unknown CR metric.")
    else:
        for span_mapping, p_clusters, g_clusters in zip(span_mappings, pred_clusters, gold_clusters):
            # first find a mapping between pred cluster and gold cluster
            new_p_clusters = []
            for pc in p_clusters:
                new_p_clusters.append([span_mapping[i_s] for i_s in pc])
            # calculation
            gold_cnt += len(g_clusters)
            pred_cnt += len(p_clusters)
            tp_cnt += get_cluster_mapping(new_p_clusters, g_clusters)[0]
        precision = tp_cnt / (pred_cnt + 1e-7)
        recall = tp_cnt / gold_cnt
    f1 = (precision * recall * 2) / (precision + recall + 1e-7) 
    return f1, precision, recall

def compute_avg_cr_f1(span_mappings=None, pred_clusters: list=None, gold_clusters: list=None):
    if span_mappings is not None:
        return compute_cr_f1(span_mappings, pred_clusters, gold_clusters)
    muc_f1, _, _ = compute_cr_f1(None, pred_clusters, gold_clusters, muc)
    b3_f1, _, _ = compute_cr_f1(None, pred_clusters, gold_clusters, b_cubed)
    ceafe_f1, _, _ = compute_cr_f1(None, pred_clusters, gold_clusters, ceafe)
    avg_f1 = sum([muc_f1, b3_f1, ceafe_f1]) / 3
    return avg_f1, muc_f1, b3_f1, ceafe_f1

def compute_re_f1(cluster_mappings=None, pred_relations: list=None, gold_relations: list=None, vertexSets: list=None,
                  dataset="docred"):
    """
    Input in batch format. Including cluster_mappings.
    """
    # find project abs dir
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    train_fact = gen_train_facts(os.path.join(root_dir, f"data/{dataset}/train_annotated.json"), 
                                 os.path.join(root_dir, f"data/{dataset}/ref/"))
    print("Finish loading train_fact, {} in total.".format(len(train_fact)))

    gold_cnt, pred_cnt, tp_cnt, ign_cnt = 0, 0, 0, 0
    if cluster_mappings is None: # pred_clusters == gold_clusters
        for p_relations, g_relations in zip(pred_relations, gold_relations):
            gold_cnt += len(g_relations)
            pred_cnt += len(p_relations)
            tp_cnt += get_relation_mapping(p_relations, g_relations)[0]
    else:
        if vertexSets is None:
            for cluster_mapping, p_relations, g_relations in zip(cluster_mappings, pred_relations, gold_relations):
                new_p_relations = [{'h': cluster_mapping[r['h']], 't': cluster_mapping[r['t']], 'r': r['r']} for r in p_relations]
                gold_cnt += len(g_relations)
                pred_cnt += len(p_relations)
                for pr in new_p_relations:
                    if pr in g_relations:
                        tp_cnt += 1
            precision = tp_cnt / (pred_cnt + 1e-7)
            recall = tp_cnt / gold_cnt
            f1 = (precision * recall * 2) / (precision + recall + 1e-7) 

            return f1, precision, recall

            # tp_cnt += get_relation_mapping(new_p_relations, g_relations)[0]
        for cluster_mapping, p_relations, g_relations, vertexSet in zip(cluster_mappings, pred_relations, gold_relations, vertexSets):
            new_p_relations = [{'h': cluster_mapping[r['h']], 't': cluster_mapping[r['t']], 'r': r['r']} for r in p_relations]
            gold_cnt += len(g_relations)
            pred_cnt += len(p_relations)

            for pr in new_p_relations:
                if pr in g_relations:
                    tp_cnt += 1
                    flag = False
                    for m1 in vertexSet[pr['h']]:
                        for m2 in vertexSet[pr['t']]:
                            if (m1['name'], m2['name'], docred_id2rel[pr['r']]) in train_fact:
                                flag = True
                                break
                    if flag:
                        ign_cnt += 1
            # tp_cnt += get_relation_mapping(new_p_relations, g_relations)[0]
    precision = tp_cnt / (pred_cnt + 1e-7)
    recall = tp_cnt / gold_cnt
    f1 = (precision * recall * 2) / (precision + recall + 1e-7) 

    ign_p = (tp_cnt - ign_cnt) / (pred_cnt - ign_cnt + 1e-7)
    ign_f1 = (ign_p * recall * 2) / (ign_p + recall + 1e-7)

    return f1, precision, recall, ign_f1

# for ign f1
def gen_train_facts(data_file_name, truth_dir):
    if not os.path.isdir(truth_dir):
        os.makedirs(truth_dir)
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train