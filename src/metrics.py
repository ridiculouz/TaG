# implementation of coref metrics
import numpy as np
from scipy.optimize import linear_sum_assignment

def muc(predicted, gold):
    num, den = 0, 0
    # build mapping
    mapping = dict()
    for i, e in enumerate(gold):
        for m in e:
            mapping[m] = i
    for c in predicted:
        den += len(c) - 1
        num += len(c)
        # find partition
        linked = set()
        for m in c:
            if m in mapping:
                linked.add(mapping[m])
            else:
                num -= 1
        num -= len(linked)
    return num, den

def b_cubed(predicted, gold):
    num, den = 0, 0
    # build mapping
    mapping = dict()
    for i, e in enumerate(gold):
        for m in e:
            mapping[m] = i

    for c in predicted:
        if len(c) == 0:
            continue
        if len(c) == 1:
            num += 1
            den += 1
            continue
        correct = 0
        reversed_mapping = dict()
        for k, v in mapping.items():
            if k in c:
                if v in reversed_mapping:
                    reversed_mapping[v] += 1
                else:
                    reversed_mapping[v] = 1
        for i_c, cnt in reversed_mapping.items():
            correct += cnt * cnt

        num += correct / len(c)
        den += len(c)
                
    return num, den

def ceafe(predicted, gold):
    scores = np.zeros((len(gold), len(predicted)))
    for i in range(len(gold)):
        for j in range(len(predicted)):
            scores[i, j] = phi4(gold[i], predicted[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(predicted), similarity, len(gold)

def phi4(key, response):
    return 2 * len([m for m in key if m in response]) / (len(key) + len(response))