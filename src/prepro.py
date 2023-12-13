
import argparse
import json
import os

def get_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dev_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--dataset", type=str, default="docred")

    return parser.parse_args()

def generate_gc_data(dev_file, test_file, dataset='docred'):
    """
    Generate data for GC (graph composition, i.e. coreference resolution + relation extraction).
    Use span prediction from ME.
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, f'data/{dataset}/')
    # process train
    raw_train = json.load(open(os.path.join(data_dir, 'train_annotated.json'), 'r', encoding='utf-8'))
    for sample in raw_train:
        entity_len = []
        spans = []
        for e in sample['vertexSet']:
            entity_len.append(len(e))
            for m in e:
                ms = [[m['sent_id'], m['pos'][0]], [m['sent_id'], m['pos'][1]]]
                spans.append(ms)
        sample['entity_len'] = entity_len
        sample['spans'] = spans
    json.dump(raw_train, open(os.path.join(data_dir, 'train_annotated_gc.json'), 'w', encoding='utf-8'))
    # process dev/test
    for split, spanfile in zip(['dev', 'test'], [dev_file, test_file]):
        raw_data = json.load(open(os.path.join(data_dir, '{}.json'.format(split)), 'r', encoding='utf-8'))
        span_data = []
        with open(spanfile, 'r') as f:
            for line in f:
                span_data.append(json.loads(line))
        assert len(span_data) == len(raw_data)
        for i, sample in enumerate(raw_data):
            spans = span_data[i]['tp'] + span_data[i]['fp']
            sample['spans'] = spans
        json.dump(raw_data, open(os.path.join(data_dir, '{}_gc.json'.format(split)), 'w', encoding='utf-8'))
    return    

def delete_invalid_spans(src, dst):
    data = []
    tp, fp, fn = 0, 0, 0
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    for entry in data:
        new_spans = []
        tp += len(entry['tp'])
        fn += len(entry['fn'])
        for span in entry['fp']:
            if order_score(span[0]) >= order_score(span[1]):
                continue
            new_spans.append(span)
            fp += 1
        entry['fp'] = new_spans
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = p*r*2/(p+r+1e-7)
    print(p, r, f1)
    with open(dst, 'w', encoding='utf-8') as f:
        for entry in data:
            jsonstr = json.dumps(entry, ensure_ascii=False)
            f.write(jsonstr+'\n')
    # json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

def order_score(pos):
    return pos[0]*100 + pos[1]

if __name__ == '__main__':
    args = get_opt()

    delete_invalid_spans(args.dev_file, args.dev_file)
    delete_invalid_spans(args.test_file, args.test_file)
    generate_gc_data(args.dev_file, args.test_file, args.dataset)