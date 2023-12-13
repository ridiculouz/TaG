import argparse
import os
from tqdm import tqdm
import json

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
# from apex import amp

from model import MEModel
from evaluation import *
from data import read_dataset

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="docred", type=str)

    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_epoch", default=40, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    
    parser.add_argument("--device", default="cuda:0", type=str,
                        help="The running device.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--notes", default="", type=str)

    return parser.parse_args()

def train(args, model: MEModel, train_features, dev_features, test_features=None):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(root_dir)
    result_dir = os.path.join(root_dir, f"result/{args.dataset}/me")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    def finetune(features, optimizer, num_epoch, num_steps):
        best_dev_score = -1
        curr_test_score = -1
        dataloader = DataLoader(features, batch_size=args.train_batch_size, 
                                shuffle=True, collate_fn=collate_fn)
        train_iterator = range(num_epoch)
        total_steps = (len(dataloader) // args.gradient_accumulation_steps) * num_epoch
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                   num_training_steps=total_steps)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epoch // 4) + 1)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            cum_loss = torch.tensor(0.0).to(args.device)
            for step, batch in tqdm(enumerate(dataloader), desc='train epoch {}'.format(epoch)):
                model.train()
                inputs = {"input_ids": batch["input_ids"].to(args.device),
                          "attention_mask": batch["attention_mask"].to(args.device),
                          "label": batch["label"].to(args.device)}
                loss = model.compute_loss(**inputs)
                loss = loss / args.gradient_accumulation_steps
                cum_loss += loss
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    cum_loss = torch.tensor(0.0).to(args.device)
                if step == len(dataloader) - 1:
                    dev_score, dev_coverage, dev_output = evaluate(args, model, dev_features, tag="dev")
                    if test_features is not None:
                        test_score, test_coverage, test_output = evaluate(args, model, test_features, tag="test")
                    print("dev f1: {} | test f1: {}".format(dev_score, test_score))
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        if args.save_path != "":
                            save_dir = os.path.dirname(args.save_path)
                            if not os.path.isdir(save_dir):
                                os.makedirs(save_dir)
                            torch.save(model.state_dict(), args.save_path)
                        dev_file = os.path.join(result_dir, "{}_dev.json".format(args.notes))
                        f_dev =  open(dev_file, 'w', encoding='utf-8')
                        for entry in dev_output:
                            jsonstr = json.dumps(entry)
                            f_dev.write(jsonstr + "\n")
                        f_dev.close()

                        curr_test_score = test_score
                        test_file = os.path.join(result_dir, "{}_test.json".format(args.notes))
                        f_test = open(test_file, 'w', encoding='utf-8')
                        for entry in test_output:
                            jsonstr = json.dumps(entry)
                            f_test.write(jsonstr + "\n")
                        f_test.close()
        log_file = os.path.join(result_dir, "{}_log".format(args.notes))
        f_log = open(log_file, 'a', encoding='utf-8')
        f_log.write("best dev f1: {} | curr test f1: {}\n".format(best_dev_score, curr_test_score))
        f_log.close()
        return num_steps
    
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "bert" in n], },
        {"params": [p for n, p in model.named_parameters() if not "bert" in n], "lr": 1e-4},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, args.num_epoch, num_steps)

def evaluate(args, model: MEModel, features, tag=""):
    model.eval()
    dataloader = DataLoader(features, batch_size=args.test_batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    gold = []
    pred = []
    doc_maps = []
    doc_lens = []
    useful_mentions = []

    for step, batch in tqdm(enumerate(dataloader), desc='eval'):
        gold.extend(batch["spans"])
        doc_maps.extend(batch["doc_map"])
        doc_lens.extend(batch["doc_len"])
        useful_mentions.extend(batch["useful_mention"])
        inputs = {"input_ids": batch["input_ids"].to(args.device),
                  "attention_mask": batch["attention_mask"].to(args.device)}
        with torch.no_grad():
            outputs = model.inference(**inputs)
            pred.extend(outputs)       

    f1, p, r = compute_me_f1(pred, gold)
    output_spans = []
    # get span mapping
    # compute coverage
    cnt_all, cnt_tp = 0, 0
    for ps, gs, us, doc_map, doc_len in zip(pred, gold, useful_mentions, doc_maps, doc_lens):
        res = {'tp': [], 'fp': [], 'fn': [], 'useful_fn': []}
        # construct reverse doc map
        reversed_doc_map = {}
        for i_sent, sent in enumerate(doc_map):
            for word in sent.keys():
                reversed_doc_map[sent[word]] = [i_sent, word]
        for i in range(doc_len):
            if i not in reversed_doc_map:
                for j in range(i - 1, -1, -1):
                    if j in reversed_doc_map:
                        reversed_doc_map[i] = reversed_doc_map[j]
                        break
            if i not in reversed_doc_map:
                raise ValueError("Unexpected!")
        # be careful about the offset
        for s in ps:
            # need to convert back to original span
            if s in gs:
                res['tp'].append([reversed_doc_map[s[0]-1], reversed_doc_map[s[1]-1]])
            else:
                res['fp'].append([reversed_doc_map[s[0]-1], reversed_doc_map[s[1]-1]])            
        for s in gs:
            if s not in ps:
                res['fn'].append([reversed_doc_map[s[0]-1], reversed_doc_map[s[1]-1]])
        for s in us:
            if s not in ps:
                res['useful_fn'].append([reversed_doc_map[s[0]-1], reversed_doc_map[s[1]-1]])
        cnt_all += len(us)
        cnt_tp += len(res["useful_fn"])
        res['stat'] = {'tp': len(res["tp"]), 'fp': len(res["fp"]), 'fn': len(res["fn"]), 'useful_fn': len(res["useful_fn"])}
        output_spans.append(res)
    coverage = cnt_tp * 1.0 / (cnt_all + 1e-7)
    return f1, coverage, output_spans

def collate_fn(batch):
    """
    Reference: https://github.com/wzhouad/ATLOP/
    """
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    label = [f["label"] + [0] * (max_len - len(f["label"])) for f in batch]
    spans = [f["spans"] for f in batch]
    doc_map = [f["doc_map"] for f in batch]
    doc_len = [f["doc_len"] for f in batch]
    useful_mention = [f["useful_mention"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    output = {"input_ids": input_ids, "attention_mask": attention_mask, "label": label, "spans": spans, 
                "doc_map": doc_map, "doc_len": doc_len, "useful_mention": useful_mention}
    return output

if __name__ == "__main__":

    args = get_opt()
    # get device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    # get config and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if config.model_type == "bert":
        bos_token, eos_token, pad_token = "[CLS]", "[SEP]", "[PAD]"
    elif config.model_type == "roberta":
        bos_token, eos_token, pad_token = "<s>", "</s>", "<pad>"
    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    config.bos_token_id, config.eos_token_id, config.pad_token_id = \
        bos_token_id, eos_token_id, pad_token_id

    config.dataset = args.dataset
    # get model
    encoder = AutoModel.from_pretrained(args.model_name_or_path)
    model = MEModel(config, encoder)
    model.to(device)

    if args.load_path == "":
        train_features = read_dataset(tokenizer, split='train_annotated', dataset=args.dataset, task='me')
        dev_features = read_dataset(tokenizer, split='dev', dataset=args.dataset, task='me')
        test_features = read_dataset(tokenizer, split='test', dataset=args.dataset, task='me')
        train(args, model, train_features, dev_features, test_features)
    else:
        raise ValueError("Test only mode is unimplemented!")
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))