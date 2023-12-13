import torch
import torch.nn.functional as F
import numpy as np
import math

def process_long_input(config, model, input_ids, attention_mask):
    return process_two_segments(config, model, input_ids, attention_mask)

def process_two_segments(config, model, input_ids, attention_mask):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor([config.bos_token_id]).to(input_ids)
    end_tokens = torch.tensor([config.eos_token_id]).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    max_len = config.max_position_embeddings
    # fix roberta issue
    if config.model_type == "roberta":
        max_len = 512
    if c <= max_len:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= max_len:
                new_input_ids.append(input_ids[i, :max_len])
                new_attention_mask.append(attention_mask[i, :max_len])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :max_len - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - max_len + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :max_len]
                attention_mask2 = attention_mask[i, (l_i - max_len): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - max_len))
                att = F.pad(attention[i], (0, c - max_len, 0, c - max_len))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:max_len - len_end]
                mask1 = attention_mask[i][:max_len - len_end]
                att1 = attention[i][:, :max_len - len_end, :max_len - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - max_len + len_end))
                mask1 = F.pad(mask1, (0, c - max_len + len_end))
                att1 = F.pad(att1, (0, c - max_len + len_end, 0, c - max_len + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - max_len + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - max_len + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - max_len + len_start, c - l_i, l_i - max_len + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention

def process_multiple_segments(config, model, input_ids, attention_mask, overlapping=100, max_segment=6):
    """
    Split the raw input into several chunks. Can either (1) use sliding window with dynamic size;
    (2) choose a fixed window size with overlapping parameter.
    """
    n, c = input_ids.size()
    start_token = torch.tensor([config.bos_token_id]).to(input_ids)
    end_token = torch.tensor([config.eos_token_id]).to(input_ids)
    pad_token = torch.tensor([config.pad_token_id]).to(input_ids)
    max_len = config.max_position_embeddings
    if overlapping is not None:
        hop = max_len - overlapping - 2
    else:
        hop = max_len - 2
    # fix roberta issue
    if config.model_type == "roberta":
        max_len = 512
    if c <= max_len:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= max_len:
                new_input_ids.append(input_ids[i, :max_len])
                new_attention_mask.append(attention_mask[i, :max_len])
                num_seg.append(1)
            else:
                n_s = 0
                left = 1
                flag = True
                while flag:
                    right = left + max_len - 2
                    pad = 0
                    if right >= l_i - 1:
                        right = l_i - 1
                        pad = (max_len - 2) - (right - left)
                        flag = False
                    if pad == 0:
                        new_input_ids.append(torch.cat([start_token, input_ids[i, left: right], end_token], dim=-1))
                        new_attention_mask.append(torch.tensor([1]*max_len).to(attention_mask))
                    else:
                        ids_with_pad = torch.cat([start_token, input_ids[i, left: right], end_token, pad_token.repeat(pad)], dim=-1)
                        mask_with_pad = torch.tensor([1]*(max_len - pad) + [0]*pad).to(attention_mask)
                        new_input_ids.append(ids_with_pad)
                        new_attention_mask.append(mask_with_pad)
                    n_s += 1
                    left += hop
                num_seg.append(n_s)
        
        # print("len: ", len(input_ids[i]), "num_seg: ", num_seg)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        hidden_dim = sequence_output.size(-1)
        num_head = attention.size(1)
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - max_len))
                att = F.pad(attention[i], (0, c - max_len, 0, c - max_len))
                new_output.append(output)
                new_attention.append(att)
            else:
                output = torch.zeros(c, hidden_dim).to(sequence_output)
                att = torch.zeros(num_head, c, c).to(attention)
                mask = torch.zeros(c).to(attention_mask)
                bias = 0
                for j in range(n_s):
                    left, right = 1, max_len - 1
                    if j == 0:
                        left = 0
                    if j == n_s - 1:
                        right = int(attention_mask[i+j].sum().item())
                    # print("bias: {}, left: {}, right: {}, c: {}".format(bias, left, right, c))
                    segment_output = sequence_output[i+j][left:right]
                    segment_attention = attention[i+j][:, left:right, left:right]
                    segment_mask = attention_mask[i+j][left:right]
                    segment_output = F.pad(segment_output, (0, 0, bias, c-(bias+right-left)))
                    segment_attention = F.pad(segment_attention, (bias, c-(bias+right-left), bias, c-(bias+right-left)))
                    segment_mask = F.pad(segment_mask, (bias, c-(bias+right-left)))
                    output += segment_output
                    att += segment_attention
                    mask += segment_mask
                    if j == 0:
                        bias += hop + 1
                    else:
                        bias += hop
                mask += 1e-10
                output = output / mask.unsqueeze(-1)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention