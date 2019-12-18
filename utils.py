import json
import logging
import os
import torch
import warnings
from itertools import chain

import torch
import torch.nn.functional as F


# Migration notes: needs to be changed to dictionary for adding special tokens
ATTR_TO_SPECIAL_TOKEN = {"bos_token": "<bos>", 
                  "eos_token": "<eos>",
                  "additional_special_tokens": ["<speaker1>", "<speaker2>"],
                  "pad_token": "<pad>"}

logger = logging.getLogger(__file__)

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """

    # import pdb; pdb.set_trace()
    special_tokens = ['<bos>', '<eos>', '<speaker1>', '<speaker2>']
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(special_tokens) 

    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    # print(sequence)
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

    return instance, sequence

# add special tokens to the tokenizer 
def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # returns 0 and doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens)



# filter output candidates when decoding 
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(personality, history, tokenizer, model, args, current_output=None):

    special_tokens = ['<bos>', '<eos>', '<speaker1>', '<speaker2>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    if current_output is None:
        current_output = []

    for i in range(args['max_length']):
        instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args['device']).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args['device']).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args['temperature']
        logits = top_filtering(logits, top_k=args['top_k'], top_p=args['top_p'])
        probs = F.softmax(logits, dim=-1)


        prev = torch.topk(probs, 1)[1] if args['no_sample'] else torch.multinomial(probs, 1)
        if i < args['min_length'] and prev.item() in special_tokens_ids:
            count = 0 
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1: 
                    break
                prev = torch.multinomial(probs, num_samples=1)
                count += 1 
                # to prevent endless looping
                if count == 10: 
                    break 

        # import pdb; pdb.set_trace()
        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

