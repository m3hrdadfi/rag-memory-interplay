import copy
import itertools
import os
from collections import defaultdict
from functools import reduce

import matplotlib.ticker as mtick
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from . import nethook
from .utils import AttrDict, _to_cuda

DECODER_START_TOKEN_ID = 0
DECODER_DUMMY_TOKEN_ID = 32099
IGNORE_INDEX = -100
DECODER_END_TOKEN_ID = 1
# IGNORE_TOKEN_IDS = [DECODER_START_TOKEN_ID, DECODER_DUMMY_TOKEN_ID, DECODER_END_TOKEN_ID, IGNORE_INDEX]
IGNORE_TOKEN_IDS = [DECODER_START_TOKEN_ID, IGNORE_INDEX]

KIND_MAP_LIST = {
    "attn": "SelfAttention",
    "cross_attn": "EncDecAttention",
    "mlp": "DenseReluDense"
}


def inverse_mapping(f):
    return f.__class__(map(reversed, f.items()))


def filter_unwanted_ids(tensor, ignore_tensor, return_index=False):
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
        
    mask = tensor.unsqueeze(1).eq(torch.tensor(ignore_tensor).to(tensor.device)).any(1)
    indices = torch.arange(len(tensor)).to(tensor.device)

    return tensor[~mask] if not return_index else indices[~mask]
    

def encode_passages(batch, tokenizer, max_length=None):
    bsz = len(batch)
    n = max([len(example) for example in batch])
    batch = [example + [""] * (n - len(example)) for example in batch]
    batch = reduce(lambda a, b: a + b, batch)
    tokens = tokenizer(
        batch,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
    return tokens


def reader_tokenize(model, query, target, target_tokens=None):
    if target_tokens is None:
        if model.opt.decoder_prompt_format is not None:
            modified_query = [model.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
            target = [q + t for (q, t) in zip(modified_query, target)]

            query_mask = model.reader_tokenizer(
                modified_query,
                # max_length=model.opt.target_maxlength,
                # padding="max_length",
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["attention_mask"]

        if model.opt.decoder_format is not None:
            target = [model.opt.decoder_format.format(target=t) for t in target]

        target = [t + "</s>" if not t.endswith("</s>") else t for t in target]
        target_maxlength = max([len(model.reader_tokenizer.encode(item)) for item in target])

        target_tokens = model.reader_tokenizer(
            target,
            # max_length=model.opt.target_maxlength,
            max_length=target_maxlength,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

    decoder_input_ids = model.reader._shift_right(target_tokens["input_ids"])
    labels = target_tokens["input_ids"].masked_fill(~target_tokens["attention_mask"].bool(), IGNORE_INDEX)

    # If decoder prompt is not None mask labels such that the model is not trained to predict the prompt
    if model.opt.decoder_prompt_format is not None:
        target_maxlength = max([len(model.reader_tokenizer.encode(item)) for item in modified_query])

        query_mask = model.reader_tokenizer(
            modified_query,
            # max_length=model.opt.target_maxlength,
            max_length=target_maxlength,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["attention_mask"]

        padding = torch.zeros((query_mask.size(0), target_tokens["input_ids"].size(-1) - query_mask.size(-1)))
        query_mask = torch.cat([query_mask, padding], dim=1)
        labels = labels.masked_fill(query_mask.bool(), IGNORE_INDEX)

    return labels.cuda(), decoder_input_ids.cuda()


def append_query(query, passages):
    q_format = "{query}"
    p_format = "{query} title: {title} context: {text}"
    return [p_format.format(query=query, **p) if p.get("text") and len(p["text"]) > 0 else q_format.format(query=query) for p in passages]


def make_inputs(model, prompts, prompt_is_dict=True, n_context=1, passages_key="passages"):
    if prompt_is_dict:

        query = [prompt["query"] for prompt in prompts]
        answers = ["<extra_id_0>" + prompt.get("answers", [""])[0].strip() for prompt in prompts]

        labels, decoder_input_ids = reader_tokenize(model, query, answers)
        retrieved_passages = [prompt[passages_key][:n_context] if n_context > 0 else [{"title": "", "text": ""}] for prompt in prompts]

        query_passages = [append_query(q, p) for q, p in zip(query, retrieved_passages)]
        tokens = encode_passages(query_passages, model.reader_tokenizer)

        tokens = _to_cuda(tokens)
    else:
        tokens = encode_passages(list(map(lambda x: [x], prompts)), model.reader_tokenizer)
        tokens = _to_cuda(tokens) 
        decoder_input_ids = torch.tensor([DECODER_START_TOKEN_ID]).unsqueeze(0).repeat(tokens["input_ids"].size(0), 1)

        query = None
        labels = None


    return AttrDict({
        "query": query,
        "input_ids": tokens['input_ids'],
        "attention_mask": tokens['attention_mask'],
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    })


def joint_probs(probs, min_prob=1e-12, smoothing_const=False, thresholding=False):

    if smoothing_const:
        probs = probs + min_prob

    if thresholding:
        probs = torch.where(probs < min_prob, torch.tensor(min_prob).to(probs.device), probs)

    if len(probs.flatten()) == 0:
        j_prob = torch.tensor(0).to(probs.device)
    elif len(probs.flatten()) < 2:
        j_prob = probs.flatten()
        j_prob = j_prob.squeeze()
    else:
        # # log_probs = torch.nn.functional.log_softmax(probs, dim=-1)
        log_probs = torch.log(probs)
        log_joint_prob = log_probs.sum(dim=-1)
        j_prob = torch.exp(log_joint_prob)
        # j_prob = probs.prod()
        j_prob = j_prob.squeeze()

    return j_prob

@torch.no_grad()
def generate_by_greedy_search(model, input_ids, attention_mask, decoder_input_ids, max_new_tokens=32):
    bsz = input_ids.size(0)
    logits = torch.tensor([]).cuda()
    
    probs = [torch.tensor([]).cuda() for _ in range(bsz)]
    generated_sequences = [torch.tensor([], dtype=torch.long).cuda() for _ in range(bsz)]
    # generated_sequences = list(torch.unbind(decoder_input_ids, dim=0))
    sequence_completed = torch.zeros(bsz, dtype=torch.bool).cuda()
    for it in range(max_new_tokens):
        if sequence_completed.all() and it >= max_new_tokens:
            break  # Stop the loop if all sequences have completed

        outputs = model.reader(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        p, next_tokens = torch.max(next_token_probs, dim=-1)
        p = p.unsqueeze(-1)
        next_tokens = next_tokens.unsqueeze(-1)
        
        # probs = torch.cat([probs, p], dim=-1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
        logits = torch.cat([logits, next_token_logits.unsqueeze(1)], dim=1)

        # # Break if the next token is the end of sentence token
        # if torch.all(next_tokens == model.reader.config.eos_token_id):
        #     break

        # Update sequences that have not yet completed
        for i in range(bsz):
            if not sequence_completed[i]:
                generated_sequences[i] = torch.cat([generated_sequences[i], next_tokens[i]], dim=-1)
                probs[i] = torch.cat([probs[i], p[i]], dim=-1)
                # Check if the EOS token is generated, marking the sequence as completed
                if next_tokens[i] == model.reader.config.eos_token_id:
                    sequence_completed[i] = True
    
    
    j_probs = [joint_probs(probs[i][filter_unwanted_ids(generated_sequences[i], IGNORE_TOKEN_IDS, return_index=True)]) for i in range(len(probs))]
    j_probs = torch.stack(j_probs).to(j_probs[0].device)

    return AttrDict({"sequences": generated_sequences, "logits": logits, "probs": probs, "j_probs": j_probs})


def generate_with_target_prob(model, input_ids, attention_mask, decoder_input_ids, target_ids=None, max_new_tokens=32):
    r = generate_by_greedy_search(model, input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, max_new_tokens=max_new_tokens)

    probs = torch.tensor([]).cuda()
    j_probs = torch.tensor([]).cuda()

    logits = []

    if r.logits.size(1) >= target_ids.size(1):
        for i in range(target_ids.size(0)):
            next_token_probs = torch.softmax(r.logits[i], dim=-1)
            mask_target_ids = target_ids[i][target_ids[i] != IGNORE_INDEX]
            p = next_token_probs[range(len(mask_target_ids)), mask_target_ids]
            jp = joint_probs(p[filter_unwanted_ids(target_ids[i], IGNORE_TOKEN_IDS, return_index=True)])

            logits.append(r.logits[i].unsqueeze(0))
            probs = torch.cat([probs, p.unsqueeze(0)], dim=0)
            j_probs = torch.cat([j_probs, jp.unsqueeze(0)], dim=0)
        
        logits = torch.cat(logits, dim=0)
    else:
        logits = torch.tensor([]).cuda()


    r = r.concat(AttrDict({"target_sequences": list(torch.unbind(target_ids, dim=0)), "target_logits": logits, "target_probs": list(torch.unbind(probs, dim=0)), "target_j_probs": j_probs}))

    return r


def predict_from_input(model, input_ids, attention_mask, decoder_input_ids, target_ids, max_new_tokens=8):    
    gene_output = generate_with_target_prob(model, input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, target_ids=target_ids, max_new_tokens=max_new_tokens)

    return AttrDict({
        "predicted": [
            model.reader_tokenizer.decode(gene_output.sequences[i][gene_output.sequences[i] != IGNORE_INDEX], skip_special_tokens=False) 
            for i in range(len(gene_output.sequences))
        ],
        "predicted_text": [
            model.reader_tokenizer.decode(gene_output.sequences[i][gene_output.sequences[i] != IGNORE_INDEX], skip_special_tokens=True) 
            for i in range(len(gene_output.sequences))
        ],
        "predicted_ids": gene_output.sequences,
        "predicted_probs": gene_output.probs,
        "predicted_j_probs": gene_output.j_probs,
        # "predicted_logits": gene_output.logits,

        "target": [
            model.reader_tokenizer.decode(gene_output.target_sequences[i][gene_output.target_sequences[i] != IGNORE_INDEX], skip_special_tokens=False) 
            for i in range(len(gene_output.target_sequences))
        ],
        "target_text": [
            model.reader_tokenizer.decode(gene_output.target_sequences[i][gene_output.target_sequences[i] != IGNORE_INDEX], skip_special_tokens=True) 
            for i in range(len(gene_output.target_sequences))
        ],
        "target_ids": [
            gene_output.target_sequences[i][gene_output.target_sequences[i] != IGNORE_INDEX]
            for i in range(len(gene_output.target_sequences))
        ],
        # "target_ids": gene_output.target_sequences,
        "target_probs": gene_output.target_probs,
        "target_j_probs": gene_output.target_j_probs,
        # "target_logits": gene_output.target_logits,
    })


def predict_token(model, prompts, n_context=1):
    results = []

    inputs = make_inputs(model, prompts, prompt_is_dict=True, n_context=n_context)

    input_ids = inputs.input_ids.cuda().view(inputs.input_ids.size(0), -1)
    attention_mask = inputs.attention_mask.cuda().view(inputs.attention_mask.size(0), -1)
    labels = inputs.labels.cuda()
    decoder_input_ids = torch.tensor([model.reader.config.decoder_start_token_id]).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)

    cfg = model.reader.encoder.config
    cfg.bsz = inputs.input_ids.size(0)
    cfg.n_context = inputs.input_ids.size(1)

    results = predict_from_input(
        model, 
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        decoder_input_ids=decoder_input_ids,
        target_ids=labels,
        max_new_tokens=labels.size(1)
    )
    return results


def tokenize(tokenizer, text):
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(text)[:-1])

def search_text_indices(tokens, sub_tokens, start_index=0, end_index=-1):
    sub_tokens_len = len(sub_tokens)
    indices = []
    for i in range(start_index, len(tokens) - sub_tokens_len + 1):
        if i < start_index:
            continue

        if end_index > 0 and i > end_index:
            continue

        if tokens[i:i + sub_tokens_len] == sub_tokens:
            indices.append((i, i + sub_tokens_len))
    return indices


def find_token_ranges(tokenizer, input_ids, search_text, bounds=None):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sub_tokens = tokenize(tokenizer, search_text)
    start_index = 0 if not isinstance(bounds, list) or len(bounds) != 2 else bounds[0]
    end_index = -1 if not isinstance(bounds, list) or len(bounds) != 2 else bounds[1]
    
    return search_text_indices(tokens, sub_tokens, start_index, end_index)

def prompt_segmenter(tokens_prompts):
    bounds = []
    for tokens in tokens_prompts:
        query_bound = [0, tokens.index("▁answer") + 4]
        context_bound = [tokens.index("▁answer") + 4, tokens.index("</s>") + 1]
        bounds.append({
            "query": query_bound,
            "context": context_bound,
        })
    
    return bounds



def collect_embedding_std(
    model, 
    prompts, 
    attribute="subj", 
    bounds_mode=None,  # None, "query", "context"
    n_context=1,
):
    alldata = []
    inputs = make_inputs(model, prompts, prompt_is_dict=True, n_context=n_context)

    cfg = model.reader.encoder.config
    cfg.n_context = inputs.input_ids.size(1)
    cfg.bsz = inputs.input_ids.size(0)

    input_ids = inputs.input_ids.cuda().view(inputs.input_ids.size(0), -1)
    attention_mask = inputs.attention_mask.cuda().view(inputs.attention_mask.size(0), -1)
    decoder_input_ids = inputs.decoder_input_ids.cuda()

    input_tokens = [model.reader_tokenizer.convert_ids_to_tokens(input_ids[i]) for i in range(input_ids.size(0))]
    input_segments = prompt_segmenter(input_tokens)
    
    if attribute in ["subj", "obj", "prop"]:
        indices_ranges_list = [find_token_ranges(model.reader_tokenizer, input_ids[i], p[attribute], bounds=input_segments[i][bounds_mode] if bounds_mode else bounds_mode) for i, p in enumerate(prompts)]
        # print(f"indices_ranges_list: {indices_ranges_list}")
    elif attribute in ["context"]:
        indices_ranges_list = [[tuple(input_segments[i]["context"])] for i, p in enumerate(prompts)]
    elif attribute in ["prompt"]:
        indices_ranges_list = [[(0, input_segments[i]["context"][-1])] for i, p in enumerate(prompts)]
    else:
        raise ValueError(f"Attribute `{attribute}` is not defined!")


    with nethook.Trace(model, "reader.encoder.embed_tokens", stop=True) as t:
        model.reader(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    embed_tokens = t.output
    if embed_tokens.size(1) != input_ids.size(1):
        raise Exception("There is something wrong with this approach!")

    for i, indices_ranges in enumerate(indices_ranges_list):
        tokens_representations = []
        for ranges in indices_ranges:
            tokens_representations.append(embed_tokens[i, ranges[0]:ranges[1]])

        alldata.append(torch.cat(tokens_representations).mean(dim=0, keepdim=True))


    # print(f"alldata: {alldata}")
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level

def reversed_token_ranges(constraint_range, exclude_ranges):
    constraint_start, constraint_end = constraint_range
    exclude_ranges = sorted(exclude_ranges)
    result_ranges = []
    current_start = constraint_start

    for exclude_start, exclude_end in exclude_ranges:
        if exclude_start >= constraint_end:
            break  # No need to process further as we are beyond the constraint range

        if exclude_start > current_start:
            result_ranges.append((current_start, exclude_start))
        
        if exclude_end > current_start:
            current_start = exclude_end
    
    # Add the last range from the end of the last exclusion to the constraint end
    if current_start < constraint_end:
        result_ranges.append((current_start, constraint_end))

    return result_ranges


def repeat_to_size(tensor, target_first_dim):
    current_size = tensor.size(0)
    
    # Calculate how many times to repeat the tensor to exceed or match the target size
    repeat_times = target_first_dim // current_size
    extra_slices_needed = target_first_dim % current_size
    
    # Repeat the tensor to get as close to the target size as possible
    repeated_tensor = tensor.repeat(repeat_times, 1, 1)
    
    # If extra slices are needed, take a portion of the tensor and concatenate it
    if extra_slices_needed > 0:
        extra_slices = tensor[:extra_slices_needed].repeat(1, 1, 1)
        final_tensor = torch.cat([repeated_tensor, extra_slices], dim=0)
    else:
        final_tensor = repeated_tensor
    
    return final_tensor


def trace_with_patch(
    model,  # The model
    input_dict,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answer_tokens,  # Answer probabilities to collect
    attributes_loc,  # a ranges of tokens to corrupt (begin, end)
    attributes_noise,
    trace_layers=None,  # List of traced outputs to return
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    noise_generated=False,
    noise_data=None,
    cf_patch=None
):

    # For reproducibility, use pseudorandom noise
    rs = np.random.RandomState(1)  

    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)


    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    
    embed_layername = "reader.encoder.embed_tokens"


    def untuple(x):
        return x[0] if isinstance(x, tuple) else x


    # Define the model-patching rule.
    dim = input_dict["input_ids"].size(-1)
    def patch_rep(x, layer):
        nonlocal noise_generated, noise_data

        if layer == embed_layername and ((len(untuple(x).size()) == 3 and untuple(x).size(1) == dim) or (len(x.size()) == 3 and x.size(1) == dim)):
            
            # Corrupt a range of token embeddings on batch items x[1:] for each specified range
            for attribute, locations in attributes_loc.items():
                if len(locations) == 0:
                    continue

                b, e = locations[0]
                if not noise_generated[attribute]:
                    if not isinstance(attributes_noise[attribute], float):
                        noise = torch.from_numpy(np.array(attributes_noise[attribute])).to(x.device)
                        noise = repeat_to_size(noise, x.size(0) - 1)
                        noise_data[attribute] = noise
                    else:
                        noise_data[attribute] = (
                            attributes_noise[attribute] * torch.from_numpy(
                                prng(x.shape[0] - 1, e - b, x.shape[2])
                            )
                        ).to(x.device)
                    
                    noise_generated[attribute] = True
                
                for location in locations:
                    b, e = location

                    if replace:
                        x[1:, b:e] = noise_data[attribute]
                    else:
                        x[1:, b:e] += noise_data[attribute]

            return x

        if layer not in patch_spec:
            return x

        # Restore the uncorrupted hidden state for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            if cf_patch and layer in cf_patch:
                h[1:, t] = cf_patch[layer][1:, t]
            else:
                h[1:, t] = h[0, t]
            
        return x

    # Run the patched model in inference with the patching rules defined.
    additional_layers = [] if trace_layers is None else trace_layers

    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers, 
        edit_output=patch_rep,
    ) as td:
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]
        decoder_input_ids = input_dict["decoder_input_ids"]

        outputs = model.reader(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits

        next_token = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1).unsqueeze(-1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
    
    probs = torch.softmax(logits[1:, -1, :], dim=-1).mean(dim=0)[answer_tokens]

    # Collect all activations together to return if tracing all layers.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return logits, noise_generated, noise_data, decoder_input_ids, probs, all_traced

    return logits, noise_generated, noise_data, decoder_input_ids, probs



def cf_patch_extract(model, layerlist, cf_input_dict):
    cf_patch = {}
    for layer in layerlist:
        with nethook.Trace(model, layer[1], stop=True) as t:
            model.reader(
                input_ids=cf_input_dict["input_ids"],
                attention_mask=cf_input_dict["attention_mask"],
                decoder_input_ids=cf_input_dict["decoder_input_ids"],
            )

        cf_patch[layer[1]] = t.output[0] if isinstance(t.output, tuple) else t.output
    
    return cf_patch


def trace_important_states(
    model, 
    num_layers, 
    input_dict, 
    answer_tokens,  
    attributes_loc, 
    attributes_noise,
    cf_tokens=None,
    cf_input_dict=None,
    module="block", 
    component="encoder", 
    uniform_noise=False, 
    replace=False,
    desc="trace_important_states"
):
    cf_tokens = cf_tokens if cf_tokens and isinstance(cf_tokens, list) else []
    ntoks = input_dict["input_ids"].shape[-1]
    table = []

    for tnum in tqdm(range(ntoks), total=ntoks,  desc=desc):
        row = []
        for layer in range(num_layers):

            decoder_input_ids = input_dict["decoder_input_ids"]
            noise_generated = {k: False for k in attributes_loc.keys()}
            noise_data = {k: None for k in attributes_loc.keys()}
            agg_ans_scores, agg_cf_scores, agg_scores = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()


            for it in range(len(answer_tokens)):
                layerlist = [(tnum if component == "encoder" else it, f"reader.{component}.{module}.{layer}")]
                cf_patch = None

                if cf_input_dict and isinstance(cf_input_dict, dict):
                    cf_patch = cf_patch_extract(model, layerlist, cf_input_dict)

                
                logits, noise_generated, noise_data, decoder_input_ids, score = trace_with_patch(
                    model=model, 
                    input_dict={
                        "input_ids": input_dict["input_ids"], 
                        "attention_mask": input_dict["attention_mask"], 
                        "decoder_input_ids": decoder_input_ids,
                    }, 
                    states_to_patch=layerlist, 
                    answer_tokens=answer_tokens[it], 
                    attributes_loc=attributes_loc if not isinstance(cf_patch, dict) or not cf_patch else {}, 
                    attributes_noise=attributes_noise if not isinstance(cf_patch, dict) or not cf_patch else None,
                    trace_layers=None,
                    uniform_noise=uniform_noise,
                    replace=replace,
                    noise_generated=noise_generated,
                    noise_data=noise_data,
                    cf_patch=cf_patch
                )

                # raise Exception("CHECKPOINT!")

                agg_scores = torch.cat([agg_scores, score.unsqueeze(0)], dim=0)
                p = torch.softmax(logits[1:, -1, :], dim=1)

                cf_scores, ans_scores = [], []
                if cf_tokens and len(cf_tokens) > 0:
                    for i, tokens in enumerate(cf_tokens):
                        cf_scores.append(p[i, cf_tokens[i][it]])
                        ans_scores.append(p[i, answer_tokens[it]])
                else:
                    for i in range(input_dict["input_ids"].shape[0] - 1):
                        cf_scores.append(torch.max(p[i], dim=-1)[0])
                        ans_scores.append(p[i, answer_tokens[it]])

                agg_cf_scores = torch.cat([agg_cf_scores, torch.tensor(cf_scores).to(agg_cf_scores.device).unsqueeze(1)], dim=1)
                agg_ans_scores = torch.cat([agg_ans_scores, torch.tensor(ans_scores).to(agg_ans_scores.device).unsqueeze(1)], dim=1)
            
            row.append({
                "score": joint_probs(agg_scores),
                "cf_score": joint_probs(agg_cf_scores),
                "ans_score": joint_probs(agg_ans_scores),
                "scores": agg_scores,
                "cf_scores": agg_cf_scores,
                "ans_scores": agg_ans_scores,
                "decoder_input_ids": decoder_input_ids[:, 1:]
            })
        table.append(row)
        
    return table

def trace_important_window(
    model, 
    num_layers, 
    input_dict, 
    answer_tokens,  
    attributes_loc, 
    attributes_noise,
    kind,
    kind_pos,
    cf_tokens=None,
    cf_input_dict=None,
    module="block",
    component="encoder", 
    window=10, 
    uniform_noise=False,
    replace=False,
    desc="trace_important_window"
):
    cf_tokens = cf_tokens if cf_tokens and isinstance(cf_tokens, list) else []
    ntoks = input_dict["input_ids"].shape[-1]

    table = []
    for tnum in tqdm(range(ntoks), total=ntoks, desc=desc):
        row = []
        for layer in range(0, num_layers):
            
            decoder_input_ids = input_dict["decoder_input_ids"]
            noise_generated = {k: False for k in attributes_loc.keys()}
            noise_data = {k: None for k in attributes_loc.keys()}
            agg_ans_scores, agg_cf_scores, agg_scores = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()

            for it in range(len(answer_tokens)):
                layerlist = [
                    (tnum if component == "encoder" else it, f"reader.{component}.{module}.{L}.layer.{kind_pos}.{kind}")
                    for L in range(
                        max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                    )
                ]
                cf_patch = None

                if cf_input_dict and isinstance(cf_input_dict, dict):
                    cf_patch = cf_patch_extract(model, layerlist, cf_input_dict)

                logits, noise_generated, noise_data, decoder_input_ids, score = trace_with_patch(
                    model=model, 
                    input_dict={"input_ids": input_dict["input_ids"], "attention_mask": input_dict["attention_mask"], "decoder_input_ids": decoder_input_ids}, 
                    states_to_patch=layerlist, 
                    answer_tokens=answer_tokens[it], 
                    attributes_loc=attributes_loc if not isinstance(cf_patch, dict) or not cf_patch else {}, 
                    attributes_noise=attributes_noise if not isinstance(cf_patch, dict) or not cf_patch else None,
                    trace_layers=None,
                    uniform_noise=uniform_noise,
                    replace=replace,
                    noise_generated=noise_generated,
                    noise_data=noise_data,
                    cf_patch=cf_patch
                )
                
                agg_scores = torch.cat([agg_scores, score.unsqueeze(0)], dim=0)
                p = torch.softmax(logits[1:, -1, :], dim=1)

                cf_scores, ans_scores = [], []
                if cf_tokens and len(cf_tokens) > 0:
                    for i, tokens in enumerate(cf_tokens):
                        cf_scores.append(p[i, cf_tokens[i][it]])
                        ans_scores.append(p[i, answer_tokens[it]])
                else:
                    for i in range(input_dict["input_ids"].shape[0] - 1):
                        cf_scores.append(torch.max(p[i], dim=-1)[0])
                        ans_scores.append(p[i, answer_tokens[it]])

                agg_cf_scores = torch.cat([agg_cf_scores, torch.tensor(cf_scores).to(agg_cf_scores.device).unsqueeze(1)], dim=1)
                agg_ans_scores = torch.cat([agg_ans_scores, torch.tensor(ans_scores).to(agg_ans_scores.device).unsqueeze(1)], dim=1)

            row.append({
                "score": joint_probs(agg_scores),
                "cf_score": joint_probs(agg_cf_scores),
                "ans_score": joint_probs(agg_ans_scores),
                "scores": agg_scores,
                "cf_scores": agg_cf_scores,
                "ans_scores": agg_ans_scores,
                "decoder_input_ids": decoder_input_ids[:, 1:]
            })

        table.append(row)
    return table

def adjust_cf_samples(counterfactuals, samples):
    if samples < len(counterfactuals):
        return counterfactuals[:samples]
    else:
        full_cycles = samples // len(counterfactuals)
        remainder_elements = samples % len(counterfactuals)
        return counterfactuals * full_cycles + counterfactuals[:remainder_elements]


def calculate_hidden_flow(
    model, 
    prompt, 
    attributes,
    attributes_noise,
    reversed_attributes=False,
    reversed_attributes_noise=0.0,
    counterfactuals=None, 
    samples=10, 
    window=10, 
    module="block", 
    component="encoder", 
    kind=None,  # "attn", "cross_attn", "mlp"
    n_context=1,
    uniform_noise=False,
    replace=False,
    num_layers=12,
    patch_type="clean"  # "clean", "counterfactual"
): 
    # attribute={"Iran": []}
    kind_pos = 0 if kind == "attn" else 2 if (kind == "mlp" and component == "decoder") else 1
    kind = KIND_MAP_LIST.get(kind)

    cf = counterfactuals if counterfactuals and isinstance(counterfactuals, list) else []

    prompts = [prompt] * (samples + 1)
    for p in prompts:
        p["cf_passages"] = p["passages"]
        
    if cf and isinstance(cf, list):
        cf = adjust_cf_samples(cf, samples)
        cf_prompts = []
        for item in cf:
            cf_prompt = copy.deepcopy(prompt)
            cf_prompt["cf_passages"] = [{"title": p["title"], "text": p["text"].replace(cf_prompt["answers"][0], item)} for p in cf_prompt["passages"]]
            cf_prompt["answers"] = [item]
            cf_prompts.append(cf_prompt)
        prompts = [prompt] + cf_prompts
    else:
        prompts = [prompt] * (samples + 1)
    
    
    inputs = make_inputs(model, prompts, prompt_is_dict=True, n_context=n_context)
    input_ids = inputs.input_ids.cuda().view(inputs.input_ids.size(0), -1)
    attention_mask = inputs.attention_mask.cuda().view(inputs.attention_mask.size(0), -1)
    labels = inputs.labels.cuda()
    __decoder_input_ids = torch.tensor([model.reader.config.decoder_start_token_id]).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)

    cf_input_dict = {}
    if patch_type == "counterfactual":
        cf_inputs = make_inputs(model, prompts, prompt_is_dict=True, n_context=n_context, passages_key="cf_passages")
        cf_input_ids = cf_inputs.input_ids.cuda().view(cf_inputs.input_ids.size(0), -1)
        cf_attention_mask = cf_inputs.attention_mask.cuda().view(cf_inputs.attention_mask.size(0), -1)
        cf_labels = cf_inputs.labels.cuda()
        __cf_decoder_input_ids = torch.tensor([model.reader.config.decoder_start_token_id]).unsqueeze(0).repeat(cf_input_ids.size(0), 1).to(cf_input_ids.device)
        cf_input_dict = {"input_ids": cf_input_ids, "attention_mask": cf_attention_mask, "decoder_input_ids": copy.deepcopy(__cf_decoder_input_ids)}

    cfg = model.reader.encoder.config
    cfg.bsz = inputs.input_ids.size(0)
    cfg.n_context = inputs.input_ids.size(1)

    base_output = predict_from_input(
        model, 
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        decoder_input_ids=copy.deepcopy(__decoder_input_ids),
        target_ids=labels,
        max_new_tokens=labels.size(1)
    )

    input_segments = prompt_segmenter([model.reader_tokenizer.convert_ids_to_tokens(input_ids[0])])
    answer = base_output.target_text[0]
    answer_tokens = base_output.target_ids[0]
    answer_tokens = answer_tokens[answer_tokens != -100]

    cf_tokens = [base_output.target_ids[i+1][base_output.target_ids[i+1] != -100] for i in range(len(cf))] if cf and len(cf) > 0 else []
    
    cr_predicted_text = base_output.target_text[0]
    cr_predicted = base_output.target[0]

    correct_prediction = True
    if answer is not None and ((cr_predicted_text.strip().lower() != answer.lower() or (cr_predicted_text.strip().lower() not in answer.lower() or answer.lower() not in cr_predicted_text.strip().lower()))):
        correct_prediction = False

    cr_ans_score = base_output.target_j_probs[0]
    cr_cf_score = base_output.target_j_probs[1:] if cf and len(cf) > 0 else base_output.predicted_j_probs[1:]

    attributes_loc = {}
    for attribute, bounds_mode in attributes.items():
        ranges = find_token_ranges(model.reader_tokenizer, input_ids[0], attribute, bounds=input_segments[0][bounds_mode] if bounds_mode else bounds_mode)
        attributes_loc[attribute] = ranges
    
    if len(list(sorted(list(itertools.chain.from_iterable(attributes_loc.values()))))) == 0:
        return AttrDict({"status": False, "attributes_loc": [], "attribute": attribute})
    
    if reversed_attributes:
        context_range = input_segments[0]["context"]
        attributes_loc = {f"reversed-{i+1}": [token_pos] for i, token_pos in enumerate(reversed_token_ranges([context_range[0] + 4, context_range[1] -1], list(sorted(list(itertools.chain.from_iterable(attributes_loc.values()))))))}
        attributes_noise = ({k: reversed_attributes_noise for k in attributes_loc.keys()})

    noise_generated = {k: False for k in attributes_loc.keys()}
    noise_data = {k: None for k in attributes_loc.keys()}
    agg_ans_scores, agg_cf_scores, agg_scores = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()
    decoder_input_ids = copy.deepcopy(__decoder_input_ids)
    for it in range(len(answer_tokens)):
        logits, noise_generated, noise_data, decoder_input_ids, score = trace_with_patch(
            model=model, 
            input_dict={"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}, 
            states_to_patch=[], 
            answer_tokens=answer_tokens[it], 
            attributes_loc=attributes_loc, 
            attributes_noise=attributes_noise,
            trace_layers=None,
            uniform_noise=uniform_noise,
            replace=replace,
            noise_generated=noise_generated,
            noise_data=noise_data
        )
        agg_scores = torch.cat([agg_scores, score.unsqueeze(0)], dim=0)
        p = torch.softmax(logits[1:, -1, :], dim=1)

        cf_scores, ans_scores = [], []
        if cf_tokens and len(cf_tokens) > 0:
            for i, tokens in enumerate(cf_tokens):
                cf_scores.append(p[i, cf_tokens[i][it]])
                ans_scores.append(p[i, answer_tokens[it]])
        else:
            for i in range(input_ids.shape[0] - 1):
                cf_scores.append(torch.max(p[i], dim=-1)[0])
                ans_scores.append(p[i, answer_tokens[it]])

        agg_cf_scores = torch.cat([agg_cf_scores, torch.tensor(cf_scores).to(agg_cf_scores.device).unsqueeze(1)], dim=1)
        agg_ans_scores = torch.cat([agg_ans_scores, torch.tensor(ans_scores).to(agg_ans_scores.device).unsqueeze(1)], dim=1)
    
    crr_predicted = model.reader_tokenizer.batch_decode(decoder_input_ids[:, 1:])
    crr_score = joint_probs(agg_scores)
    crr_cf_score = joint_probs(agg_cf_scores)
    crr_ans_score = joint_probs(agg_ans_scores)
    crr_scores = agg_scores
    crr_cf_scores = agg_cf_scores
    crr_ans_scores = agg_ans_scores

    if not kind:
        noise_generated = {k: False for k in attributes_loc.keys()}
        noise_data = {k: None for k in attributes_loc.keys()}
        decoder_input_ids = copy.deepcopy(__decoder_input_ids)
        outputs = trace_important_states(
            model=model, 
            num_layers=num_layers, 
            input_dict={"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}, 
            answer_tokens=answer_tokens, 
            attributes_loc=attributes_loc, 
            attributes_noise=attributes_noise,
            cf_tokens=cf_tokens,
            cf_input_dict=cf_input_dict,
            uniform_noise=uniform_noise,
            replace=replace,
            module=module,
            component=component, 
            desc=f"trace_important_states: {answer} - kind: None"
        )
    else:
        noise_generated = {k: False for k in attributes_loc.keys()}
        noise_data = {k: None for k in attributes_loc.keys()}
        decoder_input_ids = copy.deepcopy(__decoder_input_ids)
        outputs = trace_important_window(
            model=model, 
            num_layers=num_layers,
            input_dict={"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}, 
            answer_tokens=answer_tokens, 
            attributes_loc=attributes_loc, 
            attributes_noise=attributes_noise,
            cf_tokens=cf_tokens,
            cf_input_dict=cf_input_dict,
            window=window,
            uniform_noise=uniform_noise,
            replace=replace,
            module=module, 
            kind=kind,
            kind_pos=kind_pos,
            component=component,
            desc=f"trace_important_window: {answer} - kind: {kind}"
        )
    
    crwrr_score = torch.stack([torch.stack([r["score"] for r in output]) for output in outputs])
    crwrr_cf_score = torch.stack([torch.stack([r["cf_score"] for r in output]) for output in outputs])
    crwrr_ans_score = torch.stack([torch.stack([r["ans_score"] for r in output]) for output in outputs])
    crwrr_scores = torch.stack([torch.stack([r["scores"] for r in output]) for output in outputs])
    crwrr_cf_scores = torch.stack([torch.stack([r["cf_scores"] for r in output]) for output in outputs])
    crwrr_ans_scores = torch.stack([torch.stack([r["ans_scores"] for r in output]) for output in outputs])
    crwrr_predicted = [[model.reader_tokenizer.batch_decode(r["decoder_input_ids"]) for r in output] for output in outputs] 

    return AttrDict({
        "status": True,
        "input_ids": input_ids[0],
        "input_tokens": model.reader_tokenizer.convert_ids_to_tokens(input_ids[0]),
        "attributes": attributes,
        "attributes_loc": attributes_loc,
        "has_attribute": True,
        "answer": answer,
        "answer_tokens": model.reader_tokenizer.convert_ids_to_tokens(answer_tokens),
        "window": window,
        "component": component or None,
        "module": module or None,
        "kind": kind or None,
        "correct_prediction": correct_prediction,
        "cr_predicted": cr_predicted,
        "crr_predicted": crr_predicted,
        "crwrr_predicted": crwrr_predicted,

        "scores": crwrr_score.detach().cpu(),

        "cf": cf,

        "cr_ans_score": cr_ans_score.detach().cpu(),
        "cr_cf_score": cr_cf_score.detach().cpu(),

        "crr_score": crr_score.detach().cpu(),
        "crr_cf_score": crr_cf_score.detach().cpu(),
        "crr_ans_score": crr_ans_score.detach().cpu(),
        "crr_scores": crr_scores.detach().cpu(),
        "crr_cf_scores": crr_cf_scores.detach().cpu(),
        "crr_ans_scores": crr_ans_scores.detach().cpu(),

        "crwrr_score": crwrr_score.detach().cpu(),
        "crwrr_cf_score": crwrr_cf_score.detach().cpu(),
        "crwrr_ans_score": crwrr_ans_score.detach().cpu(),
        "crwrr_scores": crwrr_scores.detach().cpu(),
        "crwrr_cf_scores": crwrr_cf_scores.detach().cpu(),
        "crwrr_ans_scores": crwrr_ans_scores.detach().cpu(),
    })


def plot_trace_heatmap_subplot(result, subplot_index, num_subplots, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["crr_score"]
    answer = result["answer"].replace("<pad>", "").replace("<extra_id_0>", "").replace("</s>", "").strip()
    kind = result["kind"]
    window = result.get("window", 10)
    labels = list(result["input_tokens"])

    attributes_loc = result["attributes_loc"].values()
    attributes_loc = [item for sublist in attributes_loc for item in sublist]

    if isinstance(attributes_loc, list):
        for e_range in attributes_loc:
            for i in range(*e_range):
                labels[i] = labels[i] + "*"
    else:
        for i in range(*attributes_loc):
            labels[i] = labels[i] + "*"

    # with plt.rc_context(rc={"font.family": "Arial"}):
    plt.rcParams['font.size'] = 6
    plt.subplot(1, num_subplots, subplot_index)  # Create subplot
    h = plt.pcolor(
        differences,
        cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds", "cross_attn": "Oranges"}[inverse_mapping(KIND_MAP_LIST).get(kind)],
        vmin=low_score,
    )
    plt.gca().invert_yaxis()

    plt.yticks([0.5 + i for i in range(len(differences))])
    plt.xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)], labels=range(0, differences.shape[1] - 6, 5))


    plt.gca().set_yticklabels(labels)
    if not modelname:
        modelname = "ATLAS"
    if not kind:
        plt.title("Impact of restoring state after corrupted input")
        plt.xlabel(f"single restored layer within {modelname}")
    else:
        kindname = inverse_mapping(KIND_MAP_LIST).get(kind)
        plt.title(f"Impact of restoring {kindname} after corrupted input")
        plt.xlabel(f"center of interval of {window} restored {kindname} layers")
    cb = plt.colorbar(h)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if answer is not None:
        # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
        cb.ax.set_title(f"p({str(answer).strip()})", y=-0.08, fontsize=6)
        
    if subplot_index == num_subplots:
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def plot_trace_heatmap(
    scores, 
    input_tokens,
    answer,
    attributes_loc,
    low_score=None,
    high_score=None,
    kindname=None,
    window=10,
    savepdf=None, 
    title=None, 
    xlabel=None, 
    modelname=None, 
    fontsize=7, 
    ratio=0.2
):
    if low_score is None:
        low_score = scores.min()
    if high_score is None:
        high_score = scores.max()

    answer = answer.replace("<pad>", "").replace("<extra_id_0>", "").replace("</s>", "").strip()
    window = window
    labels = list(input_tokens)
    attributes_loc = [item for sublist in attributes_loc for item in sublist]
    
    if isinstance(attributes_loc, list):
        for e_range in attributes_loc:
            for i in range(*e_range):
                labels[i] = labels[i] + "*"
    else:
        for i in range(*attributes_loc):
            labels[i] = labels[i] + "*"

    plt.rcParams['font.size'] = fontsize
    fig, ax = plt.subplots(figsize=(5, len(labels) * ratio), dpi=300)
    
    ax.set_aspect('auto')

    h = ax.pcolor(
        scores,
        cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds", "cross_attn": "Oranges"}[kindname],
        vmin=low_score,
        vmax=high_score,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(scores))])
    ax.set_xticks([0.5 + i for i in range(0, scores.shape[1])])
    ax.set_xticklabels(list(range(0, scores.shape[1])), rotation=0)
    ax.set_yticklabels(labels, rotation=0)
    
    if not modelname:
        modelname = "ATLAS"
    if not kindname:
        ax.set_title("Impact of restoring state")
        ax.set_xlabel(f"single restored layer within {modelname}")
    else:
        ax.set_title(f"Impact of restoring {kindname}")
        ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
    cb = plt.colorbar(h)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif answer is not None:
        # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
        cb.ax.set_ylabel(f"IE", fontsize=10)
        # cb.ax.set_title(f"IE", y=-0.1, fontsize=10)
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def trace_with_repatch(
    model,  # The model
    input_dict,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answer_tokens,  # Answer probabilities to collect
    attributes_loc,  # a ranges of tokens to corrupt (begin, end)
    attributes_noise,
    trace_layers=None,  # List of traced outputs to return
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    noise_generated=False,
    noise_data=None,
    cf_patch=None
):

    # For reproducibility, use pseudorandom noise
    rs = np.random.RandomState(1)  

    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)


    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)
    

    # print(f"patch_spec: {patch_spec}")
    embed_layername = "reader.encoder.embed_tokens"


    def untuple(x):
        return x[0] if isinstance(x, tuple) else x


    # Define the model-patching rule.
    dim = input_dict["input_ids"].size(-1)
    def patch_rep(x, layer):
        nonlocal noise_generated, noise_data

        if layer == embed_layername and ((len(untuple(x).size()) == 3 and untuple(x).size(1) == dim) or (len(x.size()) == 3 and x.size(1) == dim)):
            
            # Corrupt a range of token embeddings on batch items x[1:] for each specified range
            for attribute, locations in attributes_loc.items():                
                if len(locations) == 0:
                    continue

                b, e = locations[0]
                if not noise_generated[attribute]:
                    if not isinstance(attributes_noise[attribute], float):
                        noise = torch.from_numpy(np.array(attributes_noise[attribute])).to(x.device)
                        noise = repeat_to_size(noise, x.size(0) - 1)
                        noise_data[attribute] = noise
                    else:
                        noise_data[attribute] = (
                            attributes_noise[attribute] * torch.from_numpy(
                                prng(x.shape[0] - 1, e - b, x.shape[2])
                            )
                        ).to(x.device)
                    
                    noise_generated[attribute] = True
                
                for location in locations:
                    b, e = location

                    # print(f"b,e: ({b}, {e}), x[1:, b:e]: {x[1:, b:e].shape}, noise_data[attribute]: {noise_data[attribute].shape}")
                    if replace:
                        x[1:, b:e] = noise_data[attribute]
                    else:
                        x[1:, b:e] += noise_data[attribute]

            return x

        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x

        # Restore the uncorrupted hidden state for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            if cf_patch and layer in cf_patch:
                h[1:, t] = cf_patch[layer][1:, t]
            else:
                h[1:, t] = h[0, t]

        
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
            
        return x

    # Run the patched model in inference with the patching rules defined.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
                
            input_ids = input_dict["input_ids"]
            attention_mask = input_dict["attention_mask"]
            decoder_input_ids = input_dict["decoder_input_ids"]

            outputs = model.reader(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

            next_token = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            if first_pass:
                first_pass_trace = td
    
    probs = torch.softmax(logits[1:, -1, :], dim=-1).mean(dim=0)[answer_tokens]

    return logits, noise_generated, noise_data, decoder_input_ids, probs


def calculate_hidden_flow_when_served(
    model, 
    prompt, 
    attributes,
    attributes_noise,
    reversed_attributes=False,
    reversed_attributes_noise=0.0,
    counterfactuals=None, 
    samples=10, 
    window=10, 
    module="block",
    component="encoder",
    n_context=1,
    uniform_noise=False,
    replace=False,
    num_layers=12,
    patch_type="clean",  # "clean", "counterfactual"
    token_range=None,
    disable_mlp=False,
    disable_attn=False,
):
    layer_info = {kind: [KIND_MAP_LIST.get(kind), 0 if kind == "attn" else 2 if (kind == "mlp" and component == "decoder") else 1] for kind in ["attn", "mlp"]}

    cf = counterfactuals if counterfactuals and isinstance(counterfactuals, list) else []

    prompts = [prompt] * (samples + 1)
    for p in prompts:
        p["cf_passages"] = p["passages"]
        
    if cf and isinstance(cf, list):
        cf = adjust_cf_samples(cf, samples)
        cf_prompts = []
        for item in cf:
            cf_prompt = copy.deepcopy(prompt)
            cf_prompt["cf_passages"] = [{"title": p["title"], "text": p["text"].replace(cf_prompt["answers"][0], item)} for p in cf_prompt["passages"]]
            cf_prompt["answers"] = [item]
            cf_prompts.append(cf_prompt)
        prompts = [prompt] + cf_prompts
    else:
        prompts = [prompt] * (samples + 1)
    
    
    inputs = make_inputs(model, prompts, prompt_is_dict=True, n_context=n_context)
    input_ids = inputs.input_ids.cuda().view(inputs.input_ids.size(0), -1)
    attention_mask = inputs.attention_mask.cuda().view(inputs.attention_mask.size(0), -1)
    labels = inputs.labels.cuda()
    __decoder_input_ids = torch.tensor([model.reader.config.decoder_start_token_id]).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)

    cf_input_dict = {}
    if patch_type == "counterfactual":
        cf_inputs = make_inputs(model, prompts, prompt_is_dict=True, n_context=n_context, passages_key="cf_passages")
        cf_input_ids = cf_inputs.input_ids.cuda().view(cf_inputs.input_ids.size(0), -1)
        cf_attention_mask = cf_inputs.attention_mask.cuda().view(cf_inputs.attention_mask.size(0), -1)
        cf_labels = cf_inputs.labels.cuda()
        __cf_decoder_input_ids = torch.tensor([model.reader.config.decoder_start_token_id]).unsqueeze(0).repeat(cf_input_ids.size(0), 1).to(cf_input_ids.device)
        cf_input_dict = {"input_ids": cf_input_ids, "attention_mask": cf_attention_mask, "decoder_input_ids": copy.deepcopy(__cf_decoder_input_ids)}

    cfg = model.reader.encoder.config
    cfg.bsz = inputs.input_ids.size(0)
    cfg.n_context = inputs.input_ids.size(1)

    base_output = predict_from_input(
        model, 
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        decoder_input_ids=copy.deepcopy(__decoder_input_ids),
        target_ids=labels,
        max_new_tokens=labels.size(1)
    )

    input_segments = prompt_segmenter([model.reader_tokenizer.convert_ids_to_tokens(input_ids[0])])
    answer = base_output.target_text[0]
    answer_tokens = base_output.target_ids[0]
    answer_tokens = answer_tokens[answer_tokens != -100]

    cf_tokens = [base_output.target_ids[i+1][base_output.target_ids[i+1] != -100] for i in range(len(cf))] if cf and len(cf) > 0 else []
    
    cr_predicted_text = base_output.target_text[0]
    cr_predicted = base_output.target[0]

    correct_prediction = True
    if answer is not None and ((cr_predicted_text.strip().lower() != answer.lower() or (cr_predicted_text.strip().lower() not in answer.lower() or answer.lower() not in cr_predicted_text.strip().lower()))):
        correct_prediction = False

    cr_ans_score = base_output.target_j_probs[0]
    cr_cf_score = base_output.target_j_probs[1:] if cf and len(cf) > 0 else base_output.predicted_j_probs[1:]

    attributes_loc = {}
    for attribute, bounds_mode in attributes.items():
        ranges = find_token_ranges(model.reader_tokenizer, input_ids[0], attribute, bounds=input_segments[0][bounds_mode] if bounds_mode else bounds_mode)
        # print(f"attribute: {attribute}, bounds_mode: {bounds_mode}, ranges: {ranges}")
        attributes_loc[attribute] = ranges

    if len(list(sorted(list(itertools.chain.from_iterable(attributes_loc.values()))))) == 0:
        return AttrDict({"status": False, "attributes_loc": [], "attribute": attribute})

    if reversed_attributes:
        context_range = input_segments[0]["context"]
        attributes_loc = {f"reversed-{i+1}": [token_pos] for i, token_pos in enumerate(reversed_token_ranges([context_range[0] + 4, context_range[1] -1], list(sorted(list(itertools.chain.from_iterable(attributes_loc.values()))))))}
        attributes_noise = ({k: reversed_attributes_noise for k in attributes_loc.keys()})

    noise_generated = {k: False for k in attributes_loc.keys()}
    noise_data = {k: None for k in attributes_loc.keys()}
    agg_ans_scores, agg_cf_scores, agg_scores = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()
    decoder_input_ids = copy.deepcopy(__decoder_input_ids)


    for it in range(len(answer_tokens)):
        logits, noise_generated, noise_data, decoder_input_ids, score = trace_with_patch(
            model=model, 
            input_dict={"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}, 
            states_to_patch=[], 
            answer_tokens=answer_tokens[it], 
            attributes_loc=attributes_loc, 
            attributes_noise=attributes_noise,
            trace_layers=None,
            uniform_noise=uniform_noise,
            replace=replace,
            noise_generated=noise_generated,
            noise_data=noise_data
        )
    
        agg_scores = torch.cat([agg_scores, score.unsqueeze(0)], dim=0)
        p = torch.softmax(logits[1:, -1, :], dim=1)

        cf_scores, ans_scores = [], []
        if cf_tokens and len(cf_tokens) > 0:
            for i, tokens in enumerate(cf_tokens):
                cf_scores.append(p[i, cf_tokens[i][it]])
                ans_scores.append(p[i, answer_tokens[it]])
        else:
            for i in range(input_ids.shape[0] - 1):
                cf_scores.append(torch.max(p[i], dim=-1)[0])
                ans_scores.append(p[i, answer_tokens[it]])

        agg_cf_scores = torch.cat([agg_cf_scores, torch.tensor(cf_scores).to(agg_cf_scores.device).unsqueeze(1)], dim=1)
        agg_ans_scores = torch.cat([agg_ans_scores, torch.tensor(ans_scores).to(agg_ans_scores.device).unsqueeze(1)], dim=1)
    
    crr_predicted = model.reader_tokenizer.batch_decode(decoder_input_ids[:, 1:])
    crr_score = joint_probs(agg_scores)
    crr_cf_score = joint_probs(agg_cf_scores)
    crr_ans_score = joint_probs(agg_ans_scores)
    crr_scores = agg_scores
    crr_cf_scores = agg_cf_scores
    crr_ans_scores = agg_ans_scores

    
    noise_generated = {k: False for k in attributes_loc.keys()}
    noise_data = {k: None for k in attributes_loc.keys()}
    decoder_input_ids = copy.deepcopy(__decoder_input_ids)
    outputs = trace_important_states_when_served(
        model=model, 
        num_layers=num_layers, 
        input_dict={"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}, 
        answer_tokens=answer_tokens, 
        attributes_loc=attributes_loc, 
        attributes_noise=attributes_noise,
        cf_tokens=cf_tokens,
        cf_input_dict=cf_input_dict,
        uniform_noise=uniform_noise,
        replace=replace,
        module=module,
        component=component, 
        desc=f"trace_important_states_when_served: {answer} - kind: None",
        layer_info=layer_info,
        token_range=token_range,
        disable_mlp=disable_mlp,
        disable_attn=disable_attn,
    )
    
    crwrr_score = torch.stack([torch.stack([r["score"] for r in output]) for output in outputs])
    crwrr_cf_score = torch.stack([torch.stack([r["cf_score"] for r in output]) for output in outputs])
    crwrr_ans_score = torch.stack([torch.stack([r["ans_score"] for r in output]) for output in outputs])
    crwrr_scores = torch.stack([torch.stack([r["scores"] for r in output]) for output in outputs])
    crwrr_cf_scores = torch.stack([torch.stack([r["cf_scores"] for r in output]) for output in outputs])
    crwrr_ans_scores = torch.stack([torch.stack([r["ans_scores"] for r in output]) for output in outputs])
    crwrr_predicted = [[model.reader_tokenizer.batch_decode(r["decoder_input_ids"]) for r in output] for output in outputs] 

    return AttrDict({
        "status": True,
        "input_ids": input_ids[0],
        "input_tokens": model.reader_tokenizer.convert_ids_to_tokens(input_ids[0]),
        "attributes": attributes,
        "attributes_loc": attributes_loc,
        "has_attribute": True,
        "answer": answer,
        "answer_tokens": model.reader_tokenizer.convert_ids_to_tokens(answer_tokens),
        "window": window,
        "component": component or None,
        "module": module or None,
        "kind": None,
        "correct_prediction": correct_prediction,
        "cr_predicted": cr_predicted,
        "crr_predicted": crr_predicted,
        "crwrr_predicted": crwrr_predicted,

        "scores": crwrr_score.detach().cpu(),

        "cf": cf,

        "cr_ans_score": cr_ans_score.detach().cpu(),
        "cr_cf_score": cr_cf_score.detach().cpu(),

        "crr_score": crr_score.detach().cpu(),
        "crr_cf_score": crr_cf_score.detach().cpu(),
        "crr_ans_score": crr_ans_score.detach().cpu(),
        "crr_scores": crr_scores.detach().cpu(),
        "crr_cf_scores": crr_cf_scores.detach().cpu(),
        "crr_ans_scores": crr_ans_scores.detach().cpu(),

        "crwrr_score": crwrr_score.detach().cpu(),
        "crwrr_cf_score": crwrr_cf_score.detach().cpu(),
        "crwrr_ans_score": crwrr_ans_score.detach().cpu(),
        "crwrr_scores": crwrr_scores.detach().cpu(),
        "crwrr_cf_scores": crwrr_cf_scores.detach().cpu(),
        "crwrr_ans_scores": crwrr_ans_scores.detach().cpu(),
    })

def trace_important_states_when_served(
    model, 
    num_layers, 
    input_dict, 
    answer_tokens,  
    attributes_loc, 
    attributes_noise,
    cf_tokens=None,
    cf_input_dict=None,
    module="block", 
    component="encoder", 
    uniform_noise=False, 
    replace=False,
    desc="trace_important_states_when_served",
    layer_info=None,
    token_range=None,
    disable_mlp=False,
    disable_attn=False,
):
    cf_tokens = cf_tokens if cf_tokens and isinstance(cf_tokens, list) else []
    ntoks = input_dict["input_ids"].shape[-1]
    table = []

    if token_range is None:
        token_range = range(ntoks)

    # print(f"ntoks: {ntoks}")
    for tnum in tqdm(token_range, total=len(token_range),  desc=desc):
        zero_mlps = []
        if disable_mlp:
            for layer in range(num_layers):
                for it in range(len(answer_tokens)):
                    zero_mlps += [(tnum if component == "encoder" else it, f"reader.{component}.{module}.{layer}.layer.{layer_info['mlp'][1]}.{layer_info['mlp'][0]}")]
        
        if disable_attn:
            for layer in range(num_layers):
                for it in range(len(answer_tokens)):
                    zero_mlps += [(tnum if component == "encoder" else it, f"reader.{component}.{module}.{layer}.layer.{layer_info['attn'][1]}.{layer_info['attn'][0]}")]
        
            
        row = []
        for layer in range(num_layers):

            decoder_input_ids = input_dict["decoder_input_ids"]
            noise_generated = {k: False for k in attributes_loc.keys()}
            noise_data = {k: None for k in attributes_loc.keys()}
            agg_ans_scores, agg_cf_scores, agg_scores = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()


            for it in range(len(answer_tokens)):
                layerlist = [(tnum if component == "encoder" else it, f"reader.{component}.{module}.{layer}")]
                cf_patch = None

                if cf_input_dict and isinstance(cf_input_dict, dict):
                    cf_patch = cf_patch_extract(model, layerlist, cf_input_dict)

                
                logits, noise_generated, noise_data, decoder_input_ids, score = trace_with_repatch(
                    model=model, 
                    input_dict={
                        "input_ids": input_dict["input_ids"], 
                        "attention_mask": input_dict["attention_mask"], 
                        "decoder_input_ids": decoder_input_ids,
                    }, 
                    states_to_patch=layerlist, 
                    states_to_unpatch=zero_mlps,
                    answer_tokens=answer_tokens[it], 
                    attributes_loc=attributes_loc if not isinstance(cf_patch, dict) or not cf_patch else {}, 
                    attributes_noise=attributes_noise if not isinstance(cf_patch, dict) or not cf_patch else None,
                    trace_layers=None,
                    uniform_noise=uniform_noise,
                    replace=replace,
                    noise_generated=noise_generated,
                    noise_data=noise_data,
                    cf_patch=cf_patch
                )

                # raise Exception("CHECKPOINT!")

                agg_scores = torch.cat([agg_scores, score.unsqueeze(0)], dim=0)
                p = torch.softmax(logits[1:, -1, :], dim=1)

                cf_scores, ans_scores = [], []
                if cf_tokens and len(cf_tokens) > 0:
                    for i, tokens in enumerate(cf_tokens):
                        cf_scores.append(p[i, cf_tokens[i][it]])
                        ans_scores.append(p[i, answer_tokens[it]])
                else:
                    for i in range(input_dict["input_ids"].shape[0] - 1):
                        cf_scores.append(torch.max(p[i], dim=-1)[0])
                        ans_scores.append(p[i, answer_tokens[it]])

                agg_cf_scores = torch.cat([agg_cf_scores, torch.tensor(cf_scores).to(agg_cf_scores.device).unsqueeze(1)], dim=1)
                agg_ans_scores = torch.cat([agg_ans_scores, torch.tensor(ans_scores).to(agg_ans_scores.device).unsqueeze(1)], dim=1)
            
            row.append({
                "score": joint_probs(agg_scores),
                "cf_score": joint_probs(agg_cf_scores),
                "ans_score": joint_probs(agg_ans_scores),
                "scores": agg_scores,
                "cf_scores": agg_cf_scores,
                "ans_scores": agg_ans_scores,
                "decoder_input_ids": decoder_input_ids[:, 1:]
            })
        table.append(row)
        
    return table


def plot_served_comparison(ordinary, no_attn, no_mlp, title, token_idx=-1, bar_width=0.2, savepdf=None):
    fig, ax = plt.subplots(1, figsize=(4, 1.5), dpi=300)
    ax.bar(
        [i - bar_width for i in range(len(ordinary[token_idx]))],
        ordinary[token_idx],
        width=bar_width,
        color="#7261ab",
        label="Impact of single state on P",
    )
    ax.bar(
        [i for i in range(len(no_attn[token_idx]))],
        no_attn[token_idx],
        width=bar_width,
        color="#f3201b",
        label="Impact with Attn severed",
    )
    ax.bar(
        [i + bar_width for i in range(len(no_mlp[token_idx]))],
        no_mlp[token_idx],
        width=bar_width,
        color="#20b020",
        label="Impact with MLP severed",
    )
    ax.set_title(
        title
    )
    ax.set_xticks([i for i in range(0, ordinary.shape[1])])
    ax.set_xticklabels(list(range(0, ordinary.shape[1])), rotation=0)
    ax.set_ylabel("Indirect Effect")

    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax.set_ylim(None, max(0.025, ordinary[token_idx].max() * 1.05))

    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.set_ylim(None, max(0.025, ordinary.max() * 1.15))
    # ax.legend(loc='best')
    ax.legend(loc='best')

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()