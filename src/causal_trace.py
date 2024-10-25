import argparse
import copy
import itertools
import json
import os
import pprint
import random
import re
import sys

import numpy as np
import torch
from tqdm import tqdm

from experiments import nethook
from experiments.dataset import KnownsDataset
from experiments.tools import calculate_hidden_flow, calculate_hidden_flow_when_served, collect_embedding_std, plot_trace_heatmap
from experiments.utils import load_atlas


def parse_attributes(attr_str):
    attrs = {}
    if attr_str:
        for pair in attr_str.split(','):
            key, value = pair.split('=')
            attrs[key] = None if value.lower() == 'none' else value
    return attrs


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def parse_attributes_noise(noise_str, model, knowns):
    # Example input format: "subj=4.0,obj=/path/to/tensor.pth"
    noises = {}
    if noise_str:
        for pair in noise_str.split(','):
            key, value = pair.split('=')

            if value.startswith("ss"):
                factor = float(value[1:]) if len(value) > 1 else 1.0
                value = factor * collect_embedding_std(model, knowns, attribute=key)
            else:
                value = float(value) if isfloat(value) else str(value)

            noises[key] = value
    return noises


def safe_division(a, b, do_log=False):
    return a/b if not do_log else (torch.log(a) - torch.log(b))


def calculate_te_ie(r, samples, patch_type="clean", do_log=True):
    cr_cf_score = r["cr_cf_score"]
    cr_ans_score = r["cr_ans_score"]

    crr_cf_score = r["crr_cf_score"]
    crr_ans_score = r["crr_ans_score"]

    crwrr_cf_score = r["crwrr_cf_score"]
    crwrr_ans_score = r["crwrr_ans_score"]

    te, ie = [], []

    for i in range(samples):
        if patch_type == "counterfactual":
            te_i = safe_division(crr_cf_score[i], crr_ans_score[i], do_log=do_log) - \
                   safe_division(cr_cf_score[i], cr_ans_score, do_log=do_log)
            ie_i = safe_division(crwrr_cf_score[:, :, i], crwrr_ans_score[:, :, i], do_log=do_log) - \
                   safe_division(cr_cf_score[i], cr_ans_score, do_log=do_log)
        else:
            te_i = safe_division(cr_cf_score[i], cr_ans_score, do_log=do_log) - \
                   safe_division(crr_cf_score[i], crr_ans_score[i], do_log=do_log)
            ie_i = safe_division(crwrr_cf_score[:, :, i], crwrr_ans_score[:, :, i], do_log=do_log) - \
                   safe_division(crr_cf_score[i], crr_ans_score[i], do_log=do_log)

        te.append(te_i)
        ie.append(ie_i.unsqueeze(-1))

    te = torch.stack(te).mean()
    ie = torch.cat(ie, axis=-1).mean(-1)

    return te, ie


def calculate_post_proc(r, prompt, samples, patch_type="clean"):
    r.prompt = {
        "subj": prompt["subj"],
        "obj": prompt["obj"],
        "prop": prompt["prop"],
        "subj_cf": prompt["subj_cf"],
        "obj_cf": prompt["obj_cf"],
        "subj_cf_diff": prompt["subj_cf_diff"],
        "obj_cf_diff": prompt["obj_cf_diff"],
    }
    
    if r["status"]:
        te_log, ie_log = calculate_te_ie(r, samples=samples, patch_type=patch_type, do_log=True)
        r.te_log = te_log
        r.ie_log = ie_log

        te, ie = calculate_te_ie(r, samples=samples, patch_type=patch_type, do_log=False)
        r.te = te
        r.ie = ie
    else:
        r.te_log = None
        r.ie_log = None
        r.te = None
        r.ie = None

    r = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in r.items()}

    return r


def sample_from_list(lst, num_samples):
    if num_samples > len(lst):
        return lst 
    # return random.sample(lst, num_samples)
    return lst[:num_samples]



def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    parser.add_argument("--model_name",
        default="atlas",
        choices=[
            "atlas",
            "gpt2-xl",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
        ]
    )
    parser.add_argument(
        "--data_path",
        help="A space-separated path to jsonl-formatted evaluation sets",
    )
    parser.add_argument("--output_dir", default="experiments/x/{model_name}/causal_trace")
    parser.add_argument("--attributes", type=str, help="Attributes in format 'key1=value1,key2=value2'")
    parser.add_argument("--attributes_noise", type=str, help="Attributes noise in format 'key1=value1,key2=value2'")
    parser.add_argument("--patch_type", type=str, help="Type of patch with respect to experiment")
    parser.add_argument("--replace", type=int, choices=[0, 1], default=0, help="replace parameter as bool")
    parser.add_argument("--reversed_attributes", type=int, choices=[0, 1], default=0, help="reversed_attributes parameter as bool")
    parser.add_argument("--reversed_attributes_noise", type=float, default=1.0, help="reversed_attributes_noise parameter as float")
    # parser.add_argument("--do_log", type=int, choices=[0, 1], default=0, help="do_log parameter as bool")

    
    parser.add_argument(
        "--reader_model_type",
        required=True,
        type=str,
        help="t5 Architecture for reader FID model, e.g. google/t5-xl-lm-adapt",
        choices=[
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-3b",
            "t5-11b",
            "google/t5-v1_1-base",
            "google/t5-v1_1-large",
            "google/t5-v1_1-xl",
            "google/t5-v1_1-xxl",
            "google/t5-base-lm-adapt",
            "google/t5-large-lm-adapt",
            "google/t5-xl-lm-adapt",
            "google/t5-xxl-lm-adapt",
        ],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="none",
        help="Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever)",
    )
    parser.add_argument(
        "--n_context",
        type=int,
        default=1,
        help="number of top k passages to pass to reader",
    )
    parser.add_argument(
        "--qa_prompt_format",
        type=str,
        default="question: {question} answer: <extra_id_0>",
        help="How to format question as input prompts when using --task qa",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="num layers",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="num counterfactual samples",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="num window for restoration section",
    )
    parser.add_argument(
        "--cf",
        type=int,
        default=3,
        help="num counterfactual",
    )
    parser.add_argument(
        "--max_knowns",
        type=int,
        default=0,
        help="maximum number of recoreds",
    )
    parser.add_argument("--disable_mlp_attn", type=int, choices=[0, 1], default=0, help="disable mlp as bool")
    

    args = parser.parse_args()
    
    print("ARGS:", '---' * 10)
    pprint.pp(args.__dict__)
    print("/", '---' * 10, "\n")

    output_dir = args.output_dir
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)


    model, opt = load_atlas(reader_model_type=args.reader_model_type, model_path=args.model_path, n_context=args.n_context, qa_prompt_format=args.qa_prompt_format)
    nethook.set_requires_grad(False, model)
    knowns = KnownsDataset(args.data_path, jsonl=True if ".jsonl" in args.data_path else False)

    print("ONE SAMPLE:", '---' * 10)
    pprint.pp(knowns[0])
    print("/", '---' * 10, "\n")

    attributes = parse_attributes(args.attributes)
    attributes_noise = parse_attributes_noise(args.attributes_noise, model, knowns)

    print(f"attributes: {attributes}")
    print(f"attributes_noise: {attributes_noise}")

    for known_id, knowledge in tqdm(enumerate(knowns)):
        if args.max_knowns > 0 and known_id > args.max_knowns:
            print(f"More than specified `{args.max_knowns}` knowns!")
            break
        
        objs = sample_from_list(knowledge["obj_cf"], args.cf)

        for kind in None, "mlp", "attn":
            kind_suffix = f"_{kind}" if kind else ""

            if args.patch_type == "counterfactual":
                prompt = copy.deepcopy(knowledge)
                prompt["passages"] = [{"title": " ", "text": p["text"]} for p in prompt["passages"]]
                
                filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.bin"
            
                _attributes = {prompt[k]: v for k, v in attributes.items()}
                _attributes_noise = {prompt[k]: v if isinstance(v, float) else prompt[v] for k, v in attributes_noise.items()}

                if args.disable_mlp_attn:
                    if kind != None:
                        continue
                    
                    if not os.path.isfile(filename):
                        ordinary_r = calculate_post_proc(
                            calculate_hidden_flow_when_served(
                                model, prompt, 
                                samples=args.samples, 
                                window=args.window, 
                                reversed_attributes=bool(args.reversed_attributes),
                                reversed_attributes_noise=float(args.reversed_attributes_noise),
                                attributes=_attributes, 
                                attributes_noise=_attributes_noise,
                                counterfactuals=prompt["obj_cf"],
                                patch_type=args.patch_type,
                                n_context=args.n_context,
                                num_layers=args.num_layers, 
                                replace=bool(args.replace),
                            ), 
                            prompt=prompt, 
                            samples=args.samples, 
                            patch_type=args.patch_type, 
                        )
                        no_attn_r = calculate_post_proc(
                            calculate_hidden_flow_when_served(
                                model, prompt, 
                                samples=args.samples, 
                                window=args.window, 
                                reversed_attributes=bool(args.reversed_attributes),
                                reversed_attributes_noise=float(args.reversed_attributes_noise),
                                attributes=_attributes, 
                                attributes_noise=_attributes_noise,
                                counterfactuals=prompt["obj_cf"],
                                patch_type=args.patch_type,
                                n_context=args.n_context,
                                num_layers=args.num_layers, 
                                replace=bool(args.replace),
                                disable_attn=True,
                            ), 
                            prompt=prompt, 
                            samples=args.samples, 
                            patch_type=args.patch_type, 
                        )
                        no_mlp_r = calculate_post_proc(
                            calculate_hidden_flow_when_served(
                                model, prompt, 
                                samples=args.samples, 
                                window=args.window, 
                                reversed_attributes=bool(args.reversed_attributes),
                                reversed_attributes_noise=float(args.reversed_attributes_noise),
                                attributes=_attributes, 
                                attributes_noise=_attributes_noise,
                                counterfactuals=prompt["obj_cf"],
                                patch_type=args.patch_type,
                                n_context=args.n_context,
                                num_layers=args.num_layers, 
                                replace=bool(args.replace),
                                disable_mlp=True,
                            ), 
                            prompt=prompt, 
                            samples=args.samples, 
                            patch_type=args.patch_type, 
                        )
                        r = {
                            "ordinary_r": ordinary_r,
                            "no_attn_r": no_attn_r,
                            "no_mlp_r": no_mlp_r,
                        }
                        torch.save(r, filename)
                    else:
                        r = torch.load(filename)

                    if not r["ordinary_r"]["status"] or not r["no_attn_r"]["status"] or not r["no_mlp_r"]["status"]:
                        print(f"Problem started from {known_id}")
                        continue 
                    
                    if known_id > 200:
                        continue

                    for r_type in ["ordinary_r", "no_attn_r", "no_mlp_r"]:
                        pdfname = f'{pdf_dir}/{str(r["ordinary_r"]["answer"]).strip()}_{known_id}{kind_suffix}_{r_type}.pdf'
                        plot_trace_heatmap(
                            scores=np.clip(r[r_type]["ie"], 0, None),
                            input_tokens=r[r_type]["input_tokens"], 
                            answer=r[r_type]["answer"], 
                            attributes_loc=r[r_type]["attributes_loc"].values(), 
                            ratio=0.15,
                            kindname=r[r_type]["kind"],
                            title=f"Impact of restoring {r[r_type]['kind'] if r[r_type]['kind'] else 'state'} - {'Experiment A`' if args.patch_type == 'counterfactual' else 'Experiment B`'}",
                            savepdf=pdfname
                        )

                else:
                    if not os.path.isfile(filename):
                        r = calculate_post_proc(
                            calculate_hidden_flow(
                                model, prompt, 
                                samples=args.samples, 
                                window=args.window, 
                                reversed_attributes=bool(args.reversed_attributes),
                                reversed_attributes_noise=float(args.reversed_attributes_noise),
                                attributes=_attributes, 
                                attributes_noise=_attributes_noise,
                                counterfactuals=prompt["obj_cf"],
                                patch_type=args.patch_type,
                                kind=kind,
                                n_context=args.n_context,
                                num_layers=args.num_layers, 
                                replace=bool(args.replace),
                            ), 
                            prompt=prompt, 
                            samples=args.samples, 
                            patch_type=args.patch_type, 
                        )
                        torch.save(r, filename)
                    else:
                        r = torch.load(filename)

                    if not r["status"]:
                        print(f"Problem started from {known_id}")
                        continue 
                    
                    print(f"knowledge_{known_id}{kind_suffix}: TE: {r['te']}, Log(TE): {r['te_log']}")

                    pdfname = f'{pdf_dir}/{str(r["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
                    if known_id > 200:
                        continue
                    plot_trace_heatmap(
                        scores=np.clip(r["ie"], 0, None),
                        input_tokens=r["input_tokens"], 
                        answer=r["answer"], 
                        attributes_loc=r["attributes_loc"].values(), 
                        ratio=0.15,
                        kindname=kind,
                        title=f"Impact of restoring {kind if kind else 'state'} - {'Experiment A' if args.patch_type == 'counterfactual' else 'Experiment B'}",
                        savepdf=pdfname
                    )

            else:
                for obj in objs:
                    obj_safe = obj.replace(" ", "-").replace("_", "-")
                    filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}_{obj_safe}.bin"

                    prompt = copy.deepcopy(knowledge)
                    # prompt["passages"] = [{"title": " ", "text": p["text"].replace(prompt["answers"][0], obj)} for p in prompt["passages"]]
                    prompt["passages"] = [{"title": " ", "text": re.sub(fr'\b{re.escape(prompt["answers"][0])}\b', obj, p["text"])} for p in prompt["passages"]]

                    _attributes = {prompt[k] if k != "obj" else obj: v for k, v in attributes.items()}
                    _attributes_noise = {prompt[k]: v if isinstance(v, float) else prompt[v] for k, v in attributes_noise.items()}
                    if args.disable_mlp_attn:
                        if kind != None:
                            continue

                        if not os.path.isfile(filename):
                            ordinary_r = calculate_post_proc(
                                calculate_hidden_flow_when_served(
                                    model, prompt, 
                                    samples=args.samples, 
                                    window=args.window, 
                                    reversed_attributes=bool(args.reversed_attributes),
                                    reversed_attributes_noise=float(args.reversed_attributes_noise),
                                    attributes=_attributes, 
                                    attributes_noise=_attributes_noise,
                                    counterfactuals=[obj],
                                    patch_type=args.patch_type,
                                    n_context=args.n_context,
                                    num_layers=args.num_layers, 
                                    replace=bool(args.replace),
                                ), 
                                prompt=prompt, 
                                samples=args.samples, 
                                patch_type=args.patch_type, 
                            )
                            no_attn_r = calculate_post_proc(
                                calculate_hidden_flow_when_served(
                                    model, prompt, 
                                    samples=args.samples, 
                                    window=args.window, 
                                    reversed_attributes=bool(args.reversed_attributes),
                                    reversed_attributes_noise=float(args.reversed_attributes_noise),
                                    attributes=_attributes, 
                                    attributes_noise=_attributes_noise,
                                    counterfactuals=[obj],
                                    patch_type=args.patch_type,
                                    n_context=args.n_context,
                                    num_layers=args.num_layers, 
                                    replace=bool(args.replace),
                                    disable_attn=True,
                                ), 
                                prompt=prompt, 
                                samples=args.samples, 
                                patch_type=args.patch_type, 
                            )
                            no_mlp_r = calculate_post_proc(
                                calculate_hidden_flow_when_served(
                                    model, prompt, 
                                    samples=args.samples, 
                                    window=args.window, 
                                    reversed_attributes=bool(args.reversed_attributes),
                                    reversed_attributes_noise=float(args.reversed_attributes_noise),
                                    attributes=_attributes, 
                                    attributes_noise=_attributes_noise,
                                    counterfactuals=[obj],
                                    patch_type=args.patch_type,
                                    n_context=args.n_context,
                                    num_layers=args.num_layers, 
                                    replace=bool(args.replace),
                                    disable_mlp=True,
                                ), 
                                prompt=prompt, 
                                samples=args.samples, 
                                patch_type=args.patch_type, 
                            )
                            r = {
                                "ordinary_r": ordinary_r,
                                "no_attn_r": no_attn_r,
                                "no_mlp_r": no_mlp_r,
                            }
                            torch.save(r, filename)
                        else:
                            r = torch.load(filename)
                        

                        if not r["ordinary_r"]["status"] or not r["no_attn_r"]["status"] or not r["no_mlp_r"]["status"]:
                            print(f"Problem started from {known_id}")
                            continue 
                        
                        if known_id > 200:
                            continue

                        for r_type in ["ordinary_r", "no_attn_r", "no_mlp_r"]:
                            pdfname = f'{pdf_dir}/{str(r["ordinary_r"]["answer"]).strip()}_{known_id}{kind_suffix}_{obj_safe}_{r_type}.pdf'
                            plot_trace_heatmap(
                                scores=np.clip(r[r_type]["ie"], 0, None),
                                input_tokens=r[r_type]["input_tokens"], 
                                answer=r[r_type]["answer"], 
                                attributes_loc=r[r_type]["attributes_loc"].values(), 
                                ratio=0.15,
                                kindname=r[r_type]["kind"],
                                title=f"Impact of restoring {r[r_type]['kind'] if r[r_type]['kind'] else 'state'} - {'Experiment A`' if args.patch_type == 'counterfactual' else 'Experiment B`'}",
                                savepdf=pdfname
                            )

                    else:
                        if not os.path.isfile(filename):
                            r = calculate_post_proc(
                                calculate_hidden_flow(
                                    model, prompt, 
                                    samples=args.samples, 
                                    window=args.window, 
                                    reversed_attributes=bool(args.reversed_attributes),
                                    reversed_attributes_noise=float(args.reversed_attributes_noise),
                                    attributes=_attributes, 
                                    attributes_noise=_attributes_noise,
                                    counterfactuals=[obj],
                                    patch_type=args.patch_type,
                                    kind=kind,
                                    n_context=args.n_context,
                                    num_layers=args.num_layers, 
                                    replace=bool(args.replace),
                                ), 
                                prompt=prompt, 
                                samples=args.samples, 
                                patch_type=args.patch_type, 
                            )

                            torch.save(r, filename)
                        else:
                            r = torch.load(filename)

                        if not r["status"]:
                            print(f"Problem started from {known_id}")
                            continue 

                        print(f"knowledge_{known_id}{kind_suffix}_{obj_safe}: TE: {r['te']}, Log(TE): {r['te_log']}")
                        
                        pdfname = f'{pdf_dir}/{str(r["answer"]).strip()}_{known_id}{kind_suffix}_{obj_safe}.pdf'
                        if known_id > 200:
                            continue

                        plot_trace_heatmap(
                            scores=np.clip(r["ie"], 0, None), 
                            input_tokens=r["input_tokens"], 
                            answer=r["answer"], 
                            attributes_loc=r["attributes_loc"].values(), 
                            ratio=0.15,
                            kindname=kind,
                            title=f"Impact of restoring {kind if kind else 'state'} - {'Experiment A' if args.patch_type == 'counterfactual' else 'Experiment B'}",
                            savepdf=pdfname
                        )

if __name__ == "__main__":
    main()
