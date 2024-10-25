import sys

if "./src" not in sys.path:
    sys.path.insert(0, "./src")

import argparse
import os

from experiments.dataset import KnownsDataset
from experiments.tools import collect_embedding_std
from experiments.utils import load_atlas

# Define the templates
TEMPLATE = """
#!/usr/bin/env bash

#SBATCH --partition=alvis
#SBATCH --account={{account_name}}
#SBATCH --nodes=1
#SBATCH --time={{time_days}}-{{time_hours}}:00:00
#SBATCH --job-name=ct-{{experiment_type}}-{{dataset_name}}-{{prop_code}}
#SBATCH --error=./logs/ct-{{experiment_type}}-{{dataset_name}}-{{prop_code}}-{{model_size}}-%J.err.log
#SBATCH --output=./logs/ct-{{experiment_type}}-{{dataset_name}}-{{prop_code}}-{{model_size}}-%J.out.log
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100:1

set -eo pipefail

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

export CUDA_LAUNCH_BLOCKING=1
export USERNAME_DIR={{username_dir}}
export WANDB_CACHE_DIR=../.cache/wandb
export TRANSFORMERS_CACHE=../.cache/huggingface/transformers
export HF_DATASETS_CACHE=../.cache/huggingface/datasets

mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"

source ./venv/bin/activate
echo "PYTHON PATH:"
which python

MODEL_NAME=atlas
SIZE={{model_size}}
READER_MODEL_TYPE=google/t5-${SIZE}-lm-adapt
MODEL_PATH=./data/${MODEL_NAME}/models/atlas_nq/${SIZE}
INDEX_PATH=./data/${MODEL_NAME}/indices/atlas_nq/wiki/${SIZE}

N_CONTEXT=1
NUM_LAYERS=12
MAIN_PORT=$(shuf -i 15000-16000 -n 1)
QA_PROMPT_FORMAT="question: {question} answer: <extra_id_0>"

DATA_PATH={{data_dir}}/{{dataset_name}}/data/matched-{{data_type}}-repr-{{model_size}}/{{prop_code}}.jsonl
OUTPUT_DIR={{output_dir}}/{{dataset_name}}/{{experiment_type}}/matched-{{data_type}}-repr-{{model_size}}/{{prop_code}}

ATTRIBUTES="{{attributes}}" 
ATTRIBUTES_NOISE="{{attributes_noise}}"
PATCH_TYPE="{{patch_type}}"
REPLACE={{replaced_noise}}
SAMPLES=6
WINDOW=6
MAX_KNOWNS=0
REVERSED_ATTRIBUTES={{reversed_attributes}}
REVERSED_ATTRIBUTES_NOISE={{reversed_attributes_noise}}
DISABLE_MLP_ATTN={{disable_mlp_attn}}

echo "EXPERIMENT ({{experiment_type}}): ${OUTPUT_DIR}"

# srun echo $PWD
srun python src/causal_trace.py \\
    --model_name="$MODEL_NAME" \\
    --data_path="$DATA_PATH" \\
    --output_dir="$OUTPUT_DIR" \\
    --attributes="$ATTRIBUTES" \\
    --attributes_noise="$ATTRIBUTES_NOISE" \\
    --patch_type="$PATCH_TYPE" \\
    --replace=$REPLACE \\
    --reader_model_type="$READER_MODEL_TYPE" \\
    --model_path="$MODEL_PATH" \\
    --n_context=$N_CONTEXT \\
    --qa_prompt_format="$QA_PROMPT_FORMAT" \\
    --num_layers=$NUM_LAYERS \\
    --samples=$SAMPLES \\
    --max_knowns=$MAX_KNOWNS \\
    --reversed_attributes=$REVERSED_ATTRIBUTES \\
    --reversed_attributes_noise=$REVERSED_ATTRIBUTES_NOISE \\
    --window=$WINDOW \\
    --disable_mlp_attn=$DISABLE_MLP_ATTN
""".strip()

def generate_bash_file(params, output_file):
    content = TEMPLATE
    for key, value in params.items():
        placeholder = f"{{{{{key}}}}}"
        content = content.replace(placeholder, str(value))
    
    with open(output_file, "w") as file:
        file.write(content)

def main():
    parser = argparse.ArgumentParser(description="Generate bash script for experiments.")
    parser.add_argument("--account_name", help="The account name for SLURM.", default="")
    parser.add_argument("--time_days", type=int, help="The number of days for SLURM time.", default=3)
    parser.add_argument("--time_hours", type=int, help="The number of hours for SLURM time.", default=24)
    parser.add_argument("--experiment_type", required=True, help="The name of the experiment.")
    parser.add_argument("--dataset_name", required=True, help="The name of the dataset.")
    parser.add_argument("--prop_code", required=True, help="The property name.")
    parser.add_argument("--data_type", default="both", help="The property name.")
    parser.add_argument("--username_dir", help="The username directory.", default="")
    parser.add_argument("--model_size", help="The size of the model.", default="base")
    parser.add_argument("--data_dir", required=True, help="The directory of the data.")
    parser.add_argument("--output_dir", help="The directory for the output.", default="")
    parser.add_argument("--output_file", help="The output file path for the generated bash script.", default="")

    args = parser.parse_args()
    experiment_type = args.experiment_type 

    if experiment_type in ["b", "c", "bb", "cc"]:
        knowns = KnownsDataset(os.path.join(args.data_dir, args.dataset_name, f"data/matched-{args.data_type}-repr-{args.model_size}/", f"{args.prop_code}.jsonl"), jsonl=True)
        model, opt = load_atlas(
            reader_model_type=f"google/t5-{args.model_size}-lm-adapt", 
            model_path=f"./data/atlas/models/atlas_nq/{args.model_size}", 
            n_context=1, 
            qa_prompt_format="question: {question} answer: <extra_id_0>"
        )
    if experiment_type == "a" or experiment_type == "aa":
        args.attributes = "obj=context"
        args.attributes_noise = "subj=subj_cf_emb,obj=obj_cf_emb"
        args.patch_type = "counterfactual"
        args.replaced_noise = "1"
        args.reversed_attributes = "0"
        args.reversed_attributes_noise = "0.0"
        args.disable_mlp_attn = 1 if experiment_type == "aa" else 0

    elif experiment_type == "b" or experiment_type == "bb":
        subj_noise_level = collect_embedding_std(model, knowns, attribute="subj", bounds_mode=None) * 3.0

        args.attributes = "subj=context"
        args.attributes_noise = f"subj={subj_noise_level},obj=obj_cf_emb"
        args.patch_type = "clean"
        args.replaced_noise = "1"
        args.reversed_attributes = "0"
        args.reversed_attributes_noise = "0.0"
        args.disable_mlp_attn = 1 if experiment_type == "bb" else 0

    elif experiment_type == "c" or experiment_type == "cc":
        context_noise_level = collect_embedding_std(model, knowns, attribute="context") * 3.0
        args.attributes = "subj=context,obj=context"
        args.attributes_noise = ""
        args.patch_type = "clean"
        args.replaced_noise = "1"
        args.reversed_attributes = "1"
        args.reversed_attributes_noise = f"{context_noise_level}"
        args.disable_mlp_attn = 1 if experiment_type == "cc" else 0
    else:
        raise ValueError('Your experiment type has not defined yet!')

    params = vars(args)
    output_file = os.path.join(params.pop("output_file"), f"ct_{args.dataset_name}_{args.experiment_type}_{args.prop_code}-{args.data_type}-{args.model_size}.sh")
    
    generate_bash_file(params, output_file)

if __name__ == "__main__":
    main()