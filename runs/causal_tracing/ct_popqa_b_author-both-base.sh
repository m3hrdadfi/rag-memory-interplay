#!/usr/bin/env bash

#SBATCH --partition=alvis
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --time=3-24:00:00
#SBATCH --job-name=ct-b-popqa-author
#SBATCH --error=./logs/ct-b-popqa-author-base-%J.err.log
#SBATCH --output=./logs/ct-b-popqa-author-base-%J.out.log
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100:1

set -eo pipefail

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

export CUDA_LAUNCH_BLOCKING=1
export USERNAME_DIR=
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
SIZE=base
READER_MODEL_TYPE=google/t5-${SIZE}-lm-adapt
MODEL_PATH=./data/${MODEL_NAME}/models/atlas_nq/${SIZE}
INDEX_PATH=./data/${MODEL_NAME}/indices/atlas_nq/wiki/${SIZE}

N_CONTEXT=1
NUM_LAYERS=12
MAIN_PORT=$(shuf -i 15000-16000 -n 1)
QA_PROMPT_FORMAT="question: {question} answer: <extra_id_0>"

DATA_PATH=./data/syn/popqa/data/matched-both-repr-base/author.jsonl
OUTPUT_DIR=./experiments/popqa/b/matched-both-repr-base/author

ATTRIBUTES="subj=context" 
ATTRIBUTES_NOISE="subj=17.200125217437744,obj=obj_cf_emb"
PATCH_TYPE="clean"
REPLACE=1
SAMPLES=6
WINDOW=6
MAX_KNOWNS=0
REVERSED_ATTRIBUTES=0
REVERSED_ATTRIBUTES_NOISE=0.0
DISABLE_MLP_ATTN=0

echo "EXPERIMENT (b): ${OUTPUT_DIR}"

# srun echo $PWD
srun python src/causal_trace.py \
    --model_name="$MODEL_NAME" \
    --data_path="$DATA_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --attributes="$ATTRIBUTES" \
    --attributes_noise="$ATTRIBUTES_NOISE" \
    --patch_type="$PATCH_TYPE" \
    --replace=$REPLACE \
    --reader_model_type="$READER_MODEL_TYPE" \
    --model_path="$MODEL_PATH" \
    --n_context=$N_CONTEXT \
    --qa_prompt_format="$QA_PROMPT_FORMAT" \
    --num_layers=$NUM_LAYERS \
    --samples=$SAMPLES \
    --max_knowns=$MAX_KNOWNS \
    --reversed_attributes=$REVERSED_ATTRIBUTES \
    --reversed_attributes_noise=$REVERSED_ATTRIBUTES_NOISE \
    --window=$WINDOW \
    --disable_mlp_attn=$DISABLE_MLP_ATTN