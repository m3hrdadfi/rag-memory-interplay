#!/usr/bin/env bash

#SBATCH --partition=...
#SBATCH --account=...
#SBATCH --nodes=1
#SBATCH --time=3-24:00:00
#SBATCH --job-name=...
#SBATCH --error=...
#SBATCH --output=...
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100:1

set -eo pipefail

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1

export CUDA_LAUNCH_BLOCKING=1
export PROJECT_DIR=
export CACHE_DIR=

# Change these blocks based on your environment
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/transformers"
export HF_HOME="${CACHE_DIR}/huggingface/transformers"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"

mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"


# Change these blocks based on your environment
source "${PROJECT_DIR}/venv/bin/activate"
echo "PYTHON PATH:"
which python

MODEL_NAME=atlas
SIZE="base"
READER_MODEL_TYPE="google/t5-${SIZE}-lm-adapt"
MODEL_PATH="${PROJECT_DIR}/data/${MODEL_NAME}/models/atlas_nq/${SIZE}"
INDEX_PATH="${PROJECT_DIR}/data/${MODEL_NAME}/indices/atlas_nq/wiki/${SIZE}"

N_CONTEXT=1
NUM_LAYERS=12
MAIN_PORT=$(shuf -i 15000-16000 -n 1)
QA_PROMPT_FORMAT="question: {question} answer: <extra_id_0>"

DATA_PATH=./data/synthetic_context/popqa/matched-both-repr/color.jsonl
OUTPUT_DIR=./experiments/ct/popqa/cc/matched-both-repr/color

ATTRIBUTES="subj=context,obj=context" 
ATTRIBUTES_NOISE=""
PATCH_TYPE="clean"
REPLACE=1
SAMPLES=6
WINDOW=6
MAX_KNOWNS=0
REVERSED_ATTRIBUTES=1
REVERSED_ATTRIBUTES_NOISE=7.335064888000488
DISABLE_MLP_ATTN=1

echo "EXPERIMENT (pse c): ${OUTPUT_DIR}"

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