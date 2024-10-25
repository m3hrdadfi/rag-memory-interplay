#!/usr/bin/env bash

echo "PYTHON PATH:"
which python
pwd


# Directory paths and other settings
DATA_DIR="./data/syn"
DATA_TYPE="both"
OUTPUT_DIR="./runs/causal_tracing"

# List of model sizes
model_sizes=(
    "base"
    # "large"
    # "xl"
)

# List of datasets and properties
datasets_props=(
    "popqa,author"
    # "popqa,capital_of"
    # "popqa,capital"
    # "popqa,color"
    # "popqa,composer"
    # "popqa,country"
    # "popqa,director"
    # "popqa,father"
    # "popqa,genre"
    # "popqa,mother"
    # "popqa,occupation"
    # "popqa,place_of_birth"
    # "popqa,producer"
    # "popqa,religion"
    # "popqa,screenwriter"
    # "popqa,sport"
)

# Experiment types
experiment_types=(
    "a"
    "b"
    "c"
    "aa"  # pse a
    "bb"  # pse b
    "cc"  # pse c
)

# Iterate over each dataset and property
for model_size in "${model_sizes[@]}"; do
    for dataset_prop in "${datasets_props[@]}"; do
        IFS=',' read -r dataset prop <<< "$dataset_prop"
        
        # Iterate over each experiment type
        for exp_type in "${experiment_types[@]}"; do
            # Construct the output directory for the current script
            output_subdir="${OUTPUT_DIR}"
            mkdir -p "$output_subdir"
                
            # Run the Python script to generate the SLURM script
            python preprocessing/scripts/generate_ct_run.py \
                --data_type "$DATA_TYPE" \
                --model_size "$model_size" \
                --data_dir "$DATA_DIR" \
                --experiment_type "$exp_type" \
                --dataset_name "$dataset" \
                --prop_code "$prop" \
                --output_dir "./experiments" \
                --output_file "$output_subdir"
            
            echo "Generated script for ${dataset}, ${prop}, experiment ${exp_type} on ${model_size} size"
        done
    done
done

echo "All scripts generated."
