import json
import os

import torch


def read_json_file(filename, jsonl=False):
    """Reads a JSON file and returns the data."""
    
    if jsonl:
        data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data


def dict_to_args_list(args_dict):
    """Converts a dictionary to a list of command-line arguments. """

    args_list = []
    for key, value in args_dict.items():
        args_list.append("--" + key)
        if not isinstance(value, bool):  # don't add a value for boolean flags
            args_list.append(str(value))
    return args_list
