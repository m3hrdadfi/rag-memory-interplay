import copy
import json
import os
import sys

import torch

from atlas.model_io import load_or_initialize_atlas_model
from atlas.options import get_options
from atlas.util import get_unwrapped_model_if_wrapped
from utils import dict_to_args_list


def load_atlas(reader_model_type, model_path, n_context=1, qa_prompt_format="question: {question} answer: <extra_id_0>", generation_num_beams=5):
    n_context = 1
    args_dict = {
        "reader_model_type": reader_model_type,
        "model_path": model_path,
        "qa_prompt_format": qa_prompt_format,
        "n_context": n_context,
        "retriever_n_context": n_context,
        "generation_num_beams": generation_num_beams,
        "target_maxlength": 32,
        "generation_max_length": 32,
        "text_maxlength": 512,
        "task": "qa"

    }
    args_list = dict_to_args_list(args_dict)
    original_argv = sys.argv
    sys.argv = [""] + args_list
    options = get_options()
    opt = options.parse()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.argv = original_argv

    torch.manual_seed(opt.seed)
    torch.cuda.set_device(0)
    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)
    unwrapped_model = get_unwrapped_model_if_wrapped(model)
    
    model = copy.deepcopy(unwrapped_model).eval().cuda()
    return model, opt


def extract_modules(model, target_paths=None, output_type="both"):
    """
    Extracts modules from a PyTorch model.
    If target_paths is provided, it extracts the specified modules.
    Otherwise, it extracts all modules.

    model: PyTorch model from which to extract modules.
    target_paths: List of module paths to extract (e.g., 'module.submodule').
                         If None, all modules are extracted.
    output_type: Type of output: "names", "modules", or "both".
    return: Depending on output_type, returns names, modules, or tuples of (name, module).
    """

    model = copy.deepcopy(model)

    def module_extractor(module, name, path, result):
        if len(path) == 0:
            if isinstance(module, nn.ModuleList):
                # Extract all modules in the ModuleList if the path ends here
                for i, submodule in enumerate(module):
                    update_result(result, f"{name}.{i}", submodule)
            else:
                update_result(result, name, module)
            return

        if hasattr(module, path[0]):
            next_module = getattr(module, path[0])
            next_name = f"{name}.{path[0]}" if name else path[0]
            module_extractor(next_module, next_name, path[1:], result)

    def update_result(result, name, module):
        if output_type == "names":
            result.append(name)
        elif output_type == "modules":
            result.append(module)
        else:  # output_type == "both"
            result.append((name, module))

    def extract_specific_modules():
        extracted_modules = []
        for path in target_paths:
            module_extractor(model, '', path.split('.'), extracted_modules)
        return extracted_modules

    def extract_all_modules(module, parent_name):
        modules = []
        for name, sub_module in module.named_children():
            module_name = f"{parent_name}.{name}" if parent_name else name
            update_result(modules, module_name, sub_module)
            modules.extend(extract_all_modules(sub_module, module_name))
        return modules

    if target_paths is not None:
        return extract_specific_modules()
    else:
        return extract_all_modules(model, '')

def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}


class AttrDict(dict):
    """
    A dictionary subclass that allows access to its keys through attribute notation.

    Example:
        d = AttrDict({'a': 1, 'b': 2})
        print(d.a)  # prints 1
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def concat(self, other):
        """
        Concatenates another AttrDict's items into this instance and returns a new AttrDict.
        """
        if not isinstance(other, AttrDict):
            raise ValueError("The 'other' argument must be an instance of AttrDict")

        # Create a new AttrDict instance that combines self and other
        merged_dict = AttrDict(self)
        merged_dict.update(other)
        return merged_dict