# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import csv
import glob
import json
import os
import random
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

from download_tools import maybe_download_file


def find_entity_in_question(template, question):
    # """Find the entity [E] in the question based on the template"""

    # Replace the placeholder [E] with a regex pattern to capture any text
    pattern = template.replace("[E]", "(.+)")
    # Search for the pattern in the question
    match = re.match(pattern, question)
    if match:
        # Return the first capture group, which corresponds to the entity
        return match.group(1)


# random 64 examples used with Atlas
nq_64shot = [
    27144,
    14489,
    49702,
    38094,
    6988,
    60660,
    65643,
    48249,
    48085,
    52629,
    48431,
    7262,
    34659,
    24332,
    44839,
    17721,
    50819,
    62279,
    37021,
    77405,
    52556,
    23802,
    40974,
    64678,
    69673,
    77277,
    18419,
    25635,
    1513,
    11930,
    5542,
    13453,
    52754,
    65663,
    67400,
    42409,
    74541,
    33159,
    65445,
    28572,
    74069,
    7162,
    19204,
    63509,
    12244,
    48532,
    72778,
    37507,
    70300,
    29927,
    18186,
    27579,
    58411,
    63559,
    4347,
    59383,
    57392,
    42014,
    77920,
    45592,
    32321,
    3422,
    61041,
    34051,
]

# random 64 examples used with Atlas
triviaqa_64shot = [
    75927,
    38807,
    452,
    68095,
    44621,
    34592,
    36091,
    65286,
    56484,
    48197,
    34692,
    28011,
    16670,
    62641,
    37865,
    6658,
    45724,
    37527,
    17740,
    31133,
    8010,
    48573,
    53670,
    15514,
    25996,
    54404,
    10739,
    55105,
    66122,
    73324,
    41202,
    71253,
    41258,
    51344,
    60092,
    50455,
    65078,
    36169,
    33408,
    55106,
    40526,
    65582,
    66337,
    39766,
    77174,
    17289,
    7367,
    50930,
    21151,
    21809,
    52804,
    26110,
    54414,
    73358,
    11459,
    66019,
    41084,
    13349,
    39059,
    6626,
    25540,
    15110,
    53320,
    61313,
]

# random 64 examples used with Atlas
popqa_64shot = [
    496,
    2842,
    13746,
    12001,
    3233,
    14034,
    3188,
    11256,
    13544,
    4938,
    14219,
    13810,
    211,
    5097,
    9991,
    10096,
    14242,
    14237,
    369,
    2894,
    13616,
    13883,
    7738,
    13955,
    2865,
    11106,
    11275,
    11284,
    13797,
    11280,
    13782,
    11007,
    17,
    57,
    104,
    107,
    122,
    151,
    170,
    184,
    218,
    222,
    225,
    263,
    283,
    284,
    304,
    349,
    380,
    397,
    441,
    466,
    481,
    500,
    519,
    522,
    523,
    529,
    535,
    548,
    561,
    562,
    564,
    571,
]

peq_64shot = [
    72443,
    156230,
    167269,
    95476,
    19073,
    25208,
    151053,
    57351,
    112610,
    33134,
    111560,
    60257,
    51242,
    152159,
    79798,
    153484,
    6000,
    38731,
    103536,
    101965,
    125890,
    13066,
    2393,
    99986,
    3589,
    173340,
    171981,
    167471,
    84589,
    36905,
    81952,
    98306,
    74290,
    8915,
    70070,
    23597,
    37528,
    63271,
    147378,
    43264,
    130719,
    10086,
    39547,
    169284,
    15111,
    50799,
    76609,
    86873,
    126141,
    49929,
    142203,
    173674,
    90656,
    31280,
    15390,
    165507,
    64359,
    136504,
    49862,
    115480,
    140909,
    109118,
    134612,
    91640,
]


peq_cat = {
    "P170": ["Who was [E] created by?", ""],
    "P112": ["Who founded [E]?", ""],
    "P276": ["Where is [E] located?", ""],
    "P106": ["What kind of work does [E] do?", ""],
    "P131": ["Where is [E] located?", ""],
    "P495": ["Which country was [E] created in?", ""],
    "P175": ["Who performed [E]?", ""],
    "P127": ["Who owns [E]?", ""],
    "P159": ["Where is the headquarter of [E]?", ""],
    "P26": ["Who is [E] married to?", ""],
    "P413": ["What position does [E] play?", ""],
    "P800": ["What is [E] famous for?", ""],
    "P136": ["What type of music does [E] play?", ""],
    "P740": ["Where was [E] founded?", ""],
    "P407": ["Which language was [E] written in?", ""],
    "P50": ["Who is the author of [E]?", ""],
    "P19": ["Where was [E] born?", ""],
    "P20": ["Where did [E] die?", ""],
    "P17": ["Which country is [E] located in?", ""],
    "P69": ["Where was [E] educated?", ""],
    "P176": ["Which company is [E] produced by?", ""],
    "P40": ["Who is [E]'s child?", ""],
    "P264": ["What music label is [E] represented by?", ""],
    "P36": ["What is the capital of [E]?", ""],
}


def convert_peq(ex):
    # {
    #     'question': 'Who was Adoration of the Shepherds created by?',
    #     'answers': ['Le Nain Brothers'],
    #     'cat': 'P170'
    # }
    cat = ex["cat"]
    prop = peq_cat[cat]
    prop = prop[0] if not prop[1] else prop[1]

    return {
        "question": ex["question"],
        "answers": ex["answers"],
        "triplet": {
            "subj": find_entity_in_question(prop, ex["question"]),
            "prop": prop,
            "prop_code": cat,
            "obj": ex["answers"][0],
        },
    }


def convert_popqa(ex):
    # {
    #     'id': 4222362,
    #     'subj': 'George Rankin',
    #     'prop': 'occupation',
    #     'obj': 'politician',
    #     'subj_id': 1850297,
    #     'prop_id': 22,
    #     'obj_id': 2834605,
    #     's_aliases': '["George James Rankin"]',
    #     'o_aliases': '["political leader","political figure","polit.","pol"]',
    #     's_uri': 'http://www.wikidata.org/entity/Q5543720',
    #     'o_uri': 'http://www.wikidata.org/entity/Q82955',
    #     's_wiki_title': 'George Rankin',
    #     'o_wiki_title': 'Politician',
    #     's_pop': 142,
    #     'o_pop': 25692,
    #     'question': "What is George Rankin's occupation?",
    #     'possible_answers': '["politician", "political leader", "political figure", "polit.", "pol"]'
    # }
    return {
        "question": ex["question"],
        "answers": ast.literal_eval(ex["possible_answers"]),
        "triplet": {
            "subj": ex["subj"],
            "prop": ex["prop"],
            "obj": ex["obj"],
        },
        "views": {
            "s_pop": ex["s_pop"],
            "o_pop": ex["o_pop"],
        },
    }


def convert_triviaqa(ex):
    target = ex["Answer"]["Value"]
    if target.isupper():
        target = target.title()
    return {
        "question": ex["Question"],
        "answers": ex["Answer"]["Aliases"],
        "target": target,
    }


def convert_nq(ex):
    return {"question": ex["question"], "answers": ex["answer"]}


def preprocess_peq(orig_dir, output_dir):
    data = {}
    data["train.64-shot"] = []

    for split in ["train", "dev", "test"]:
        data[split] = []

        idx = 0
        for jfile in glob.glob(str(orig_dir / split / "*.json")):
            with open(jfile, "r", encoding="utf-8") as fj:
                jdata = json.load(fj)
                for row in jdata:

                    row["cat"] = jfile.split("/")[-1].split(".")[0]
                    example = convert_peq(row)

                    if split == "train":
                        if idx in peq_64shot:
                            data["train.64-shot"].append(example)

                    data[split].append(example)
                    idx += 1

        if split == "train":
            random.shuffle(data[split])

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w") as fout:
            for ex in data[split]:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")


def preprocess_popqa(orig_dir, output_dir):
    data = {}
    row_key = [
        "id",
        "subj",
        "prop",
        "obj",
        "subj_id",
        "prop_id",
        "obj_id",
        "s_aliases",
        "o_aliases",
        "s_uri",
        "o_uri",
        "s_wiki_title",
        "o_wiki_title",
        "s_pop",
        "o_pop",
        "question",
        "possible_answers",
    ]

    with open(orig_dir / "popQA.tsv", "r") as fin:
        data["test"] = []
        data["test.64-shot"] = []
        reader = csv.reader(fin, delimiter="\t")

        for i, row in enumerate(reader):
            if i == 0:
                continue

            row = {key: row[j] for j, key in enumerate(row_key)}
            example = convert_popqa(row)

            data["test"].append(example)

            if i in popqa_64shot:
                data["test.64-shot"].append(example)

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w") as fout:
            for ex in data[split]:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")


def preprocess_triviaqa(orig_dir, output_dir, index_dir):
    data, index = {}, {}
    for split in ["train", "dev", "test"]:
        with open(index_dir / ("TQA." + split + ".idx.json"), "r") as fin:
            index[split] = json.load(fin)

    with open(orig_dir / "triviaqa-unfiltered" / "unfiltered-web-train.json") as fin:
        originaltrain = json.load(fin)["Data"]
    with open(orig_dir / "triviaqa-unfiltered" / "unfiltered-web-dev.json") as fin:
        originaldev = json.load(fin)["Data"]

    data["train"] = [convert_triviaqa(originaltrain[k]) for k in index["train"]]
    data["train.64-shot"] = [
        convert_triviaqa(originaltrain[k]) for k in triviaqa_64shot
    ]
    data["dev"] = [convert_triviaqa(originaltrain[k]) for k in index["dev"]]
    data["test"] = [convert_triviaqa(originaldev[k]) for k in index["test"]]

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w") as fout:
            for ex in data[split]:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")


def preprocess_nq(orig_dir, output_dir, index_dir):
    data, index = {}, {}
    for split in ["train", "dev", "test"]:
        with open(index_dir / ("NQ." + split + ".idx.json"), "r") as fin:
            index[split] = json.load(fin)

    originaltrain, originaldev = [], []
    with open(orig_dir / "NQ-open.dev.jsonl") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaldev.append(example)

    with open(orig_dir / "NQ-open.train.jsonl") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaltrain.append(example)

    data["train"] = [convert_nq(originaltrain[k]) for k in index["train"]]
    data["train.64-shot"] = [convert_nq(originaltrain[k]) for k in nq_64shot]
    data["dev"] = [convert_nq(originaltrain[k]) for k in index["dev"]]
    data["test"] = [convert_nq(originaldev[k]) for k in index["test"]]

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w") as fout:
            for ex in data[split]:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")


def main(args):
    output_dir = Path(args.output_directory)

    index_tar = output_dir / "index.tar"
    index_dir = output_dir / "dataindex"

    original_triviaqa_dir = output_dir / "original_triviaqa"
    triviaqa_dir = output_dir / "triviaqa_data"
    triviaqa_tar = output_dir / "triviaqa_data.tar"

    nq_dir = output_dir / "nq_data"
    original_nq_dir = output_dir / "original_naturalquestions"

    original_popqa_dir = output_dir / "original_popqa"
    popqa_dir = output_dir / "popqa_data"
    popqa_tsv = original_popqa_dir / "popQA.tsv"

    original_peq_dir = (
        output_dir / "original_peq"
    )
    peq_dir = output_dir / "peq_data"
    peq_zip = (
        original_peq_dir / "peq.zip"
    )

    if args.overwrite:
        print("Overwriting NaturalQuestions and TriviaQA and PopQA")
        download_triviaqa = True
        download_nq = True
        download_popqa = True
        download_peq = True
    else:
        download_triviaqa = not triviaqa_dir.exists()
        download_nq = not nq_dir.exists()
        download_popqa = not popqa_dir.exists()
        download_peq = not peq_dir.exists()

    if download_peq:
        peq_dir.mkdir(parents=True, exist_ok=True)
        original_peq_url = (
            "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip"
        )
        maybe_download_file(
            original_peq_url, peq_zip
        )
        # if not os.path.exists(original_peq_dir):
        with zipfile.ZipFile(peq_zip, "r") as zp:
            zp.extractall(original_peq_dir)

        preprocess_peq(
            original_peq_dir / "dataset",
            peq_dir,
        )
    else:
        print("PopQA data already exists, not overwriting")

    if download_popqa:
        popqa_dir.mkdir(parents=True, exist_ok=True)
        original_popqa_url = "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv"
        maybe_download_file(original_popqa_url, popqa_tsv)
        preprocess_popqa(original_popqa_dir, popqa_dir)
    else:
        print("PopQA data already exists, not overwriting")

    if download_triviaqa or download_nq:
        index_url = "https://dl.fbaipublicfiles.com/FiD/data/dataindex.tar.gz"
        maybe_download_file(index_url, index_tar)
        if not os.path.exists(index_dir):
            with tarfile.open(index_tar) as tar:
                tar.extractall(index_dir)

    if download_triviaqa:
        triviaqa_dir.mkdir(parents=True, exist_ok=True)
        original_triviaqa_url = (
            "http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz"
        )
        maybe_download_file(original_triviaqa_url, triviaqa_tar)
        if not os.path.exists(original_triviaqa_dir):
            with tarfile.open(triviaqa_tar) as tar:
                tar.extractall(original_triviaqa_dir)
        preprocess_triviaqa(original_triviaqa_dir, triviaqa_dir, index_dir)
    else:
        print("TriviaQA data already exists, not overwriting")

    if download_nq:
        nq_dir.mkdir(parents=True, exist_ok=True)
        nq_dev_url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl"
        nq_train_url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl"
        maybe_download_file(nq_dev_url, original_nq_dir / "NQ-open.dev.jsonl")
        maybe_download_file(nq_train_url, original_nq_dir / "NQ-open.train.jsonl")
        preprocess_nq(original_nq_dir, nq_dir, index_dir)
    else:
        print("NaturalQuestions data already exists, not overwriting")

    triviaqa_tar.unlink(missing_ok=True)
    index_tar.unlink(missing_ok=True)
    if original_peq_dir.exists():
        shutil.rmtree(original_peq_dir)
    if original_popqa_dir.exists():
        shutil.rmtree(original_popqa_dir)
    if original_triviaqa_dir.exists():
        shutil.rmtree(original_triviaqa_dir)
    if original_nq_dir.exists():
        shutil.rmtree(original_nq_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data/",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite data")
    args = parser.parse_args()
    main(args)
    main(args)
