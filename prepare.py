# coding=utf-8

import argparse
import json
import shutil
import pickle
import os
import logging
import multiprocessing as mp
from os.path import dirname, exists, join
import pandas as pd
import torch
import tqdm
from inputters import inputters
from utils.building_utils import build_model
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str,default="strat")
parser.add_argument('--inputter_name', type=str, default="strat")
parser.add_argument('--train_input_file', type=str, default="")
parser.add_argument('--max_input_length', type=int, default=300, help='discard data longer than this')
parser.add_argument('--max_decoder_input_length', type=int, default=50, help='discard data longer than this')
parser.add_argument('--max_knowledge_length', type=int, default=None, help='discard data longer than this')
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--only_encode', action='store_true', help='only do encoding')
parser.add_argument('--single_processing',type=bool, default=True)
parser.add_argument('--reframing', type=bool, default=True)
parser.add_argument('--max_reframing_length', type=int, default=50, help='discard data longer than this')
parser.add_argument("--go_emotions", type=str, default="")
args = parser.parse_args()
sys_path = sys.path[0]
names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

inputter = inputters[args.inputter_name]()
toker = build_model(only_toker=True, **names)

from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('')
reframing_data_df=None
reframing_situation_and_thought_emb=None

if args.go_emotions is not None:
    from transformers import pipeline

    classifier = pipeline(task="text-classification", model=args.go_emotions)
    with open(sys_path + (f'/CONFIG/{args.config_name}.json'), 'r', encoding='utf-8') as c:
        emo_dict = json.load(c)["emolabel2id"]


with open(args.train_input_file) as f:
    reader = f.readlines()

if not os.path.exists(f'./DATA'):
    os.mkdir(f'./DATA')
save_dir = f'./DATA/{args.inputter_name}.{args.config_name}'
if not exists(save_dir):
    os.mkdir(save_dir)

kwargs = {
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knowledge_length': args.max_knowledge_length,
    'label_num': args.label_num,
    'only_encode': args.only_encode,
    'reframing':args.reframing,
    "max_reframing_length":args.max_reframing_length,
}

def process_data(line):
    data = json.loads(line)
    inputs = inputter.convert_data_to_inputs(
        data=data,
        toker=toker,
        reframing_data=reframing_data_df,
        reframing_data_emb=reframing_situation_and_thought_emb,
        model=sentence_model,
        classifier=classifier,
        K=1,
        **kwargs
    )
    features = inputter.convert_inputs_to_features(
        inputs=inputs,
        toker=toker,
        **kwargs,
    )
    return features

processed_data = []
if args.single_processing:
    for features in map(process_data, tqdm.tqdm(reader, total=len(reader))):
        processed_data.extend(features)
else:
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for features in pool.imap(process_data, tqdm.tqdm(reader, total=len(reader))):
            processed_data.extend(features)

# save data
data_path = f'{save_dir}/data.pkl'
with open(data_path, 'wb') as file:
    pickle.dump(processed_data, file)
kwargs.update({'n_examples': len(processed_data)})
# save relevant information to reproduce
with open(f'{save_dir}/meta.json', 'w') as writer:
    json.dump(kwargs, writer, indent=4)
torch.save(toker, f'{save_dir}/tokenizer.pt')
