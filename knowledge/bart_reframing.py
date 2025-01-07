import json
import math
import os
import pandas as pd
import numpy as np
from random import shuffle
import time
import argparse
from torch import Tensor
from random import shuffle
from tqdm import tqdm
import numpy as np
from os.path import join
import json
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer,RobertaForSequenceClassification,AdamW,RobertaConfig
from torch.utils.data import DataLoader, TensorDataset
thinking_traps=["negative feeling or emotion","all-or-nothing thinking","overgeneralizing","labeling","emotional reasoning",
                "comparing and despairing","disqualifying the positive","mind reading","blaming","fortune telling",
                "should statements","personalizing","catastrophizing","not distorted"]

def pad_dataset(dataset, padding=1):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_en_l = max(len(x) for x in dataset["input_ids"])
    max_de_l =max(len(x) for x in dataset["decoder_input_ids"])
    dataset["input_ids"] = [x + [padding] * (max_en_l - len(x)) for x in dataset["input_ids"]]
    dataset["token_type_ids"] = [x + [padding] * (max_en_l - len(x)) for x in dataset["token_type_ids"]]
    dataset["lm_labels"] = [x + [padding] * (max_de_l - len(x)) for x in dataset["lm_labels"]]
    dataset["decoder_input_ids"] = [x + [padding] * (max_de_l - len(x)) for x in dataset["decoder_input_ids"]]
    return dataset


def build_input_from_segments(history, reply,tokenizer,  with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos = BOS_TOKEN_ID
    eos = EOS_TOKEN_ID
    sp1, sp2 = tokenizer.convert_tokens_to_ids(["<speaker1>", "<speaker2>"])
    token_type_ids = []
    for i, u in enumerate(history):
        if i == 0:
            history[i] = [bos] + u
            token_type_ids.append([sp1] * len(history[i]))
            if len(history) == 1:
                break
        elif i % 2 == 1:
            history[i] = u + [eos]
            token_type_ids.append([sp2] * len(history[i]))
        elif i % 2 == 0:
            history[i] = u + [eos]
            token_type_ids.append([sp1] * len(history[i]))
    history = list(chain(*history))
    token_type_ids = list(chain(*token_type_ids))
    if len(reply) > MAX_LENGTH[1]:
        reply = reply[:MAX_LENGTH[1]]
    if len(history) > MAX_LENGTH[0]:
        history = history[:MAX_LENGTH[0]]
        token_type_ids = token_type_ids[:MAX_LENGTH[0]]
    instance = {}
    instance["input_ids"] = history
    instance["token_type_ids"] = token_type_ids
    instance["decoder_input_ids"] =  [bos] + reply
    instance["lm_labels"] =reply + ([eos] if with_eos else [])
    return instance


def get_data_loaders(config, tokenizer):
    # pad=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    """ Prepare the dataset for training and evaluation """
    pad=PAD_TOKEN_ID
    # personachat = get_dataset(tokenizer, config.dataset_path, config.dataset_cache,SPECIAL_TOKENS)
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        chat_dataset = json.loads(f.read())
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list),"test": defaultdict(list)}
    for dataset_name, dataset in chat_dataset.items():
        for dialog in dataset:
            for utterance in dialog["utterances"]:
                history = [tokenizer.encode(s) for s in utterance["history"][-(2 * config.max_history + 1):]]
                reply = tokenizer.encode(utterance["reply"][-1])
                instance= build_input_from_segments(history,reply,tokenizer)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": [],"test":[]}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=pad)
        for input_name in MODEL_INPUTS:
            if input_name=="attention_mask":
                for inputs in dataset["input_ids"]:
                    att_mask=[0 if ids==pad else 1 for ids in inputs ]
                    dataset["attention_mask"].append(att_mask)
            tensor = torch.tensor(dataset[input_name])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["test"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if config.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if config.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=config.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler

'''
Arguments
'''
parser = argparse.ArgumentParser()
parser.add_argument('--training_path', type=str, default='cognitive_reframing/reframing_dataset.csv')
parser.add_argument('--output_dir', type=str, default='cognitive_reframing/model')
parser.add_argument('--thinking_traps_path', type=str, default='cognitive_reframing/thinking_traps.jsonl', help='Path to the training data')
parser.add_argument('--model_path', type=str, default='../../../bart-base', help='pre-trained model path')
parser.add_argument('--test_path', type=str, default='data/sample_test.csv', help='Path to the test data')
parser.add_argument('--output_path', type=str, default='data/sample_test_output.csv', help='Path to the output file')
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--distributed', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--valid_batch_size', type=int, default=1)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument("--fp16", type=bool, default=False)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--top_k', type=int, default=5, help='Number of top matches to retrieve')
parser.add_argument("--num_optim_steps", type=int, default=20000,
                    help="new API specifies num update steps")
parser.add_argument('--max_grad_norm', type=float, default=1.0)
# parser.add_argument('--gpt3_model', type=str, default='text-davinci-003', help='GPT3 model to use')
parser.add_argument('--top_p', type=float, default=0.6, help='Temperature to use for GPT3')

args = parser.parse_args()
init_args_dict = vars(args).copy()
print('###################')
print('Arguments:')

print('Training path: {}'.format(args.training_path))
print('Thinking traps path: {}'.format(args.test_path))

print('###################')

'''
Load the training data
'''
# training_df = pd.read_csv(args.training_path)
# print('Number of training examples: {}'.format(len(training_df)))

with open(args.thinking_traps_path,'r',encoding='utf-8') as f:
    i=0
    traps_data=[]
    for line in f:
        d=json.loads(line)
        traps_data.append(d)
        i+=1
print('Number of training examples: {}'.format(i))
'''
Sentence transformer model
'''
output_dir=args.output_dir
if args.local_rank == -1 :
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(init_args_dict, f, ensure_ascii=False, indent=2)


if args.local_rank == -1 :
    train_logger = open(join(output_dir, 'train_log.csv'), 'a+', buffering=1)
    eval_logger = open(join(output_dir, 'eval_log.csv'), 'a+', buffering=1)
    print('epoch,global_step,step,tmp_loss,tmp_ppl,mean_loss,mean_ppl,n_token_real,'
          'n_token_total,epoch_time', file=train_logger)
    print('epoch,global_step,step,freq_loss,freq_ppl', file=eval_logger)


robera_config=RobertaConfig.from_pretrained(args.model_path)
robera_config.num_labels=13
tokenizer=RobertaTokenizer.from_pretrained(args.model_path)
model = RobertaForSequenceClassification(robera_config)
model.from_pretrained(args.model_path)
train_dataloader,valid_dataloader=get_data_loaders(args, tokenizer, traps_data)
optimizer = AdamW(model.parameters(), lr=args.lr)
if torch.cuda.is_available():
    model.cuda()

while True:
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    step=0
    global_step=0
    for k in range(args.epoch):
        for i,batch in tqdm(enumerate(train_dataloader)):
        # activate new training mode
            input_ids,labels=batch
            if args.device=="cuda":
                input_ids=input_ids.to("cuda")
                labels=labels.to("cuda")
            outputs = model(input_ids=input_ids,labels=labels,return_dict=True)
            loss=outputs.loss
            loss.backward()
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        for i,batch in tqdm(enumerate(valid_dataloader)):
            ppl_lst=[]
        # activate new training mode
            input_ids,labels=batch
            if args.device=="cuda":
                input_ids=input_ids.to("cuda")
                labels=labels.to("cuda")
            outputs = model(input_ids=input_ids,labels=labels,return_dict=True)
            eval_loss=outputs.loss
            ppl=math.exp(loss.data)
            ppl_lst.append(ppl)
        eval_ppl=np.mean(ppl_lst)
        print(f'{k},{global_step + 1},{step + 1},{eval_loss},{eval_ppl}', file=eval_logger)
