# coding=utf-8

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import PreTrainedModel
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH
import nltk
from random import shuffle
import sys
sys_path = sys.path[0]
with open(sys_path + (f'/CONFIG/strat.json'), 'r', encoding='utf-8') as c:
    emo_dict = json.load(c)["emolabel2id"]
class Inputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features
        
        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader
        
        # valid
        self.valid_dataloader = DynamicBatchingLoader
        
        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
        self,
        input_ids,
        decoder_input_ids, labels,persona_ids,emotion,comet_ids,last_text,last_text_ids,sys_begin_ids,usr_begin_ids,reframing_ids
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        self.comet_length=len(comet_ids)
        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels
        self.persona_ids=persona_ids
        self.emotion = emotion
        self.comet_ids=comet_ids
        self.last_text=last_text
        self.last_text_ids=last_text_ids
        self.sys_begin_ids=sys_begin_ids
        self.usr_begin_ids=usr_begin_ids
        self.reframing_ids=reframing_ids
        self.input_len = self.input_length + self.decoder_input_length
        self.padding_length = max(self.input_length, len(persona_ids))


def featurize(
    toker,bos, eos,
    context, max_input_length,
    response, max_decoder_input_length, strat_id,persona,emotion,comet,last_text,reframing_thoughts
):

    seeker_ids=toker.convert_tokens_to_ids("[seeker]")
    supportor_ids=toker.convert_tokens_to_ids("[supportor]")
    # context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]+[supportor_ids]
    input_ids = input_ids[-max_input_length:]
    comet_ids = comet[-max_input_length:]
    labels = ([strat_id] + response + [eos])[:max_decoder_input_length + 1]
    decoder_input_ids = [bos] + labels[:-1]
    last_text_ids = [toker("[seeker]").input_ids[0]] + toker(last_text).input_ids + [eos]
    persona_ids=persona#[-max_input_length:]
    # final_persona_ids = final_persona  # [-max_input_length:]
    sys_begin_ids=[]
    for i in range(len(input_ids)):
        if input_ids[i] ==supportor_ids:
            sys_begin_ids.append(i)
    usr_begin_ids = []
    for i in range(len(input_ids)):
        if input_ids[i] == supportor_ids:
            usr_begin_ids.append(i)
    reframing_thought_ids=reframing_thoughts[:50]
    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,persona_ids,emotion,comet_ids,last_text,last_text_ids,sys_begin_ids,usr_begin_ids,reframing_thought_ids
    )

def preprocess_comet(c):
    pre_c=[]
    for item in c:
        item=item.strip(" ")
        if item!= "none":
            pre_c.append(item.replace("to ",""))
    return pre_c[:1]

def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))
def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm=torch.nn.functional.normalize(a,p=2,dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def convert_data_to_inputs(data, toker: PreTrainedTokenizer,reframing_data,reframing_data_emb,model:PreTrainedModel,classifier:PreTrainedModel,K, **kwargs):
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    user_number=0
    dialog = data['dialog']
    inputs = []
    context = []
    context_ids=[]
    # problem_type=data["problem_type"]
    persona_list=data["persona"]
    seeker_id=process('[seeker]')[0]
    # reframing_id= process('[reframing]')[0]
    supportor_id = process('[supportor]')[0]
    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        text_ids = process(text)
        if dialog[i]['speaker'] != 'sys':
            user_number += 1
        if dialog[i]['speaker'] == 'sys':
            strat_id = process('[' + dialog[i]['strategy'] + ']')
            assert len(strat_id) == 1
            strat_id = strat_id[0]
        
        if i > 0 and dialog[i]['speaker'] == 'sys':
            a = 1
            while dialog[i - a]['speaker'] == 'sys':
                a = a + 1
            s1 = "My intent is to " + " ".join(preprocess_comet(dialog[i - a]['comet'][0]))+"."
            s2 = "I " + " ".join(preprocess_comet(dialog[i - a]['comet'][1]))+"."
            s3 = "I need to " + " ".join(preprocess_comet(dialog[i - a]['comet'][2]))+"."
            s4 = "I want to " + " ".join(preprocess_comet(dialog[i - a]['comet'][3]))+"."
            s5 = "I am " + " ".join(preprocess_comet(dialog[i - a]['comet'][4]))+"."
            self_cs_list = [s1, s2, s3, s4, s5]
            comet=process("[comet] "+"[comet] ".join(self_cs_list))
            last_text = _norm(dialog[i - a]['text'])
            emotion=classifier(last_text)
            if kwargs["reframing"] == True and user_number > 2 :
                reframing_thoughts=process("[reframing] "+dialog[i-a]['reframing_thought'])
            else:
                reframing_thoughts =process("[reframing] "+dialog[i-a]['text'])
            # 选择相关的个性化信息

            if user_number > 2:
                persona = persona_list[user_number - 3]
                persona = persona.replace('<persona>', ' ').strip()
                persona = persona.replace('<input>', ' ', 1).strip()
                persona="[persona] "+ persona
                # persona_sen_list=nltk.sent_tokenize(persona)
                # context_text=" ".join(context.copy()[-2:])
                # conetxt_encode_outputs = model.encode(context_text)
                # sentence_encode_outputs =model.encode(persona_sen_list)
                # persona_scores = dot_score(conetxt_encode_outputs ,sentence_encode_outputs)[0].cpu().tolist()
                # persona_ids=[i for i in range(len(persona_sen_list))]
                # persona_score_pairs = list(zip(persona_ids, persona_scores))
                # persona_sorted = sorted(persona_score_pairs, key=lambda x: x[1],reverse=True)
                # matched_persona_ids = [x[0] for x in persona_sorted[:K]]
                # # shuffle(matched_thought_record_ids)
                # persona=""
                # for i in matched_persona_ids:
                #     persona=persona+persona_sen_list[i]
            else:
                persona = "[persona]"

            persona = process(persona)
            res = {
                'context': context_ids.copy(),
                'response': text_ids,
                'comet':comet,
                'persona':persona,
                'emotion': emotion,
                'strat_id': strat_id,
                'last_text':last_text,
                'reframing_thoughts':reframing_thoughts
            }
            
            inputs.append(res)

        if dialog[i]['speaker'] == 'sys':
            text="[supportor]"+" ["+dialog[i]['strategy']+"] "+text
            text_ids = [supportor_id]+[strat_id] + text_ids
        else :
            text="[seeker] "+text
            text_ids=[seeker_id]+text_ids
        context.append(text)
        context_ids = context_ids + [text_ids]

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')
    
    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
    
    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(toker,
            bos, eos,
            ipt['context'], max_input_length,
            ipt['response'], max_decoder_input_length, ipt["strat_id"], ipt['persona'],ipt['emotion'],ipt["comet"],ipt["last_text"],ipt["reframing_thoughts"]
        )
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
        max_len = max([f.padding_length for f in features])
        max_turn = max([len(f.usr_begin_ids) for f in features])
        sys_input_turns = torch.tensor([len(f.sys_begin_ids) for f in features], dtype=torch.long)
        sys_begin_ids = pad_sequence([torch.tensor(f.sys_begin_ids, dtype=torch.long) for f in features],
                                    batch_first=True, padding_value=pad)
        usr_input_turns = torch.tensor([len(f.usr_begin_ids) for f in features], dtype=torch.long)
        usr_begin_ids = pad_sequence([torch.tensor(f.usr_begin_ids, dtype=torch.long) for f in features],
                                     batch_first=True, padding_value=pad)
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                          batch_first=True, padding_value=0.)
        input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)
        comet_ids = pad_sequence([torch.tensor(f.comet_ids, dtype=torch.long) for f in features],
                                 batch_first=True, padding_value=pad)
        comet_attention_mask = pad_sequence([torch.tensor([1.] * f.comet_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)
        persona_ids = pad_sequence([torch.tensor(f.persona_ids, dtype=torch.long) for f in features],
                                 batch_first=True, padding_value=pad)
        persona_attention_mask = pad_sequence([torch.tensor([1.] * len(f.persona_ids), dtype=torch.float) for f in features],
                                            batch_first=True, padding_value=0.)

        reframing_ids = pad_sequence([torch.tensor(f.reframing_ids, dtype=torch.long) for f in features],
                                   batch_first=True, padding_value=pad)
        reframing_attention_mask = pad_sequence(
            [torch.tensor([1.] * len(f.reframing_ids), dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.)
        last_text = [[f.last_text] for f in features]
        # reframing_ids = pad_sequence([torch.tensor(sen, dtype=torch.long) for f in features for sen in f.reframing_ids],
        #                            batch_first=True, padding_value=pad)
        # reframing_attention_mask = pad_sequence(
        #     [torch.tensor([1.] * len(sen), dtype=torch.float) for f in features for sen in f.reframing_ids],
        #     batch_first=True, padding_value=0.)
        # reframing_ids = []
        # reframing_attention_mask = []
        # max_reframing_len = 50
        # for f in features:
        #     one_reframing = []
        #     one_reframing_msk = []
        #     for one in f.reframing_ids:
        #         if len(one) < max_reframing_len:
        #             one_msk = [1.] * len(one) + [0.] * (max_reframing_len - len(one))
        #             one = one + [pad] * (max_reframing_len - len(one))
        #         else:
        #             one = one[:max_reframing_len]
        #             one_msk = [1.] * max_reframing_len
        #         one_reframing.append(one)
        #         one_reframing_msk.append(one_msk)
        #     reframing_ids.append(one_reframing)
        #     reframing_attention_mask.append(one_reframing_msk)
        # reframing_ids = torch.LongTensor(reframing_ids)
        # reframing_attention_mask = torch.FloatTensor(reframing_attention_mask)
        last_input_ids = pad_sequence([torch.tensor(f.last_text_ids[:-1], dtype=torch.long) for f in features],
                                   batch_first=True, padding_value=pad)
        last_labels = pad_sequence([torch.tensor(f.last_text_ids[1:], dtype=torch.long) for f in features],
                                      batch_first=True, padding_value=pad)
        last_attention_mask = pad_sequence(
            [torch.tensor([1.] * len(f.last_text_ids[:-1]), dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.)
        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                              batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels[1:], dtype=torch.long) for f in features],
                              batch_first=True, padding_value=-100)
            decoder_attention_mask=pad_sequence(
            [torch.tensor([1.] * len(f.decoder_input_ids), dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None
            decoder_attention_mask=None

        strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) #- len(toker) + 8
        emotion = torch.tensor([emo_dict[f.emotion[0]["label"] ]for f in features], dtype=torch.long)  # - len(toker) + 8
        emotion_score = torch.tensor([f.emotion[0]["score"] for f in features], dtype=torch.float)
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            #'input_length': input_length,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'strat_id': strat_id,
            "emotion":emotion,
            "emotion_score":emotion_score,
            "persona_ids":persona_ids,
            'persona_attention_mask':persona_attention_mask,
            # "final_persona_ids": final_persona_ids,
            # 'final_persona_attention_mask': final_persona_attention_mask,
            'comet_ids': comet_ids,
            'comet_attention_mask':comet_attention_mask,
            'reframing_ids':reframing_ids,
            'reframing_attention_mask':reframing_attention_mask,
            # 'last_text_ids':last_text_ids,
            # "sys_input_turns":sys_input_turns,
            # "sys_begin_ids":sys_begin_ids,
            "usr_input_turns": usr_input_turns,
            "usr_begin_ids": usr_begin_ids,
            "last_input_ids":last_input_ids,
            "last_labels":last_labels,
            'last_text':last_text,
            'last_attention_mask':last_attention_mask,
            "decoder_attention_mask":decoder_attention_mask
        }
        
        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """
    def __init__(self, corpus_file, toker, batch_size,reframing_data, reframing_data_emb, sentence_model, classifier,**kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.reframing_data = reframing_data
        self.reframing_data_emb = reframing_data_emb
        self.sentence_model = sentence_model
        self.classifier=classifier
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()
            
            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker,self.reframing_data,self.reframing_data_emb,self.sentence_model,self.classifier,1  ,**self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []
                    
            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch
                
        except StopIteration:
            pass
    
    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


# for inference
def prepare_infer_batch(features, toker, interact=None):
    res = FeatureDataset.collate(features, toker, True)
    
    res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_strat_id': 'pred_strat_id',
    }

    if interact is None and GOLDEN_TRUTH:
        other_res['cls_strat_id'] = res.get('strat_id')-len(toker)+8
    else:
        other_res['cls_strat_id'] = res.pop('strat_id')-len(toker)+8

    return res


def get_infer_batch(infer_input_file, toker,reframing_data_df,reframing_situation_and_thought_emb,sentence_model,classifier, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()
    
    features = []
    sample_ids = []
    posts = []
    references = []
    reframing_thoughts = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker,reframing_data_df,reframing_situation_and_thought_emb,sentence_model,classifier,1, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            post=ipt['context'][-4:]
            posts.append([toker.decode(sen) for sen in post])
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)
            if ipt['reframing_thoughts'] is None:
                reframing_thoughts.append('')
            else:
                reframing_thoughts.append(ipt['reframing_thoughts'])
            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references,reframing_thoughts, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references,reframing_thoughts, sample_ids
