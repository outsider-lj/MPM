import json
import pickle
import csv
from collections import defaultdict
relation=['SimilarTo','IsA','FormOf']
def to_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
def is_all_eng(strs):
    import string
    for i in strs:
        if i not in string.ascii_lowercase+string.ascii_uppercase:
            return False
    return True
#处理conceptnet的常识数据集
from ast import literal_eval
def get_ngrams(utter, n):
    sep_utter=[]
    for s in utter:
        sep_s=s.split(" ")
        sep_utter.append(sep_s)
    utter=utter.split(" ")
    total = []
    for i in range(len(utter)):
        for j in range(i, max(i - n, -1), -1):
            total.append("_".join(utter[j:i + 1]))
    return total

def get_all_ngrams(examples, n):
    all_ngrams = []
    for ex in examples:
        for utterance in ex["dialog"]:
            utter=utterance["content"]#+utterance["reply"][0]
            all_ngrams.extend(get_ngrams(utter, n))
    return set(all_ngrams)

def filtered_knowledge(word,concept):
    emo_scores = []
    filtered_knowledge = set()
    filtered_conceptnet = set()
    concepts = set()
    filtered_concepts = sorted(concept, key=lambda x: x[2], reverse=True)
    for c,r,w in filtered_concepts:
        if c not in concepts:#将重复的尾结点去除
            filtered_conceptnet.add((c,r,w))
            concepts.add(c)
    # max+=min+0.1
    for triple in iter(filtered_conceptnet):
        if triple[1] in relation and triple[2] > 1 : #and triple[0] in nrc.keys():
            concept_embeddings = []
            tail_entity = triple[0].split("_")
            if is_all_eng(tail_entity[0]) is False:
                continue
            filtered_knowledge.add((triple[0], triple[1], triple[2]))

        else:
            continue
        # filtered_knowledge=remove_KB_duplicates(filtered_knowledge)
    return filtered_knowledge

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default="concept")
parser.add_argument('--n', default=1)
args = parser.parse_args()
#
data=json.load(open("../../PESConv.json",encoding="utf-8"))
dataset = args.dataset
n = args.n
ngrams = get_all_ngrams(data,n)
print("Loading conceptnet...")
csv_reader = csv.reader(open("../../../ConceptNet/assertions.csv", "r"), delimiter="\t")
concept_dict = defaultdict(set)
for i, row in enumerate(csv_reader):
    if i % 1000000 == 0:
        print("Processed {0} rows".format(i))
    lang = row[2].split("/")[2]
    if lang == 'en':
        c1 = row[2].split("/")[3]
        c1_lang=row[2].split("/")[2]
        c2 = row[3].split("/")[3]
        c2_lang=row[3].split("/")[2]
        r=row[1].split("/")[2]
        weight = literal_eval(row[-1])["weight"]
        if c1 in ngrams and c1_lang=='en':
            concept_dict[c1].add((c2,r, weight))
        if c2 in ngrams and c2_lang=='en':
            concept_dict[c2].add((c1,r, weight))
print("Saving concepts...")
conceptnet=defaultdict(list)
to_pickle(concept_dict, "concept.pkl")
concept = pickle.load(open("concept.pkl", "rb"))
for w,c in concept.items():
        # filtered_conceptnet = remove_KB_duplicates()
    filtered_conceptnet=filtered_knowledge(w,c)
    if len(filtered_conceptnet) <1:
            continue
    filtered_conceptnet = sorted(filtered_conceptnet, key=lambda x: x[2], reverse=True)
    conceptnet[w]=filtered_conceptnet
pickle.dump(conceptnet,open("filtered_conceptnet.pkl","wb"))
print("saved filter_conceptnet!")
