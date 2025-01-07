import json
import tqdm
from comet import Comet
relations = ["xIntent","xEffect","xNeed", "xWant", "xReact"]
def get_commonsense(comet, utter):
    self_cs_list = []
    # other_cs_list=[]
    for rel in relations:
        self_cs_res = comet.generate(utter, rel)
        self_cs_list.append(self_cs_res)
    return self_cs_list#,other_cs_list
comet = Comet("/home/lijing/code/Comet_model", "cuda")
from transformers import BartForConditionalGeneration, BartTokenizer,pipeline
bart_tokenizer = BartTokenizer.from_pretrained("/home/lijing/code/ESC_LJ/codes_frame/output/reframer")
bart_model = BartForConditionalGeneration.from_pretrained("/home/lijing/code/ESC_LJ/codes_frame/output/reframer")
reframer = pipeline('summarization', model=bart_model, tokenizer=bart_tokenizer)
def reprocess_data(data):
    problem_type=data["problem_type"]
    for i in range(len(data["dialog"])):
        if data["dialog"][i]["speaker"]=="usr":
            text=data["dialog"][i]["text"]
            self_cs_list = get_commonsense(comet, text)
            curr_thought=problem_type + ' ' + text
            # model_inputs = bart_toker(inputs, max_length=512, truncation=True)
            predictions =reframer(curr_thought)[0]['summary_text']
            data["dialog"][i].update({"comet":self_cs_list,"reframing_thought":predictions})
    return data

train_comet=[]
with open('./train.txt', 'r',encoding="utf-8") as f:
    reader = f.readlines()
with open('./train.txt', 'r', encoding="utf-8") as f:
    for line in tqdm.tqdm(reader, total=len(reader)):
        data = json.loads(line)
        re_data=reprocess_data(data)
        train_comet.append(re_data)
with open('./train_new.txt', 'w') as f:
    for e in train_comet:
        f.write(json.dumps(e) + '\n')

valid_comet=[]
with open('./valid.txt', 'r',encoding="utf-8") as f:
    reader = f.readlines()
with open('./valid.txt', 'r', encoding="utf-8") as f:
    for line in tqdm.tqdm(reader, total=len(reader)):
        data = json.loads(line)
        re_data=reprocess_data(data)
        valid_comet.append(re_data)
with open('./valid_new.txt', 'w') as f:
    for e in valid_comet:
        f.write(json.dumps(e) + '\n')

test_comet=[]
with open('./test.txt', 'r',encoding="utf-8") as f:
    reader = f.readlines()
with open('./test.txt', 'r', encoding="utf-8") as f:
    for line in tqdm.tqdm(reader, total=len(reader)):
        data = json.loads(line)
        re_data=reprocess_data(data)
        test_comet.append(re_data)
with open('./test_new.txt', 'w') as f:
    for e in test_comet:
        f.write(json.dumps(e) + '\n')