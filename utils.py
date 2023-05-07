from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import xml.etree.ElementTree as ET
from transformers import T5ForConditionalGeneration as T5, AutoTokenizer
import torch
import openai  # For GPT-3 API ...
import os
import multiprocessing
import json
import numpy as np
import random
import torch
# import torchtext
import re
import random
import time
import datetime
import pandas as pd
import requests
import json


# https://review-of-my-life.blogspot.com/2017/11/python-dict-shuffle.html
def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(args, input, max_length, i, k, n, rd,rd_TYPE,stop):
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    time.sleep(5)

    # https://beta.openai.com/account/api-keys

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # print(openai.api_key)

    # Specify engine ...
    # Instruct GPT3
    if args.model == "gpt3":
        engine = "text-ada-001"
    elif args.model == "gpt3-medium":
        engine = "text-babbage-001"
    elif args.model == "gpt3-large":
        engine = "text-curie-001"
    elif args.model == "gpt3-xl":
        engine = "text-davinci-002"
    elif args.model == "codex":
        engine = "code-davinci-002"
    elif args.model == "codex-001":
        engine = "code-davinci-001"
    else:
        raise ValueError("model is not properly defined ...")

    response = openai.Completion.create(
        engine=engine,
        prompt=input,
        max_tokens=max_length,
        temperature=k,
        stop=stop,
        n=n
    )

    return [text["text"] for text in response["choices"]]


class Decoder():
    def __init__(self, args):
        print_now()
        if args.model == 'UL2':
            self.model = T5.from_pretrained("google/ul2", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
            self.tokenizer = AutoTokenizer.from_pretrained("google/ul2")
            device_map = {
                6: [0, 1, 2, 3, 4],
                7: [5, 6, 7, 8, 9, 10, 11, 12, 13, ],
                8: [14, 15, 16, 17, 18, 19, 20, 21, 22, ],
                9: [23, 24, 25, 26, 27, 28, 29, 30, 31],
            }
            self.model.parallelize(device_map)
        else:
            self.rd = 0

    def decode(self, args, input, max_length, i, k, n,stop,is_turn_to_declarative=False,RT=1):
        try:
            if args.model in ("gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "codex", "codex-001"):
                response = decoder_for_gpt3(args, input, max_length, i, k, n, self.rd,RT,stop)
                if self.rd != 4:
                    self.rd += 1
                else:
                    self.rd = 1
            elif args.model in ("UL2"):
                if is_turn_to_declarative:
                    input_ids = self.tokenizer('[S2S] ' + input[81:] + '\" <extra_id_0> \"', return_tensors="pt").input_ids.to(6)
                else:
                    input_ids = self.tokenizer('[S2S] ' + input + '<extra_id_0>', return_tensors="pt").input_ids.to(6)
                output = self.tokenizer.batch_decode(
                    self.model.generate(input_ids, temperature=k, do_sample=True, top_k=10, num_return_sequences=n,
                                        max_length=args.max_length_cot))
                response = [i.replace('<pad> ', '').replace('<pad>', '').replace('</s>', '').split('<extra')[0] for i in output]
            elif args.model in ('GLM'):
                url = args.GLM_API
                if type(input) == list:
                    input = [i.replace('\n','<n>') for i in input]
                else:
                    input = input.replace('\n','<n>')
                    input = [input] * n
                config = {
                    'prompt': input,
                    'seed': '1',
                    'max_tokens': max_length,
                    'sampling_strategy': 'BaseStrategy',
                    'num_beams': n,
                    'temprature': k,
                    'top_k':40,
                    'deterministic': True,}
                r = requests.post(url, data=json.dumps(config))
                response = [i[0].split('\n')[0] for i in json.loads(r.text)['text']]
        except:
            response = []

        return response


def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset == "commonsensqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("bigbench_date", "object_tracking"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            if args.dataset == "bigbench_date":
                choice_index = ['A', 'B', 'C', 'D', 'E', 'F']
            elif args.dataset in ("object_tracking"):
                choice_index = ['A', 'B', 'C']
            else:
                raise ValueError("dataset is not properly defined ...")
            for line in json_data:
                q = line["input"].strip()
                if args.dataset == "bigbench_date":
                    choice = "Answer Choices:"
                    # Randomly shuffle the answer choice dictionary because the original answer is always A ...
                    choice_dic = shuffleDict(line["target_scores"])
                elif args.dataset == "object_tracking":
                    choice = "\nWhich choice is true ? Answer Choices:"
                    choice_dic = line["target_scores"]
                else:
                    raise ValueError("dataset is not properly defined ...")
                for i, key_value in enumerate(choice_dic.items()):
                    key, value = key_value
                    choice += " ("
                    choice += choice_index[i]
                    choice += ") "
                    choice += key
                    if value == 1:
                        a = key
                        # a = key
                q = q
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("meddialog"):
        with open(args.dataset_path,encoding='utf-8') as f:
            json_data = json.load(f)
            for line in json_data:
                q = 'Patient Description: '+ line["description"] + '\nPatient: ' + ':'.join(line['utterances'][-2].split(':')[1:])
                a = ':'.join(line['utterances'][-1].split(':')[1:])
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("COVID"):
        with open(args.dataset_path,encoding='utf-8') as f:
            file = f.read().split('\n')
            for line in range(len(file)):
                if file[line][:3] != 'id=':
                    if file[line] == 'Description':
                        q = 'Patient Description: '+ file[line+1]
                    if file[line] == 'Dialogue':
                        for ps in range(1,len(file)-line):
                            if file[line+ps]=='Patient:':
                                q += 'Patient: '
                            elif file[line+ps]=='Doctor:':
                                questions.append(q)
                                ans = ''
                                for ps2 in range(1,len(file)-line-ps):
                                    if file[line+ps+ps2] == '' or file[line+ps+ps2] == 'Patient:':
                                        break
                                    ans += file[line+ps+ps2]
                                answers.append(ans)
                                q += 'Doctor: '
                            elif file[line+ps]=='' or file[line+ps][:2]=='id':
                                break
                            else:
                                q += file[line+ps]

    elif args.dataset in ("MedDG"):
        data = pd.read_pickle(args.dataset_path)
        for i in range(10):
            qs = ''
            for his in data[i]:
                if his['id'] == 'Patients':
                    qs += '患者: '
                    qs += his['Sentence']
                    qs += '\n'
                else:
                    questions.append(qs)
                    answers.append(his['Sentence'])
                    qs += '医生: '
                    qs += his['Sentence']
                    qs += '\n'
    elif args.dataset in ("CMDD"):
        datas = []
        for file in os.listdir(args.dataset_path):
            for f_item in os.listdir(args.dataset_path+'/'+file):
                if '.csv' in f_item:
                    data = pd.read_csv(args.dataset_path+'/'+file+'/'+f_item,encoding='ANSI')
                    for i in range(100):
                        questions.append('患者描述：{}\n 患者: {}'.format(data.iloc[i]['ask'],data.iloc[i]['title']))
                        answers.append(data.iloc[i]['answer'])

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output


def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True)

    return dataloader


# ver 0.2
def answer_cleansing(args, pred):
    if args.method in ("few_shot", "few_shot_cot",'verifier', 'verifier_cot','verifier_TF_cot', "zero_shot_verifier_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = [pred[1:-1]]
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot", "verifier","verifier_cot",'verifier_TF_cot',"zero_shot_verifier_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


def answer_cleansing_verifier(args, pred):
    if args.method in ("few_shot", "few_shot_cot",'verifier', 'verifier_cot','verifier_TF_cot', "zero_shot_verifier_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ("aqua"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq", "bigbench_date"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip", "commonsensqa"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot",'verifier', "verifier_cot",'verifier_TF_cot', "zero_shot_verifier_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


def create_demo_text(args, cot_flag):
    x, z, y, c = [], [], [], []

    # example sentences ...    
    if args.dataset in ("multiarith", "gsm8k", "addsub", "svamp", "singleeq"):

        x.append(
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append(
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")

        x.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append(
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")

        x.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")

        x.append(
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")

        x.append(
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append(
            "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")

        x.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append(
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")

    elif args.dataset in ("aqua"):
        x.append(
            "John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? ")
        c.append(
            "Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64")
        z.append(
            "If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.")
        y.append("A")

        x.append("If a / b = 3/4 and 8a + 5b = 22, then find the value of a. ")
        c.append("Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2")
        z.append(
            "If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.")
        y.append("B")

        x.append(
            "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? ")
        c.append(
            "Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km")
        z.append("The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km.")
        y.append("E")

        x.append(
            "How many keystrokes are needed to type the numbers from 1 to 500? ")
        c.append("Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788")
        z.append(
            "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.")
        y.append("B")

    elif args.dataset in ("commonsensqa"):
        x.append(
            "What do people use to absorb extra ink from a fountain pen? ")
        c.append("Answer Choices: (A) shirt pocket (B) calligrapher's hand (C) inkwell (D) desk drawer (E) blotter")
        z.append(
            "The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink.")
        y.append("E")

        x.append(
            "What home entertainment equipment requires cable? ")
        c.append("Answer Choices: (A) radio shack (B) substation (C) television (D) cabinet")
        z.append(
            "The answer must require cable. Of the above choices, only television requires cable.")
        y.append("C")

        x.append(
            "The fox walked from the city into the forest, what was it looking for? ")
        c.append("Answer Choices: (A) pretty flowers (B) hen house (C) natural habitat (D) storybook")
        z.append(
            "The answer must be something in the forest. Of the above choices, only natural habitat is in the forest.")
        y.append("C")

        x.append(
            "Sammy wanted to go to where the people were. Where might he go? ")
        c.append("Answer Choices: (A) populated areas (B) race track (C) desert (D) apartment (E) roadblock")
        z.append(
            "The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people.")
        y.append("A")

        x.append(
            "Where do you put your grapes just before checking out? ")
        c.append("Answer Choices: (A) mouth (B) grocery cart (C) super market (D) fruit basket (E) fruit market")
        z.append(
            "The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items.")
        y.append("B")

        x.append(
            "Google Maps and other highway and street GPS services have replaced what? ")
        c.append("Answer Choices: (A) united states (B) mexico (C) countryside (D) atlas")
        z.append(
            "The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions.")
        y.append("D")

        x.append(
            "Before getting a divorce, what did the wife feel who was doing all the work? ")
        c.append("Answer Choices: (A) harder (B) anguish (C) bitterness (D) tears (E) sadness")
        z.append(
            "The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness.")
        y.append("C")

    elif args.dataset in ("bigbench_date"):
        x.append(
            "2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?")
        z.append(
            "If 2015 is coming in 36 hours, then it is coming in 2 days. 2 days before 01/01/2015 is 12/30/2014, so today is 12/30/2014. So one week from today will be 01/05/2015.")
        y.append("01/05/2015")

        x.append(
            "The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date today in MM/DD/YYYY?")
        z.append(
            "If the first day of 2019 was Tuesday, then 01/01/2019 was a Tuesday. Today is the first monday, would be six days later. So today is 01/07/2019.")
        y.append("01/07/2019")

        x.append(
            "The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY?")
        z.append(
            "One day after 06/01/1943 is 06/02/1943, so today is 06/02/1943. 10 days before today is 05/23/1943.")
        y.append("05/23/1943")

        x.append(
            "It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?")
        z.append(
            "Today is 04/19/1969. 24 hours later is one day after today, which would be 04/20/1969.")
        y.append("04/20/1969")

        x.append(
            "Jane thought today is 3/11/2002, but today is in fact Mar 12, which is 1 day later. What is the date 24 hours later in MM/DD/YYYY?")
        z.append(
            "Today is 03/12/2002. So the date 24 hours later will be 03/13/2002")
        y.append("03/13/2002")

        x.append(
            "Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the date yesterday in MM/DD/YYYY?")
        z.append(
            "The last day of February is the 28th, so Jane was born on 02/28/2001. Today is her 16-year old birthday, so today is 02/28/2017. So yesterday was 02/27/2017. ")
        y.append("02/27/2017")

    elif args.dataset in ("meddialog",'COVID'):
        x.append(
            "Patient: can the meat i eat give me coronavirus? will cooking kill the coronavirus?")
        z.append(
            """Chief Complaint: Concerns about potential transmission of coronavirus through food and the effectiveness of cooking in killing the virus. Current Medical History: None relevant. Auxiliary Examination: None performed. Past History: None relevant. Diagnosis: None, as this is not a medical concern. Suggestion: Cooking adequately (e.g. 30 seconds in a microwave) will kill the coronavirus. It is unlikely that meat would transmit the virus, as it is not a common mode of transmission. It is recommended to follow general food safety guidelines and properly handle and cook meat to avoid any potential foodborne illness.""")
        y.append("cooking will kill. cooking adequately will kill the virus-30 sec of ,microwave")

        x.append(
            "Patient: my 2year old son has had a dry cough and a runny nose since thursday . he's not talking yet so it is difficult to tell if he has any trouble breathing so i'm concerned. he does not have temperature ?")
        z.append(
            """Chief Complaint: Dry cough and runny nose since Thursday. Current Medical History: 2 year old son, not yet talking. Auxiliary Examination: No observed difficulty breathing, no fever. Past History: None relevant. Diagnosis: Common infection. Suggestion: Monitor for refusal to play or eat, and for difficulty breathing. Seek medical attention if symptoms worsen.""")
        y.append("in brief: observatn is enough. by looking at your child you'll know if he is having breathing difficulties. cough with a runny nose is common in children. you should be concerned only if he is refusing to play and eat, or if he develops breathing difficulties. having a fever doesn't mean he has the coronavirus. he is more likely to be infected with any of the common infections in your area of gauteng.")

        x.append(
                "Patient: hello doctor, i am a 23-year-old man. i have anxiety and depression but no immunodeficiency disorders or chronic diseases. first, i wanna know if my immune system is weakened and how likely am i to die of coronavirus. second, i have itchiness in my throat and shortness of breath. i always have it because of anxiety but now it is more persistent than before. i also feel a very subtle feeling of pain, burning sensation and itchiness in my chest. i did not go out during the past ten days and have not been in contact with someone with positive covid. and i quit smoking past week. down to one or zero cigarettes from 20 a day. why am i feeling itchiness and pain and burning sensation? how likely am i to have covid-19? and how likely am i to die from it?")
        z.append(
            """Chief Complaint: Itchiness in throat and shortness of breath Current Medical History: 23-year-old male with anxiety and depression but no immunodeficiency disorders or chronic diseases. Quit smoking one week ago, has not been in contact with anyone with positive COVID-19. Auxiliary Examination: No further examinations performed. Past History: Anxiety and depression, no previous chronic diseases or immunodeficiency disorders. Diagnosis: Anxiety manifesting as physical symptoms of itchiness in throat and shortness of breath. Suggestion: Continue taking medications for anxiety, consider visiting a psychiatrist for further evaluation and treatment. Engage in deep breathing exercises and progressive muscle relaxation. Consider gargling with salt water to alleviate throat irritation.""")
        y.append("hello. anxiety can manifest itself in physical or psychological symptoms or both. the irritation sensation you are experiencing is a part of your anxiety. also please do not believe the hype about covid. it has a low mortality rate, of 2-3 percent, that too mortality is very high in people above 60-70 years with other co-morbidities. you do not have any such disorders and you have not even been in contact with anyone, so do not worry. please continue to take your medicines for anxiety if you are taking them, if not, please consider visiting a psychiatrist and get started on some low dose ssri type of medicines. also, if needed a low dose benzodiazepines can be added temporarily. also, please do some deep breathing exercises, or progressive muscle relaxation. you can also take some honey with water to reduce the itching in the throat and also try doing some gargles with lukewarm salt water.")

        x.append(
            "Patient: i believe the incubation period for covid 19 is 7 days. so if you get sick after 7 days does the 14 say quarantine period starts from the moment you show symptoms or is the 7 days incubation period included in 14 day quarantine period? ")
        z.append(
            """Chief Complaint: Patient is concerned about the incubation period for COVID-19 and the quarantine guidelines. Current Medical History: Patient is currently asymptomatic and has not been exposed to anyone with COVID-19. Auxiliary Examination: No testing or additional exams have been performed at this time. Past History: No relevant past medical history is mentioned. Diagnosis: None, as patient is currently asymptomatic and has not been exposed to anyone with COVID-19. Suggestion: Patient should follow the guidelines set forth by the National Institute for Communicable Diseases (NICD) for clinical management of suspected or confirmed COVID-19 disease. This includes a 14-day quarantine period after symptoms appear, or 14 days after achieving clinical stability for those with severe disease.""")
        y.append("after symptoms show. as per the nicd guidelines for clinical management of suspected or confirmed covid-19 disease, \"those with mild disease may be de-isolated 14 days after symptom onset, while those with severe disease may be de-isolated 14 days after achieving clinical stability (e.g. once supplemental oxygen is discontinued)")

    elif args.dataset in ("CMDD"):

        x.append(
            "患者描述：几天前左下腹剧痛，查出左输尿管结石。\n患者：尿结石多长时间能排出来")
        z.append(
            "主诉：左下腹剧痛。当前病史：几天前左下腹出现剧痛，查出左输尿管结石。辅助检查：CT。既往病史：无。诊断：左输尿管结石。建议：坚持治疗，注意饮食，不要食用刺激食物，合理饮食，必要时定期复诊。")
        y.append("结石的排出是没有具体的时间的，个人的情况不同排出的时间也会有差别。指导意见：你的情况是需要坚持治疗的，时间长没有排出可能由于结石比较大的缘故。，相信大家都知道泌尿系结石对男人的伤害比较大，因此要尽快接受治疗，生活中要注意饮食问题，不要食用刺激食物，合理饮食。而且必要时可以定期复诊。")


        x.append(
            "患者描述：腰椎椎体排列整齐，曲度变直，各椎体形态、信号未见异常；各椎间盘于T2WI信号未见减低，腰4-5椎间盘\n患者：腰椎椎体排列整齐，曲度变直，各椎体形态、")
        z.append(
            "主诉：腰间盘突出。当前病史：椎体排列整齐，曲度变直。辅助检查：无需。既往病史：无。诊断：腰间盘突出，需要康复治疗。建议：腰椎间盘突出倒走，如果感觉症状有减轻，可以考虑使用负跟鞋。")
        y.append("你好，对于腰椎间盘突出治疗方法有两种，康复治疗和手术治疗，康复治疗也称保守疗法，手术治疗后，仍然需要进行康复治疗，以巩固疗效。指导意见：建议腰椎间盘突出倒走是目前最有效的方法如果感觉症状有减轻，可以考虑使用负跟鞋，鞋底是前高后低的。")


        x.append(
            "患者描述：致使小儿癫痫病的因素有哪些呢?小孩子怎么会得癫痫病呢，有时候经常睡午觉的时候半夜里惊醒，就会哭啼，不知晓怎么办 在乎怎样的帮助：致使小儿癫痫病的因素有哪些? \n患者：引起孩子癫痫病的因素有哪些")
        z.append(
            "主诉：当前睡午觉时半夜惊醒哭啼，是否是癫痫。当前病史：小孩癫痫。辅助检查：无需。既往病史：无。诊断：生长发育中受到病毒感染或隔代遗传。建议：癫痫病可能由于闹危害或脑损伤等找出，或由于胚胎发育不良；除此之外也有可能是隔代遗传，在许多有癫痫病史或有先天性中枢神经系统或心脏变形的患者宗族中简单呈现出癫痫。")
        y.append("一、脑危害与脑损伤在胚胎生长发育中受到病毒感染放射线映照，或其它缘由致使的胚胎发育不良能致使癫痫，胎儿生产过程中产伤也是致使癫痫的个主要缘由，颅脑外伤也可致使癫痫。二、隔代遗传要素是构成小儿癫痫的缘由地点，在许多有癫痫病史或有先天性中枢神经系统或心脏变形的患者宗族中简单呈现出癫痫。")

        x.append(
            "患者描述：37了，之前查出脑发育不良，从小时候到现在身体状况良好，无抽搐，智力无影响，请问这疾病是保持现状还是会继续发展，对寿命有没有影响。\n患者：脑发育不良会对寿命有影响吗")
        z.append(
            "主诉：脑发育不良对寿命的影响。当前病史：脑发育不良。辅助检查：脑CT。既往病史：脑发育不良。诊断：脑发育不良，需尽早治疗。建议：积极配合治疗，保持乐观心态。")
        y.append("您好 现在治疗不晚的 一般3岁之内是最佳治疗时机越早越好的建议积极治疗 康复治疗配合营养脑细胞治疗至于时间不好说 因为不知孩子的情况咋样 积极配合治疗 ，脑发育这种疾病不易痊愈。患者朋友应当保持良好的心态，用积极的心态去面对它，只有这样才能提高患者对抗脑发育的信心，相信这样一定能得到康复。")
    else:
        raise ValueError("dataset is not properly defined ...")

    # randomize order of the examples ...
    index_list = list(range(len(x)))
    #random.shuffle(index_list)
    if args.FN != 0:
        index_list = index_list[:args.FN]

    # Concatenate demonstration examples ...
    demo_text = ""
    demo_text2 = ""
    for i in index_list:
        if len(c) > 0:
            if cot_flag:
                demo_text += "Q: " + x[i] + c[i] + "\nA: " + z[i] + " " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += "Q: " + x[i] + c[i] + "\nA: " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        if args.dataset in ('meddialog'):
            if cot_flag:
                demo_text += x[i] + "\nRecord Report: " + z[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + ".\n\n"
                demo_text2 += "Record Report: " + z[i] + "\n" + x[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += x[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        elif args.dataset in ('CMDD'):
            if cot_flag:
                demo_text += x[i] + "\n医疗报告： " + z[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + ".\n\n"
                demo_text2 += "医疗报告： " + z[i] + "\n" + x[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += x[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            if cot_flag:
                demo_text += "Q: " + x[i] + "\nA: " + z[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += "Q: " + x[i] + "\nA: " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text,demo_text2


def question_turn_decalrative(args, text, answer, answers_0, function, declarative):
    global new_question
    if 'Answer Choices' in text:
        text = text.split('Answer Choices')[0]
    try:
        if args.dataset in ("commonsensqa"):
            text = text.replace(',', '.')
            position_fullstop = text[::-1].find('.')

            question = text[len(text) - position_fullstop:]
            ts = text[:len(text) - position_fullstop]

            if ts[0] == ' ':
                ts = ts[1:]
            if ts[-1] != ' ':
                ts += ' '
            ts = ts.replace(' .', '.')
            if args.model == 'UL2':
                return ts, 'yes', "'{} The answer is {}' If the question and answer are changed into fluent declarative sentences: ".format(
                    question, answer)
            else:
                return ts, 'yes', "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                    question, answer)

        text = text.replace(',', '.')
        position_fullstop = text[::-1].find('.')

        question = text[len(text) - position_fullstop:]
        ts = text[:len(text) - position_fullstop]
        if args.dataset in ('bigbench_date'):
            declarative = question[17:-15] + ' is ' + answer + '.'
        else:
            if declarative == '':
                try:
                    declarative = function(args,
                                           "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                                               question, answer), args.max_length_cot, 0, 0, 1,'\n',is_turn_to_declarative=True)[0]
                except:

                    declarative = function(args,
                                               "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                                                   question, answer), args.max_length_cot, 0, 0, 1,'\n',is_turn_to_declarative=True)[0]

            else:
                if answers_0 in declarative:
                    declarative = declarative[:len(declarative) - declarative[::-1].find(answers_0[::-1]) - len(
                        answers_0)] + answer + declarative[len(declarative) - declarative[::-1].find(answers_0[::-1]):]
                else:
                    try:
                        declarative = function(args,
                                               "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                                                   question, answer), args.max_length_cot, 0, 0, 1,'\n',is_turn_to_declarative=True)[0]
                    except:
                        declarative = "{} The answer is {}.".format(question, answer)

        new_question_number = [s for s in re.findall(r'-?\d+\.?\d*', ts)]

        sentences, ans = [], []
        for nqn in range(len(new_question_number)):
            new_ts = ''
            number_find = False
            for i in ts.split('.'):
                if new_question_number[nqn] in i and number_find == False:
                    new_question = [p for p in i]
                    new_question[
                    i.find(new_question_number[nqn]):i.find(new_question_number[nqn]) + len(new_question_number[nqn])] = "'X'"
                    new_question = ''.join(new_question) + '.'
                    new_question.replace(' .', '.')
                    new_ts += new_question
                else:
                    new_ts += i + '.'
            new_ts = new_ts.replace('..', '.')

            if new_ts[0] == ' ':
                new_ts = new_ts[1:]
            if new_ts[-1] != ' ':
                new_ts += ' '
            new_ts = new_ts.replace(' .', '.')

            sentences.append('"' + new_ts + declarative + '"' + args.verifier_text)
            ans.append(new_question_number[nqn])
        return sentences[:3], ans[:3],declarative

    except:

        return '', '', ''
