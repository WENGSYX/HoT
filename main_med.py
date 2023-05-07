import argparse
import logging
import torch
import random
import time
from tqdm import tqdm
import os
from utils import *

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.nist_score import sentence_nist
import jieba
from nltk.util import ngrams

def log_data(text, path):
    with open(path + '/loggings.txt', 'a', encoding='utf-8') as f:
        f.write(text)
        print(text)
        f.write('\n')


def log_data_self(text, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')


def log_start(MODEL, DATA, N, K,FN):
    log_name = MODEL + "_" + DATA + "_" + str(N) + "_" + str(K) + "_" + str(FN)
    try:
        os.mkdir('medlog/' + log_name)
    except:
        log_name += time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('medlog/' + log_name)

    with open('medlog/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__), 'r', encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'medlog/' + log_name
    return path


def get_metrics(pred, target,args):
    turns = len(target)
    bleu_2 = 0
    bleu_4 = 0
    meteor = 0
    nist_2 = 0
    nist_4 = 0
    for index in range(turns):
        pred_utt = pred[index]
        target_utt = target[index]
        min_len = min(len(pred_utt), len(target_utt))
        lens = min(min_len, 4)
        if lens == 0:
            continue
        if lens >= 4:
            bleu_4_utt = sentence_bleu([target_utt], pred_utt, weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=SmoothingFunction().method1)
            nist_4_utt = sentence_nist([target_utt], pred_utt, 4)
        else:
            bleu_4_utt = 0
            nist_4_utt = 0
        if lens >= 2:
            bleu_2_utt = sentence_bleu([target_utt], pred_utt, weights=(0.5, 0.5),
                                       smoothing_function=SmoothingFunction().method1)
            nist_2_utt = sentence_nist([target_utt], pred_utt, 2)
        else:
            bleu_2_utt = 0
            nist_2_utt = 0

        bleu_2 += bleu_2_utt
        bleu_4 += bleu_4_utt
        if args.dataset in ('CMDD'):
            ref_tokens = jieba.cut(target_utt)
            cand_tokens = jieba.cut(pred_utt)
            meteor += meteor_score([ref_tokens], cand_tokens)
        meteor += meteor_score([set(target_utt.split(' '))], pred_utt.split(' '))
        nist_2 += nist_2_utt
        nist_4 += nist_4_utt

    bleu_2 /= turns
    bleu_4 /= turns
    meteor /= turns
    nist_2 /= turns
    nist_4 /= turns
    return bleu_2, bleu_4, meteor, nist_2, nist_4

def main():
    args = parse_arguments()
    path = log_start(args.model, args.dataset, args.method, 1,args.FN)
    log_data('*****************************', path)
    print(args)
    log_data('*****************************', path)
    fix_seed(args.random_seed)

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    log_data("setup data loader ...", path)
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo,demo2 = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == 'few_shot_fot':
        demo,demo2 = create_demo_text(args, cot_flag=True)
    else:
        pass

    total = 0
    correct_list = []
    answer_list = []
    tk = tqdm(dataloader)
    loggings = []
    for i, data in enumerate(tk):

        log_data('*************************', path)
        log_data("{}st data".format(i + 1), path)

        # Prepare question template ...
        x, y = data

        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        if args.method == "zero_shot":
            x = x[0]
            y = y[0].strip()
            x = x + " " + args.direct_answer_trigger_for_zeroshot
            pred = decoder.decode(args, x, max_length, i, 0, 1,'\n')
            if len(pred) == 0:
                continue
            pred = pred[0]
            answer_list.append(pred)
            correct_list.append(y)
            log_data_self(pred,path+'/pred.txt')
            log_data_self(y,path+'/target.txt')
            log_data('pred: '+pred,path)
            log_data('target: '+y,path)

        elif args.method == "zero_shot_cot":
            x = x[0]
            y = y[0].strip()
            log_data('{}'.format(x.replace('\n','')), path)
            x1 = x + args.direct_answer_trigger_for_zeroshotcot[0]

            pred1 = decoder.decode(args, x1, max_length, i, 0, 1,'\n')
            if len(pred1) == 0:
                continue

            for i in pred1:
                log_data('{}'.format(i),path)

            x2 = x + args.direct_answer_trigger_for_zeroshotcot[0]+pred1[0]+args.direct_answer_trigger_for_zeroshotcot[1]
            pred = decoder.decode(args, x2, max_length, i, 0, 1,'\n')
            if len(pred) == 0:
                continue
            pred = pred1[0]+' Therefore, '+pred[0].replace('"','')


            answer_list.append(pred)
            correct_list.append(y)
            log_data_self(pred,path+'/pred.txt')
            log_data_self(y,path+'/target.txt')
            log_data('pred: '+pred,path)
            log_data('target: '+y,path)

        elif args.method == "few_shot":
            x = x[0]
            y = y[0].strip()
            log_data('{}'.format(x.replace('\n','')), path)
            x = demo + x + args.direct_answer_trigger_for_fewshot

            pred = decoder.decode(args, x, max_length, i, 0, 1,'\n')
            if len(pred) == 0:
                continue
            pred = pred[0]
            answer_list.append(pred)
            correct_list.append(y)
            log_data_self(pred,path+'/pred.txt')
            log_data_self(y,path+'/target.txt')
            print('pred: '+pred)
            print('target: '+y)

        elif args.method == "hot":
            x = x[0]
            y = y[0].strip()
            log_data('{}'.format(x.replace('\n','')), path)
            x1 = [x + fot for fot in args.direct_answer_trigger_for_zeroshotfot[3]]

            pred1 = decoder.decode(args, x1, 30, i, 0, 1,'\n')


            if len(pred1) == 0:
                continue
            pred1 = [args.direct_answer_trigger_for_zeroshotfot[3][i].split('A: ')[1]+pred1[i] for i in range(len(pred1))]
            pred1 = '.'.join(pred1)
            log_data('focused thinking:{}'.format(pred1), path)

            x2 = x + args.direct_answer_trigger_for_zeroshotfot[0]

            pred2 = decoder.decode(args, x2, max_length, i, 0.5, 4, '\n')
            if len(pred2) == 0:
                continue
            for i in pred2:
                log_data('diffused thinking:{}'.format(i),path)


            x3  =  args.direct_answer_trigger_for_zeroshotfot[4] + pred1+ '\n'+args.direct_answer_trigger_for_zeroshotfot[1]+"\"{}\"\n\n".format(''.join(pred2)) + x + args.direct_answer_trigger_for_zeroshotfot[2]
            pred3 = decoder.decode(args, x3, max_length, i, 0, 1,'\n')
            if len(pred3) == 0:
                continue
            pred3 = pred3[0]
            answer_list.append(pred3)
            correct_list.append(y)
            log_data_self(pred3,path+'/pred.txt')
            log_data_self(y,path+'/target.txt')
            print('pred: '+pred3)
            print('target: '+y)

        else:
            raise ValueError("method is not properly defined ...")
    bleu_2, bleu_4, meteor, nist_2, nist_4 = get_metrics(answer_list,correct_list,args)
    log_data('bleu_2: {}\n bleu_4: {}\n meteor: {}\n nist_2: {}\n nist_4: {}'.format(bleu_2, bleu_4, meteor, nist_2, nist_4),path)

def parse_arguments():
    parser = argparse.ArgumentParser(description="HoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="CMDD",
        choices=["meddialog","COVID",'CMDD'], help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="codex", choices=["codex",'codex-001','ul2','GLM'],
        help="model used for decoding."
    )

    parser.add_argument(
        "--method", type=str, default="hot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot",'hot'], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=1000,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=350,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=0.9, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--FN", type=int, default=0, help="few-shot number"
    )
    parser.add_argument(
        "--GLM_API",type=str,default='http://xxx.xxx.xxx.xxx:5000/generate',help="GLM's API"
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "meddialog":
        args.dataset_path = "./dataset/MedDialog/english-test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
        args.direct_answer_trigger_for_zeroshotcot = ["\nDoctor: 'Let\"s think step by step. ","therefore"]
        args.direct_answer_trigger_for_zeroshot = "\nDoctor: "
        args.direct_answer_trigger_for_fewshot = "\nDoctor: "
        args.direct_answer_trigger_for_zeroshotfot = ["\nDoctor:",
                                                      'Doctor may think: ',
                                                      '\nDoctor: ',
                                                      ['\nQ: What is the Chief Complaint? \nA: Chief Complaint is',
                                                       '\nQ: What is the Past History?\nA: Past History is',
                                                       '\nQ: What is the assistant examination?\nA: assistant examination is',
                                                       '\nQ: What is the Diagnosi?\nA: Diagnosi is',
                                                       '\nQ: What is the Suggestion?\nA: Suggestion is'],
                                                      'Medical Record:']
    elif args.dataset == "COVID":
        args.dataset_path = "./dataset/COVID-Dialogue-master/COVID-Dialogue-Dataset-English.txt"
        args.direct_answer_trigger = "\nTherefore, the answer is"
        args.direct_answer_trigger_for_zeroshotcot = ["\nDoctor: 'Let\"s think step by step. ","therefore"]
        args.direct_answer_trigger_for_zeroshot = "\nDoctor: "
        args.direct_answer_trigger_for_fewshot = "\nDoctor: "
        args.direct_answer_trigger_for_zeroshotfot = ["\nDoctor：",
                                                      'Doctor may think： ',
                                                      '\nDoctor： ',
                                                      ['\nQ: What is the Chief Complaint? \nA: Chief Complaint is',
                                                       '\nQ: What is the Past History?\nA: Past History is',
                                                       '\nQ: What is the assistant examination?\nA: assistant examination is',
                                                       '\nQ: What is the Diagnosi?\nA: Diagnosi is',
                                                       '\nQ: What is the Suggestion?\nA: Suggestion is'],
                                                      'Medical Record:']
    elif args.dataset == "MedDG":
        args.dataset_path = "./dataset/MedDG/data/test.pk"
        args.direct_answer_trigger = "医生： "
        args.direct_answer_trigger_for_fewshot = "\n医生： "
        args.direct_answer_trigger_for_zeroshotcot = ["\n医生：让我一步步思考。  ","因此"]
        args.direct_answer_trigger_for_zeroshot_fot = "\" 作为医生，可以提供一些建议吗? \nA: "
        args.direct_answer_trigger_for_zeroshot = "\n医生： "
        args.direct_answer_trigger_for_zeroshotfot = ["\n医生：",
                                                      '医生心里可能会想： ',
                                                      '\n医生： ',
                                                      '\n医疗报告： ']

    elif args.dataset == "CMDD":
        args.dataset_path = "./dataset/CMDD"
        args.direct_answer_trigger = "医生： "
        args.direct_answer_trigger_for_fewshot = "\n医生： "
        args.direct_answer_trigger_for_zeroshotcot = ["\n医生：让我一步步思考。  ","因此"]
        args.direct_answer_trigger_for_zeroshot_fot = "\" 作为医生，可以提供一些建议吗? \nA: "
        args.direct_answer_trigger_for_zeroshot = "\n医生： "
        args.direct_answer_trigger_for_zeroshotfot = ["\n医生：",
                                                      '医生心里可能会想： ',
                                                      '\n医生： ',
                                                      ['\nQ: 主诉是什么？\nA: 主诉是',
                                                       '\nQ: 现病史是什么？\nA: 现病史是',
                                                       '\nQ: 辅助检查是什么？\nA: 辅助检查是',
                                                       '\nQ: 诊断是什么？\nA: 诊断是',
                                                       '\nQ: 医学建议是什么？\nA: 医学建议是'],
                                                      '患者病历：']


    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."


    if args.dataset in ("commonsensqa"):
        args.verifier_text = ' Judge whether this statement is normal (yes or no)'
    else:
        args.verifier_text = " What is the answer of 'X'?"
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")

    return args


if __name__ == "__main__":
    main()
