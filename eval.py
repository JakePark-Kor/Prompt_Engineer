""
 Copyright (c) 2024, 
 Gwangju Institute of Science and Technology,
 
 All rights reserved.
"""
#MoverScore
from __future__ import absolute_import, division, print_function
import string
from pyemd import emd, emd_with_flow
from torch import nn
from math import log
from itertools import chain
from multiprocessing import Pool
from functools import partial
from typing import List, Union, Iterable
from itertools import zip_longest
import sacrebleu
from langdetect import detect
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

# [Text Generation part by the model]
import torch
from utils.config import Config
from models.videochat import VideoChat
from conversation import Chat
from utils.easydict import EasyDict
from collections import Counter, defaultdict
import json
import pandas as pd
from glob import glob
import os
import re
import numpy as np

from transformers import (BertForSequenceClassification, 
                          BertTokenizer,
                          BertConfig,
                          XLNetLMHeadModel, 
                          XLNetTokenizer,
                          AutoTokenizer,
                          AutoModel,
                          RobertaModel, 
                          RobertaTokenizer)

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import wordnet as wn

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

# ========================================
#           Model Initialization
# ========================================

def init_model():
    
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device('cuda:2'))
    
    layer_params = defaultdict(int)
    
    for name, param in model.named_parameters():
        
        layer_name = name.split('.')[0] 
        layer_params[layer_name] += param.numel()

    for layer, num_params in layer_params.items():
        
        print(f"Layer: {layer} | Total Parameters: {num_params}")
   
    model = model.eval()
    chat = Chat(model)
    
    print('Initialization Finished')
    
    return chat
#%%
print('Initializing Evaluation.....')

# [Full Description Eval] - [50 %]

# ========================================
#       Model Evaluation [CHRF Score]
# ========================================

# ngram with world net
'''
def ngram(text, n=1):
    
    """Generate n-gram based on text"""
    
    return [text[i:i+n] for i in range(len(text)-n+1)]

'''
def syn_char_ngrams(word, n=1):
    """Generate character n-grams considering synonyms."""
    synonyms = {lemma.name().replace('_', ' ') for synset in wn.synsets(word) for lemma in synset.lemmas()}
    
    all_ngrams = set()
    
    for synonym in synonyms:
        chars = list(synonym)
        
        for i in range(len(chars) - n + 1):
            ngram = ''.join(chars[i:i+n])
            all_ngrams.add(ngram)
            
    return all_ngrams

def precision_recall_fscore(reference, 
                            translation, 
                            config,  
                            beta = 1.0):
    
    # Similar to usual CHRF, but consider synonyms in char n-gram generation
    
    ref_ngrams = Counter()
    for word in reference.split():
        ref_ngrams.update(syn_char_ngrams(word, n=config['n-gram']))

    trans_ngrams = Counter()
    
    for word in translation.split():
        trans_ngrams.update(syn_char_ngrams(word, n=config['n-gram']))
        
    '''
    ref_ngrams   = Counter(ngram(reference,   config['n-gram']))
    trans_ngrams = Counter(ngram(translation, config['n-gram']))
    '''
    
    common_ngrams = sum((ref_ngrams & trans_ngrams).values())
    total_ref_ngrams = sum(ref_ngrams.values())
    total_trans_ngrams = sum(trans_ngrams.values())

    # Percision
    
    precision = common_ngrams / total_trans_ngrams if total_trans_ngrams > 0 else 0
    
    # Recall
    
    recall = common_ngrams / total_ref_ngrams if total_ref_ngrams > 0 else 0
    
    # F-score 
    
    if precision + recall > 0:
        
        fscore = (1 + beta**2)*(precision * recall) / ((beta**2 * precision) + recall)
        
    else:
        
        fscore = 0
        
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-score: {fscore:.3f}")
        
    return precision, recall, fscore


'''

def chrf_mean(reference, translation, exponential =1.0):

    
    results = {}
    total_precision, total_recall, total_fscore = 0, 0, 0
    ngram_count = 4  

    for n in range(1, ngram_count + 1):  # n = 1, 2, 3, 4
        
        config = {'n-gram': n}
        precision, recall, fscore = precision_recall_fscore(reference, 
                                                            translation, 
                                                            config, 
                                                            exponential)
        
        results[n] = {
            'precision': precision,
            'recall': recall,
            'f-score': fscore
        }

        total_precision += precision
        total_recall += recall
        total_fscore += fscore


    avg_precision = total_precision / ngram_count
    avg_recall = total_recall / ngram_count
    avg_fscore = total_fscore / ngram_count

    print("\nAverage Results:")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F-score: {avg_fscore:.3f}")
    
    return avg_precision, avg_recall, avg_fscore


'''

def load_json_data(file_path):

    with open(file_path, 'r') as file:
        
        data = json.load(file)
        
    return data


def evaluate_chat_session(generated_answers, reference_answers, config):
    
    results = []
    generated_dialogues = generated_answers['Dialogues']
    reference_answers = {f'Q{i+4}': dialogue[f'A{i+4}'] for i, 
                         dialogue in enumerate(reference_answers['Dialogues'])}

    total_scores = {'Total Score': 0, 
                    'METEOR': 0,
                    'BELU-4': 0,
                    'Count': 0}
    
    for dialogue in generated_dialogues:
        
        for i in range(4,8):
            
            question_key = f'Q{i}'
            answer_key = f'A{i}'
        
        
            if question_key in dialogue and answer_key in dialogue and question_key in reference_answers:
            
                user_input = dialogue[question_key]
                response = dialogue[answer_key]
                reference = reference_answers.get(question_key, '')
                
                # METEOR Score
                meteor_score = calculate_meteor(reference, response)
                bleu_score = calculate_bleu_4(reference, response)
                
                weight = 0.70 if i == 4 else 0.1
                
                if i== 4:
                    
                    precision, recall, fscore = precision_recall_fscore(reference, response, 
                                                                        config, beta = 1.0)
                    appropriateness = evaluate_appropriateness(response, user_input)
                    #mover_score = moverscore(reference,response) #[ver.1]
                    #mover_score = corpus_score(response, reference)
                    mover_score = sentence_score(response, reference)
                    '''
                    idf_dict_ref = get_idf_dict(reference)
                    idf_dict_res = get_idf_dict(response)
                    mover_score = word_mover_score(reference, response, 
                                                   idf_dict_ref, idf_dict_res, 
                                                   stop_words=[], n_gram=1, 
                                                   remove_subwords=True)
                      
                    if isinstance(mover_score, list) and len(mover_score) > 0:
                    
                        mover_score = sum(mover_score) / len(mover_score)
                    else:
                        mover_score = 0    
                    '''                 

                    
                    print(f"Averaged mover_score: {mover_score}") 
                    
                    plot_file_name = f"plot_{i}_{os.path.splitext(generated_answers['Video_Name'])[0]}"
                    plot_example(True, reference, response, config['eval_path'], plot_file_name)
                    
                    fs_weight = 0.714 # ~= 50%
                    
                else:
                    
                    precision, recall, fscore = precision_recall_fscore(reference, response, config, beta = 1)
                    appropriateness = None
                    mover_score = None
                    fs_weight = 1.0
                    
                total_score = (fscore * fs_weight) * weight   
                
                if appropriateness is not None:
                
                    total_score += (appropriateness * 0.10)           
                
                if mover_score is not None:
                    
                    total_score += (mover_score * 0.10)
                  
                meteor_weighted = meteor_score  * weight
                bleu_weighted = bleu_score * weight
                
                total_scores['Total Score'] += total_score
                total_scores['METEOR'] += meteor_weighted
                total_scores['BELU-4'] += bleu_weighted
                total_scores['Count'] += 1
                
                results.append({
                    "Video_name":generated_answers["Video_Name"],
                    "Question": user_input,
                    "Generated_Ans": response,
                    "Reference_Ans": reference,
                    "Precision": precision,
                    "Recall": recall,
                    "F-score": fscore,
                    "Appropriateness": appropriateness,
                    "Mover Score": mover_score,
                    "Individual Score": total_score,
                    "F1 Total Score": "",
                    "METEOR": meteor_score,
                    "METEOR Weighted Score": meteor_weighted,
                    "METEOR Total Score": "",
                    "BLEU-4": bleu_score,
                    "BLEU-4 Weighted Score": bleu_weighted,
                    "BLEU-4 Total Score ": ""
                    
                })
                
    if total_scores['Count'] > 0:
        
        results.append({
            "Video_name": generated_answers["Video_Name"],
            "Question": "Total Score",
            "Generated_Ans": "",
            "Reference_Ans": "",
            "Precision": "",
            "Recall": "",
            "F-score": "",
            "Appropriateness": "",
            "Mover Score": "",
            "Individual Score": "",
            "F1 Total Score": total_scores['Total Score'],
            "METEOR": "",
            "METEOR Weighted Score": "",
            "METEOR Total Score": total_scores['METEOR'],
            "BLEU-4": "",
            "BLEU-4 Weighted Score": "",
            "BLEU-4 Total Score ": total_scores['BELU-4']                     
        })     
           
    return pd.DataFrame(results)

def save_results_to_excel(df, file_name):
    
    df.to_excel(file_name, index=False)
    
def process_all_json_files(chat_saving_path, answer_label_path, config):
    
    final_results = pd.DataFrame()
    chat_json_paths = glob(os.path.join(chat_saving_path, 'chat_session_*.json'))

    results_list = []
    
    for chat_json_path in chat_json_paths:
 
        chat_load = load_json_data(chat_json_path)
        match = re.search(r'chat_session_(\d+).json', os.path.basename(chat_json_path))

        #video_name = os.path.basename(chat_json_path).split('.')[0]
        if match:
            
            number = match.group(1)
            ans_json_path = os.path.join(answer_label_path, f'answer_sheet_{number}.json')
            
            if os.path.exists(ans_json_path):
                ans_load = load_json_data(ans_json_path)
                eval_results = evaluate_chat_session(chat_load, ans_load, config)
                results_list.append(eval_results)
                #final_results = pd.concat([final_results, eval_results], ignore_index=True)
                
    if results_list:
        
        #final_results = final_results.append(average_row, ignore_index=True)   
        final_results = pd.concat(results_list, ignore_index=True)
    
    save_results_to_excel(final_results, os.path.join(config['eval_path'], config['eval_file_name']))

# [3 Questions Eval] - [ 30 %]

# ========================================
#      Model Evaluation [Sentence Flow]
# ========================================

# [Appropriateness]  - [  10 %]

def evaluate_appropriateness(text, context):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=bert_config)
   
    inputs = tokenizer(text, context, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad(): 
        outputs = model(**inputs)

    logits = outputs[0]

    probabilities = torch.softmax(logits, dim=1)
    appropriateness_score = probabilities[:, 1].item()

    return appropriateness_score

# [Fluency] - [  0 %]

def evaluate_fluency_with_xlnet(text):
    
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
    
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, labels=inputs["input_ids"])
    
    loss = outputs.loss.item()
    
    fluency_score = loss
    
    if fluency_score >= 1:
        
        fluency_socre -= int(fluency_score)
        
    print("Fluency_score", fluency_score)
    
    return fluency_score

def calculate_meteor(reference, candidate):
    
    
    reference = reference.split()  
    candidate = candidate.split()  
    
    score = single_meteor_score(reference, candidate)
    
    return score

def calculate_bleu_4(reference, candidate):
    
    reference_tokenized = reference.split()  
    candidate_tokenized = candidate.split()  
    
    
    score = sentence_bleu([reference_tokenized], candidate_tokenized)
    smoothie = SmoothingFunction().method4
    weights = (0.25, 0.75, 0.0, 0.0)
    score = sentence_bleu([reference_tokenized], candidate_tokenized, smoothing_function=smoothie, weights=weights)
    
    return score

#[ver.1]
def moverscore(refs, hyps):
    
    
    model_name='roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
   
    """
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    """
    
    encoded_refs = tokenizer(refs, return_tensors='pt', padding=True, truncation=True)
    encoded_hyps = tokenizer(hyps, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        
        ref_embeddings = model(**encoded_refs).last_hidden_state.mean(dim=1)
        hyp_embeddings = model(**encoded_hyps).last_hidden_state.mean(dim=1)

    # Cosine Similarity 
    
    cos_sim = torch.nn.functional.cosine_similarity(ref_embeddings, hyp_embeddings)

    # Avg Cosine Similarity
    
    return cos_sim.mean().item()


#[ver.2]

nltk.download('punkt')

def is_english(text):
    
    try:
        return detect(text) == 'en'
    
    except:
        return False

def filter_english_sentences(paragraph):
    
    sentences = re.split(r'[.!?]', paragraph)
    english_sentences = [s.strip() for s in sentences if is_english(s)]
    return ' '.join(english_sentences)

def moverscore_paragraph_similarity(paragraph, reference_sentences, 
                                  stop_words=[], n_gram=1, remove_subwords=True):
    
    filtered_paragraph = filter_english_sentences(paragraph)
    filtered_reference = [filter_english_sentences(sent) for sent in reference_sentences]
    
    idf_dict_ref = get_idf_dict(filtered_reference)
    idf_dict_res = get_idf_dict([filtered_paragraph])
    
    mover_score = word_mover_score(filtered_reference, [filtered_paragraph], 
                                   idf_dict_ref, idf_dict_res, stop_words, n_gram, remove_subwords)

    #average_score = sum(mover_score) / len(mover_score) if mover_score else 0
    
    if isinstance(mover_score, list) and len(mover_score) > 0:
    
        mover_score = sum(mover_score) / len(mover_score)
    
    else:
    
        mover_score = 0

    return mover_score

#© 2019 Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger. 
#Published by the Association for Computational Linguistics.

def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, 
                              idf_dict_ref, idf_dict_hyp, stop_words=[], 
                              n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        
        print(hypothesis, references, sentence_score)
            
    return sentence_score


def corpus_score(sys_stream: List[str],
                     ref_streams:Union[str, List[Iterable[str]]], trace=0):

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]

    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    fhs = [sys_stream] + ref_streams

    corpus_score = 0
    
    for lines in zip_longest(*fhs):
        
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
            
        hypo, *refs = lines
        corpus_score += sentence_score(hypo, refs, trace=0)
        
    corpus_score /= len(sys_stream)

    return corpus_score

device = 'cuda'

if os.environ.get('MOVERSCORE_MODEL'):
    model_name = os.environ.get('MOVERSCORE_MODEL')
else:
    model_name = 'distilbert-base-uncased'
    
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
model.eval()
model.to(device)

def truncate(tokens):
    
    if len(tokens) > tokenizer.model_max_length - 2:
        tokens = tokens[0:(tokenizer.model_max_length - 2)]
    return tokens

def process(a):
    
    a = ["[CLS]"]+truncate(tokenizer.tokenize(a))+["[SEP]"]
    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, nthreads=4):
    
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict

def padding(arr, pad_token, dtype=torch.long):
    
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        result = model(x, attention_mask = attention_mask)
    if model_name == 'distilbert-base-uncased':
        return result[1] 
    else:
        return result[2] 

#with open('stopwords.txt', 'r', encoding='utf-8') as f:
#    stop_words = set(f.read().strip().split(' '))

def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]",device='cuda:0'):
    
    tokens = [["[CLS]"]+truncate(tokenize(a))+["[SEP]"] for a in arr]  
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    
    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)
    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)

    return padded, padded_idf, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1,device='cuda:0'):

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                      tokenizer.tokenize, 
                                                      tokenizer.convert_tokens_to_ids,
                                                      idf_dict,device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    
    with torch.no_grad():
        
        for i in range(0, len(all_sens), batch_size):
            
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    
    return total_embedding, lens, mask, padded_idf, tokens

def _safe_divide(numerator, denominator):
    
    return numerator / (denominator + 1e-30)

def batched_cdist_l2(x1, x2):
    
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], 
                     n_gram=1, remove_subwords = True, batch_size=256, 
                     device='cuda:0'):
    preds = []
    
    for batch_start in range(0, len(refs), batch_size):
        
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]
        
        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(batch_refs, 
                                                                                     model, 
                                                                                     tokenizer, 
                                                                                     idf_dict_ref,
                                                                                     device=device)
        
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, 
                                                                                     model,
                                                                                     tokenizer, 
                                                                                     idf_dict_hyp,
                                                                                     device=device)

        ref_embedding = ref_embedding[-1]
        hyp_embedding = hyp_embedding[-1]
        
        batch_size = len(ref_tokens)
        
        for i in range(batch_size):  
            
            ref_ids = [k for k, w in enumerate(ref_tokens[i]) 
                                if w in stop_words or '##' in w 
                                or w in set(string.punctuation)]
            
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) 
                                if w in stop_words or '##' in w
                                or w in set(string.punctuation)]
          
            ref_embedding[i, ref_ids,:] = 0                        
            hyp_embedding[i, hyp_ids,:] = 0
            
            ref_idf[i, ref_ids] = 0
            hyp_idf[i, hyp_ids] = 0
            
        raw = torch.cat([ref_embedding, hyp_embedding], 1)
                             
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
        
        distance_matrix = batched_cdist_l2(raw, raw).double().cpu().numpy()
                
        for i in range(batch_size):  
            
            c1 = np.zeros(raw.shape[1], dtype=np.float)
            c2 = np.zeros(raw.shape[1], dtype=np.float)
            c1[:len(ref_idf[i])] = ref_idf[i]
            c2[len(ref_idf[i]):] = hyp_idf[i]
            
            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            
            dst = distance_matrix[i]
            _, flow = emd_with_flow(c1, c2, dst)
            flow = np.array(flow, dtype=np.float32)
            score = 1./(1. + np.sum(flow * dst))#1 - np.sum(flow * dst)
            preds.append(score)

    return preds

def plot_example(is_flow, reference, translation, output_dir, file_name ,device='cuda:0'):
    
    idf_dict_ref = defaultdict(lambda: 1.) 
    idf_dict_hyp = defaultdict(lambda: 1.)
    
    ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding([reference], 
                                                                                 model, tokenizer, 
                                                                                 idf_dict_ref,device=device)
    hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding([translation], 
                                                                                 model, tokenizer, 
                                                                                 idf_dict_hyp,device=device)
   
    ref_embedding = ref_embedding[-1]
    hyp_embedding = hyp_embedding[-1]
               
    raw = torch.cat([ref_embedding, hyp_embedding], 1)            
    raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
    
    distance_matrix = batched_cdist_l2(raw, raw)
    masks = torch.cat([ref_masks, hyp_masks], 1)        
    masks = torch.einsum('bi,bj->bij', (masks, masks))
    distance_matrix = masks * distance_matrix              

    
    i = 0
    c1 = np.zeros(raw.shape[1], dtype=np.float)
    c2 = np.zeros(raw.shape[1], dtype=np.float)
    c1[:len(ref_idf[i])] = ref_idf[i]
    c2[len(ref_idf[i]):] = hyp_idf[i]
    
    c1 = _safe_divide(c1, np.sum(c1))
    c2 = _safe_divide(c2, np.sum(c2))
    
    dst = distance_matrix[i].double().cpu().numpy()

    if is_flow:   
             
        _, flow = emd_with_flow(c1, c2, dst)
        new_flow = np.array(flow, dtype=np.float32)    
        res = new_flow[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i])+len(hyp_tokens[i]))]
    else:    
        res = 1./(1. + dst[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i])+len(hyp_tokens[i]))]) 

    r_tokens = ref_tokens[i]
    h_tokens = hyp_tokens[i]
    
    fig, ax = plt.subplots(figsize=(len(r_tokens)*0.8, len(h_tokens)*0.8))
    im = ax.imshow(res, cmap='Blues')
    
    ax.set_xticks(np.arange(len(h_tokens)))
    ax.set_yticks(np.arange(len(r_tokens)))
  
    ax.set_xticklabels(h_tokens, fontsize=14)
    ax.set_yticklabels(r_tokens, fontsize=14)
    plt.xlabel("System Generation", fontsize=18)
    plt.ylabel("Human Reference", fontsize=18)
    plt.title("Flow Matrix", fontsize=18)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

#    for i in range(len(r_tokens)):
#        for j in range(len(h_tokens)):
#            text = ax.text(j, i, '{:.2f}'.format(res[i, j].item()),
#                           ha="center", va="center", color="k" if res[i, j].item() < 0.6 else "w")    
    fig.tight_layout()
    plt.show()

    output_path = os.path.join(output_dir, f"{file_name}.png")
    
    plt.savefig(output_path)
    
    plt.close(fig)


def main(config):   
    
    # Evaluation initialization
    
    process_all_json_files(
        chat_saving_path=config['chat_saving_path'],
        answer_label_path=config['answer_label_path'],
        config=config
    )    
    
    '''
    chat_json_path = f"{config['chat_saving_path']}/chat_session_3.json"
    chat_load = load_json_data(chat_json_path)
    
    ans_json_path = f"{config['answer_label_path']}/2.json"
    ans_load = load_json_data(ans_json_path)
    
    # [Full Description Eval] - [ 60 % ]
    # [3 Questions Eval]      - [ 30 % ]
    
    eval_results = evaluate_chat_session(chat_load, ans_load, config)
    
    # [Appropriateness]  - [ 5 % ]
    # [Fluency]          - [ 5 % ]
    
    save_results_to_excel(eval_results, f"{config['eval_path']}/evaluation_results_prompt2_accident.xlsx")
    '''
    
if __name__ == "__main__":
    
    config = {
        
        'data_path': "/home/workspace/data/accident_test/accident/",
        'chat_saving_path': '/home/workspace/Ask-Anything/video_chat/eval/hand_in_sheet/chat_history/prompt1/4/',
        'answer_label_path':'/home/workspace/Ask-Anything/video_chat/eval/answer_sheet/label/',
        'eval_path': '/home/workspace/Ask-Anything/video_chat/eval/eval_prompt1/4/',
        'eval_file_name':  'evaluation_results_prompt1_accident_45.xlsx',
        
        # Evaluation setup 
        'n-gram': 3
    }
    
    main(config)
