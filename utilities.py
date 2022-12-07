import csv
from turtle import shape
import pandas as pd
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
import contractions
import re
from tensorflow.keras import callbacks, models, layers, preprocessing as kprocessing 
import nltk
from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset,Dataset,DatasetDict

def read_data(filename):
    my_file = open(filename, "r",encoding="utf-8")
    corpus = pd.read_csv(my_file,header=0,names=['review','meta_review'])
    my_file.close()
    return corpus

def read_extractive_data(filename):
    my_file = open(filename, "r",encoding="utf-8")
    corpus = pd.read_csv(my_file,header=0,names=['review','meta_review','ext_review'])
    my_file.close()
    return corpus

class BARTScorer():
    def __init__(self, device='cpu', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)


    def score(self,srcs, tgts, batch_size=4):
        print("test1")
        score_list = [] 
        for i in range(0, len(srcs), batch_size):
            print("test2")
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    print("test3")
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    print("test4")
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list
                    print("test5")
            except RuntimeError:
                print("test6")
                traceback.print_exc()
                print("test7")
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

def utils_preprocess_text(txt, punkt=True, lower=True, slang=True, stopwords=None, stemm=False, lemm=True):
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    if punkt is True:
        txt = re.sub(r'[^\w\s]', '', txt)
    else: txt
    txt = " ".join([word.strip() for word in txt.split()])
    if lower is True:
        txt = txt.lower() 
    else: txt
    if slang is True:
        txt = contractions.fix(txt) 
    else: txt   
    lst_txt = txt.split()
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]
    if stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in stopwords]
    txt = " ".join(lst_txt)
    return txt

def get_tokenizer(corpus):
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    ngram=[1]
    freq = pd.DataFrame()
    for n in ngram:
        words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        corpus_n = pd.DataFrame(words_freq.most_common(), columns=["word","freq"])
        corpus_n["ngrams"] = n
        freq = freq.append(corpus_n)
        freq["word"] = freq["word"].apply(lambda x: " ".join(string for string in x) )
        freq_X= freq.sort_values(["ngrams","freq"], ascending=[True,False])

    thres = 5 
    X_top_words = len(freq_X[freq_X["freq"]>thres])
    tokenizer = kprocessing.text.Tokenizer(num_words=X_top_words, lower=False, split=' ', oov_token=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(corpus)
    return tokenizer

def get_embedding_vector(X_dic_vocabulary):
    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
    X_embeddings = np.zeros((len(X_dic_vocabulary)+1, 300))
    for word,idx in X_dic_vocabulary.items():
        try:
            X_embeddings[idx] = model.encode(word)
        except:
            pass
    return X_embeddings

def get_avgs(name):
    file1 = open(name, 'r')
    Lines = file1.readlines()
    lst = []
    for line in Lines:
        lst.append(float(line))

    return sum(lst)/len(lst)

