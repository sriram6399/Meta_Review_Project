from datasets import load_dataset  
import pandas as pd  
import numpy  
import matplotlib.pyplot as plt  
import re
import nltk  
import contractions  
import gensim 
import utilities

def textrank(corpus, ratio=0.2):    
    lst_summaries = [gensim.summarization.summarize(txt,  
                     ratio=ratio) for txt in corpus]    
    return lst_summaries


corpus = utilities.read_data('Data/test_data.csv')
preds = textrank(corpus['review'],ratio = 0.7)
score = utilities.BARTScorer.score(preds,corpus['meta_review'])
