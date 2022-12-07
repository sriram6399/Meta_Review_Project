import csv
from transformers import pipeline
import utilities
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from evaluate import load

bertscore = load("bertscore")
model = SentenceTransformer('bert-base-nli-mean-tokens')
summarizer = pipeline("summarization",truncation=True)

def com_sentences():
    data = utilities.read_data("Data/data1.csv")
    out_list=[]
    for x in data['review']:
        sen_dict = dict()
        out = ""
        uncom = ""
        com =""
        corpus  = x.split(". ")
        sen_embeddings = model.encode(corpus)
        used = []
        for i in range(0,len(sen_embeddings)):
            if i in used:
                continue
            if i not in sen_dict.keys():
                sen_dict[i] =1
            for j in range(i+1,len(sen_embeddings)):
                if j in used:
                    continue
                if cosine_similarity([sen_embeddings[i]],[sen_embeddings[j]]) >0.75 :
                    sen_dict[i] = sen_dict[i] +1
                    used.append(j)
        
        for y in sen_dict.keys():
            if sen_dict[y] == 1:
                uncom = uncom + corpus[y] + ". "
            else:
                com = com + corpus[y]+ ". "
        
        '''uncom_words = uncom.split(" ")
        if(len(uncom_words)>1000):
            uncom=""
            for i in range(0,1000):
                uncom = uncom + uncom_words[i] +" "'''
        print("doing")
        out = out + com + summarizer(uncom)[0]['summary_text']
        
        out_list.append(out)   
    
    list=[]
    for i in range(0,len(out_list)):
        list.append([data['review'][i],data['meta_review'][i],out_list[i]])

    with open('Data/com_sentence_data.csv', 'w',newline='',encoding="utf-8") as file:
        write = csv.writer(file,delimiter=',')
        write.writerows(list)

def results():
    my_file = open('Data/com_sentence_data.csv', "r",encoding="utf-8")
    corpus = pd.read_csv(my_file,header=0,names=['review','meta_review','drafts'])
    my_file.close()
    srcs= [x for x in corpus['meta_review']]
    trgs = [y for y in corpus['drafts']]
    with open('Results/com_sentence_bertscore.txt', 'w',encoding="utf-8") as file_list:
        res =bertscore.compute(predictions=srcs, references=trgs, lang="en")
        for element in res["f1"] :
            file_list.write('%s\n' % element)

    with open('Results/com_sentence_bartscore.txt', 'w',encoding="utf-8") as file_list:
        for element in utilities.BARTScorer().score(srcs,trgs):
            file_list.write('%s\n' % element)

results()

#com_sentences()