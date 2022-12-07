import csv
import pandas as pd
import sem_f1
from evaluate import load


bertscore = load("bertscore")
rouge = load('rouge')

def read_data(filename):
    my_file = open(filename, "r",encoding="utf-8")
    corpus = pd.read_csv(my_file,header=0,names=['gen_review','ori_review','bert_score'])
    my_file.close()
    return corpus

def read_extractive_data(filename):
    my_file = open(filename, "r",encoding="utf-8")
    corpus = pd.read_csv(my_file,header=0,names=['review','ori_review','ext_review'])
    my_file.close()
    return corpus

def compute_bert_semf1(filename):
    data  = read_extractive_data(filename)
    res_f1 = sem_f1.get_f1(data['ext_review'],data['ori_review'])
    bert_f1 = bertscore.compute(predictions=data['ext_review'], references=data['ori_review'], lang="en")
    rouge_res = rouge.compute(predictions=data['ext_review'], references=data['ori_review'])
    print(bert_f1)
    with open('Results/ext_com_sentence.csv', 'w',newline='',encoding="utf-8") as file_list:
        write = csv.writer(file_list,delimiter=',')
        temp = []
        f1_tot=0
        bert_tot = 0
        for i in range(0,len(data['review'])):
            temp.append([data['review'][i],data['ori_review'][i],data['ext_review'][i],bert_f1['f1'][i],res_f1[i]])
            f1_tot = f1_tot + res_f1[i] 
            bert_tot = bert_tot + bert_f1['f1'][i]
        
        write.writerows(temp)
         
    with open('Results/ext_bert_semf1_stats.txt','a+',newline='',encoding="utf-8") as file_list2:
        file_list2.write("\n Extractive Com Sentence: \n")
        file_list2.write(str({"sem_f1_score":(f1_tot/len(data['ori_review'])), "bert_f1_score":(bert_tot/len(data['ori_review'])),"rouge":rouge_res}))




def compute_sem_f1(filename):
    data  = read_data(filename)
    res_f1 = sem_f1.get_f1(data['gen_review'],data['ori_review'])
    with open('Results/t5_f_outputs1.csv', 'w',newline='',encoding="utf-8") as file_list:
        write = csv.writer(file_list,delimiter=',')
        temp = []
        f1_tot=0
        for i in range(0,len(data['gen_review'])):
            temp.append([data['ori_review'][i],data['gen_review'][i],data['bert_score'][i],res_f1[i]])
            f1_tot = f1_tot + res_f1[i] 
        
        write.writerows(temp)
         
    with open('Results/sem_f1_stats.txt','a+',newline='',encoding="utf-8") as file_list2:
        file_list2.write("\n T5 With fine tuning1: \n")
        file_list2.write(str({"sem_f1_score":(f1_tot/len(data['gen_review']))}))

compute_bert_semf1("Data/com_sentence_data.csv")
