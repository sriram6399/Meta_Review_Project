import json
from multiprocessing.spawn import prepare
import pandas as pd
import os
import csv

from utilities import utils_preprocess_text

def data_prepare(dir):
    corpus = []
    ver = ['ICLR 2017 conference AnonReviewer1','ICLR 2017 conference AnonReviewer3','ICLR 2017 conference AnonReviewer2']
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        with open(f) as file:
            data = json.load(file)
        cl = list()
        rev =''
        meta_rev=''
        for d in data['reviews']:
            if d['IS_META_REVIEW'] == True: 
                if len(d['comments']) >0 and len(meta_rev)==0:
                    meta_rev = meta_rev + d['comments']
                continue
            if d['TITLE'] not in cl:
                if len(d['comments']) >0 :
                    if d['OTHER_KEYS'] not in ver:
                        continue
                    cl.append(d['TITLE'])
                    rev = rev + d['comments'] +' '

        rev = rev[:-1]
        cl.clear()
        if meta_rev == '':
            continue
        corpus.append([rev,meta_rev])

    print(corpus[0][0])
    print("\n hi \n")
    print(corpus[0][1])
    return corpus


def save_csv(corpus,file_name):
    file = open(file_name, 'w', encoding="utf-8", newline='')
    with file:
        writer = csv.writer(file)
        for row in corpus:
            writer.writerow(row)


corpus = data_prepare('Data/iclr_2017/test/reviews')
save_csv(corpus,'Data/test_data.csv')

def file_maker(path):
    x1=0
    y=0
    l=0
    files = os.listdir(path)
    for file in files:
        temp =[]
        if os.path.isfile(os.path.join(path, file)):
            with open(os.path.join(path, file),'r',encoding='utf-8') as f:
                reader_obj = csv.reader(f)
                for x in reader_obj:
                    s = x[0]+" "+x[1]+" "+x[2]
                    s = utils_preprocess_text(s, False, False, False, None, False, False)
                    x1 = x1 + len(s.split(" "))
                    y= y+ len(utils_preprocess_text(x[3], False, False, False, None, False, False).split(" "))
                    temp.append([s,utils_preprocess_text(x[3], False, False, False, None, False, False)])
                    l=l+1
            f.close()

        with open('Data/data1.csv','a+',newline='',encoding='utf-8') as file:
            write = csv.writer(file,delimiter=',')
            write.writerows(temp)

    print("X:",(x1/l) )
    print("Y:",(y/l) )
    print("LEN:",l)

file_maker('Data/ICLR/')