import csv
import math
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import utilities
from evaluate import load

bertscore = load("bertscore")

embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def cluster():
    data = utilities.read_data("Data/data1.csv")
    out_list=[]
    for x in data['review']:
        out = ""
        corpus  = x.split(".")
        corpus_embeddings = embedder.encode(corpus)

        num_clusters = math.floor(len(corpus)/3)
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        for i,cluster in enumerate(clustered_sentences):
            out = out+ str(cluster[0])+"."
        
        out_list.append(out)
        
    print(len(data['review']))
    list=[]
    for i in range(0,len(data['review'])):
        list.append([data['review'][i],data['meta_review'][i],out_list[i]])

    with open('Data/sentence_cluster_data.csv', 'w',newline='',encoding="utf-8") as file:
        write = csv.writer(file,delimiter=',')
        write.writerows(list)
            
def results():
    my_file = open('Data/sentence_cluster_data.csv', "r",encoding="utf-8")
    corpus = pd.read_csv(my_file,header=0,names=['review','meta_review','drafts'])
    my_file.close()
    srcs= [x for x in corpus['meta_review']]
    trgs = [y for y in corpus['drafts']]
    with open('Results/sentence_cluster_bertscore_f1score.txt', 'w',encoding="utf-8") as file_list:
        res =bertscore.compute(predictions=srcs, references=trgs, lang="en")
        for element in res["f1"] :
            file_list.write('%s\n' % element)

    with open('Results/sentence_cluster_bartscore_f1score.txt', 'w',encoding="utf-8") as file_list:
        for element in utilities.BARTScorer().score(srcs,trgs):
            file_list.write('%s\n' % element)

results()