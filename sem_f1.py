from evaluate import load
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List
import tensorflow as tf
from torchmetrics.functional.text.bert import bert_score




#bertscore = load("bertscore")
'''gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])
  
print(embeddings)'''


def convert_list_of_list(word_list:List[str]):
    out_lst = []
    for x in word_list:
        lst = []
        for y in x.split(". "):
            lst.append(y)
        out_lst.append(lst)

    return out_lst

def get_f1(preds,ref):
    srcs = convert_list_of_list(ref)
    tgts = convert_list_of_list(preds)
    
    
    '''x1 = use_similarity(srcs,tgts) 
    
    x2 = use_similarity(tgts,srcs)
    x = []
    for i in range(len(x1)):
        x.append(((sum(x1[i])/len(x1[i]))+(sum(x2[i])/len(x2[i])))/2)'''
    
    
    y1 = sbert_similarity(srcs,tgts,'paraphrase-distilroberta-base-v1')
    y2 =sbert_similarity(tgts,srcs,'paraphrase-distilroberta-base-v1')
    
    y = []
    for i in range(len(y1)):
        y.append(((sum(y1[i])/len(y1[i]))+(sum(y2[i])/len(y2[i])))/2)

    
    return y
    #return [sum(x)/len(x) ,sum(y)/len(y)]



def compute_cosine_similarity(pred_embeds, ref_embeds):
    cosine_scores = cosine_similarity(pred_embeds, ref_embeds)
    return np.max(cosine_scores, axis=-1).tolist() #, np.argmax(cosine_scores, axis=-1).tolist()




def use_similarity(predictions: List[List[str]], references: List[List[str]]):
    '''
    Universal Sentence Encoder
    :param predictions: List of predicted summaries. Each sample should also be a list, a list of sentences
    :param references: List of reference summaries. Each sample should also be a list, a list of sentences
    :return:
    '''

    assert len(predictions) == len(references), 'Mismatch in number of predictions to the number of references'
    print("In Universal Sentence Encoder")



    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    out_scores = [0]*len(predictions)
    for idx, (preds, refs) in enumerate(zip(predictions, references)):
        pred_embeddings = model(preds)
        ref_embeddings = model(refs)
        out_scores[idx] = compute_cosine_similarity(pred_embeddings, ref_embeddings)



    return out_scores




def sbert_similarity(predictions: List[List[str]], references: List[List[str]], model_name):
    '''
    Paraphrase similarity and Semantic Textual similarity
        :param predictions: List of predicted summaries. Each sample should also be a list, a list of sentences
        :param references: List of reference summaries. Each sample should also be a list, a list of sentences
        :return:
    '''



    assert model_name in ["paraphrase-distilroberta-base-v1", 'stsb-roberta-large']
    assert len(predictions) == len(references), 'Mismatch in number of predictions to the number of references'

    print("In sbert sim. Model: {model_name}")

    model = SentenceTransformer(model_name)
    out_scores = [0] * len(predictions)
    for idx, (preds, refs) in enumerate(zip(predictions, references)):
        pred_embeddings = model.encode(preds)
        ref_embeddings = model.encode(refs)
        out_scores[idx] = compute_cosine_similarity(pred_embeddings, ref_embeddings)



    return out_scores


#print(get_f1(["I love watching movies. playing cricket."],["he love watching movies. playing vricket."]))