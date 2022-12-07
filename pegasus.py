import csv
from html.entities import html5
from datasets import load_dataset,Dataset,DatasetDict,load_metric
from transformers import DataCollatorWithPadding,AutoTokenizer,AutoModel,AutoConfig,AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import utilities
from sklearn.model_selection import train_test_split
import pyarrow as pa
from evaluate import load
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import gc
from torch.utils.data import DataLoader


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    gc.collect()
    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

bertscore = load("bertscore")
rouge = load('rouge')

metric = load_metric('bertscore')


max_input = 1024
max_target = 140
batch_size = 1
model_checkpoints = "google/pegasus-cnn_dailymail"
data = utilities.read_data("Data/data1.csv")
#x,y = train_test_split(data,test_size=0.2,shuffle=True)
train,test = train_test_split(data,test_size=0.4,shuffle=False)
train,valid = train_test_split(train,test_size=0.05,shuffle= False)
data = DatasetDict({
    'train': Dataset(pa.Table.from_pandas(train)),
    'test': Dataset(pa.Table.from_pandas(test)),
    'valid': Dataset(pa.Table.from_pandas(valid))})
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)

def preprocess_data(data):
    inputs = [review for review in data['review']]
    model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)

    with tokenizer.as_target_tokenizer():
        inputs = [review for review in data['meta_review']]
        targets = tokenizer(inputs, max_length=max_target, padding='max_length', truncation=True)
    
    model_inputs['labels'] = targets['input_ids']

    return model_inputs

tokenize_data = data.map(preprocess_data, batched = True)



tokenize_data.set_format("torch",columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)
#model.gradient_checkpointing_enable()

i=0
for p in model.parameters():
    i=i+1
    p.requires_grad = False
    if(i==650):
        break

print("No of parameters",i)

def computeMetrics(eval_pred):
    predictions, labels= eval_pred
    
    srcs = [tokenizer.decode(x) for x in predictions]
    trgs = [tokenizer.decode(x) for x in labels]
    print(utilities.BARTScorer().score(srcs,trgs))
    return metric.compute(predictions=srcs,references=trgs,lang="en")

args = Seq2SeqTrainingArguments(
    'logs/pegasus_meta-review', #save directory
    evaluation_strategy= "epoch",
    learning_rate=2e-2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size= batch_size,
    gradient_accumulation_steps=5,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=2,
    predict_with_generate=True,
    #gradient_checkpointing=True,
    eval_accumulation_steps=5,
    )

trainer = Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenize_data['train'],
    eval_dataset=tokenize_data['valid'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics= computeMetrics
    
)

def training():
    trainer.train()
    trainer.save_model('models/pegasus_trainer')

def testing():
    '''saved_model = AutoModelForSeq2SeqLM.from_pretrained('models/trainer1')
    trainer = Seq2SeqTrainer(
    saved_model, 
    args,
    train_dataset=tokenize_data['train'],
    eval_dataset=tokenize_data['valid'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics= computeMetrics
    
    )'''
    #test_loader = DataLoader(tokenize_data['test'], batch_size=1, shuffle=False)
    #raw_pred,labels,_ =trainer.prediction_loop(test_loader,description="prediction")
    #print(tokenizer.decode(raw_pred[0]))
    raw_pred,labels,_ =trainer.predict(tokenize_data['test'])
    srcs = [tokenizer.decode(x, skip_special_tokens =True,clean_up_tokenization_spaces=True) for x in raw_pred]
    trgs = [tokenizer.decode(x,skip_special_tokens =True,clean_up_tokenization_spaces=True) for x in labels]
    print("doing 1")
    bert_res = bertscore.compute(predictions=srcs, references=trgs, lang="en")
    rouge_res = rouge.compute(predictions=srcs, references=trgs)
    print("doing")
    with open('Results/pegasus_f_outputs.csv', 'w',newline='',encoding="utf-8") as file_list:
        write = csv.writer(file_list,delimiter=',')
        temp = []
        bert_tot=0
        for i in range(0,len(srcs)):
            temp.append([srcs[i],trgs[i],bert_res['f1'][i]])
            bert_tot = bert_tot + bert_res['f1'][i]   
        
        write.writerows(temp)
    print(bert_tot)      
    with open('Results/finals_stats.txt','a+',newline='',encoding="utf-8") as file_list2:
        file_list2.write("\n Pegasus With fine tuning: \n")
        file_list2.write(str({"bert_score":(bert_tot/len(srcs)),"rouge":rouge_res}))
    
           
training()
#free_gpu_cache()
testing()



