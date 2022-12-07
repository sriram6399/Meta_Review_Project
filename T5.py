import csv
from html.entities import html5
from datasets import load_dataset,Dataset,DatasetDict,load_metric
from transformers import DataCollatorWithPadding,T5Tokenizer, T5ForConditionalGeneration,AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
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
model_checkpoints = "t5-base"
data = utilities.read_data("Data/data1.csv")
#x,y = train_test_split(data,test_size=0.2,shuffle=True)
train,test = train_test_split(data,test_size=0.4,shuffle=False)
train,valid = train_test_split(train,test_size=0.05,shuffle= False)
data = DatasetDict({
    'train': Dataset(pa.Table.from_pandas(train)),
    'test': Dataset(pa.Table.from_pandas(test)),
    'valid': Dataset(pa.Table.from_pandas(valid))})
    
tokenizer = T5Tokenizer.from_pretrained(model_checkpoints)

def preprocess_data(data):
    inputs = [review for review in data['review']]
    model_inputs = tokenizer.encode(inputs,return_tensors="pt",  max_length=max_input, padding='max_length', truncation=True)

    with tokenizer.as_target_tokenizer():
        inputs = [review for review in data['meta_review']]
        targets = tokenizer.encode(inputs,return_tensors="pt", max_length=max_target, padding='max_length', truncation=True)
    
    

    return model_inputs

#tokenize_data = data.map(preprocess_data, batched = True)



#tokenize_data.set_format("torch",columns=["input_ids", "attention_mask", "labels"])

#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
device = torch.device("cuda")
model = T5ForConditionalGeneration.from_pretrained(model_checkpoints)
model.to(device)
#model.gradient_checkpointing_enable()

i=0
for p in model.parameters():
    i=i+1
    '''p.requires_grad = False
    if(i==250):
        break'''

print("No of parameters",i)

def computeMetrics(eval_pred):
    predictions, labels= eval_pred
    
    srcs = [tokenizer.decode(x) for x in predictions]
    trgs = [tokenizer.decode(x) for x in labels]
    print(utilities.BARTScorer().score(srcs,trgs))
    return metric.compute(predictions=srcs,references=trgs,lang="en")

args = Seq2SeqTrainingArguments(
    'logs/t5_meta-review', #save directory
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
    #train_dataset=tokenize_data['train'],
    #eval_dataset=tokenize_data['valid'],
    #data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics= computeMetrics
    
)

def training():
    trainer.train()
    trainer.save_model('models/t5_trainer')

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
    tokens_input = [tokenizer.encode(x,return_tensors="pt", max_length=512, truncation=True) for x in data['test']['review']]

    summary_ids = [model.generate(x.to(device),min_length=60,max_length=180,length_penalty=4.0) for x in tokens_input]
    srcs = [tokenizer.decode(x[0], skip_special_tokens =True,clean_up_tokenization_spaces=True) for x in summary_ids]
    trgs = [x for x in data['test']['meta_review']]
    print("doing 1")
    bert_res = bertscore.compute(predictions=srcs, references=trgs, lang="en")
    rouge_res = rouge.compute(predictions=srcs, references=trgs)
    print("doing")
    with open('Results/t5_wf_outputs.csv', 'w',newline='',encoding="utf-8") as file_list:
        write = csv.writer(file_list,delimiter=',')
        temp = []
        bert_tot=0
        for i in range(0,len(srcs)):
            temp.append([srcs[i],trgs[i],bert_res['f1'][i]])
            bert_tot = bert_tot + bert_res['f1'][i]   
        
        write.writerows(temp)
    print(bert_tot)      
    with open('Results/finals_stats.txt','a+',newline='',encoding="utf-8") as file_list2:
        file_list2.write("\n T5 Without fine tuning: \n")
        file_list2.write(str({"bert_score":(bert_tot/len(srcs)),"rouge":rouge_res}))
    
           
#training()
#free_gpu_cache()
testing()



