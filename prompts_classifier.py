import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import  AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from typing import List
def split_data(data: pd.DataFrame,text_column:str='prompt',label_column:str='is_unsafe') :
    data = data.rename(columns={text_column: 'text', label_column: 'label'})
    train_data, test_data = train_test_split(data,
                                             test_size=0.2,
                                             stratify=data['from_dataset'],
                                             random_state=17)
    train_data, val_data = train_test_split(train_data,
                                            test_size=0.125,
                                            stratify=train_data['from_dataset'],
                                            random_state=17)
    train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_data[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_data[['text', 'label']])
    return train_dataset, val_dataset, test_dataset


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512
    )


def compute_metrics(pred) -> dict:
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall,

        'precision': precision
    }

def load_pretrained_model(path_to_model:str,device:str='cpu'):
    tokenizer=AutoTokenizer.from_pretrained(path_to_model)
    model=AutoModelForSequenceClassification.from_pretrained(path_to_model)
    model.eval()
    model.to(device)
    return model, tokenizer

def predict_batch(prompts:List,model,tokenizer,batch_size:int=32,device:str='cpu'):
    results=[]
    for i in tqdm(range(0,len(prompts),batch_size)):
        prompt_batch=prompts[i:i+batch_size]
        ##обавить **kwargs в tokenize_function
        inputs=tokenizer(
            prompt_batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs=model(**inputs)
            logits=outputs.logits
            probabilities=torch.softmax(logits,dim=1)
            predictions=torch.argmax(logits,dim=1)

        for j,text_prompt in  enumerate(prompt_batch):
            is_unsafe=predictions[j].item()
            confidence=probabilities[j][is_unsafe].item()
            results.append({
                'prompt': text_prompt,
                'is_unsafe_prompt':is_unsafe,
                'confidence':confidence
            })
    return results

def define_prompt_safety(data:pd.DataFrame,model,tokenizer,batch_size:int=32,device:str='cpu',prompt_column:str='prompt',min_confidence:float=None):
    prompts=data[prompt_column].tolist()
    results=predict_batch(prompts,model,tokenizer,batch_size,device)
    results=pd.DataFrame(results)
    if min_confidence is not None:
        results=results[results['confidence']>=min_confidence]
    data=pd.merge(data.drop(columns='is_unsafe_prompt'),results[['prompt','is_unsafe_prompt']],how='right',on='prompt')
    return data
