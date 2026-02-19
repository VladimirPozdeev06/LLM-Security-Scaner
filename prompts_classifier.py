from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer


def split_data(data:pd.DataFrame):
    data=data.rename(columns={'is_unsafe':'label','prompt':'text'})
    train_data,test_data=train_test_split(data,
                                          test_size=0.2,
                                          stratify=data['from_dataset'],
                                          random_state=17)
    train_data,val_data=train_test_split(train_data,
                                         test_size=0.125,
                                         stratify=train_data['from_dataset'],
                                         random_state=17)
    train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_data[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_data[['text', 'label']])
    return train_dataset,val_dataset,test_dataset

def tokenize_function(examples,tokenizer):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512
    )

