from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import Literal
import re
def load_dataset_from_source(path_to_dataset:str,
                 source_type:Literal['csv','kaggle','Hugging Face'],
                 file_name:str=None,
                 split:str=None,
                 print_info:bool=False)->Dataset:
    if source_type=='csv':
        dataset=Dataset.from_csv(path_to_dataset)
    if source_type == 'kaggle':
        dataset = kagglehub.dataset_load(
            KaggleDatasetAdapter.HUGGING_FACE,
            path_to_dataset,
            file_name,
        )
    if source_type == 'Hugging Face':
        dataset=load_dataset(path_to_dataset,split=split)
    if print_info:
        print('dataset information: ',dataset)
    return dataset
def clean_prompt_text(text:str)->str:

    text=text.strip()
    text=re.sub(r'\s+',' ',text)
    return text
def change_dataset_column_to_necessary_form(dataset:Dataset,
                                            prompt_column:str,
                                            different_prompt_category:bool=False,
                                            is_unsafe:bool=None,
                                            category_column:str=None,
                                            unsafe_prompt_category:str=None)->pd.DataFrame:
    dataset=dataset.to_pandas()
    dataset=dataset.rename(columns={prompt_column:'prompt'})
    dataset['prompt']=dataset['prompt'].apply(clean_prompt_text)
    if different_prompt_category:
        dataset['is_unsafe']=np.where(dataset[category_column]==unsafe_prompt_category,1,0)
    else:
        dataset['is_unsafe']=int(is_unsafe)

    dataset=dataset.dropna(subset=['prompt'])
    dataset=dataset[['prompt','is_unsafe']]
    return dataset

def complete_process_loading_dataset(path_to_dataset:str,
                 source_type:Literal['csv','kaggle','Hugging Face'],
                 prompt_column: str,
                 dataset_name:str=None,
                 file_name:str=None,
                 split:str=None,
                 print_info:bool=False,
                 different_prompt_category: bool = False,
                 is_unsafe: bool = None,
                 category_column: str = None,
                 unsafe_prompt_category: str = None

                                     )->pd.DataFrame:
    data=load_dataset_from_source(path_to_dataset=path_to_dataset,
                                  source_type=source_type,
                                  file_name=file_name,
                                  split=split,
                                  print_info=print_info)
    data=change_dataset_column_to_necessary_form(dataset=data,
                                                 prompt_column=prompt_column,
                                                 different_prompt_category=different_prompt_category,
                                                 is_unsafe=is_unsafe,
                                                 category_column=category_column,
                                                 unsafe_prompt_category=unsafe_prompt_category)
    data=data.drop_duplicates(subset=['prompt'])
    data['from_dataset']=f'{dataset_name}'
    return data

if __name__=='__main__':

    Prompt_Injection_Malignant_dataset=complete_process_loading_dataset(
        path_to_dataset="marycamilainfo/prompt-injection-malignant",
        source_type='kaggle',
        prompt_column='text',
        dataset_name='prompt_injection_malignant',
        file_name='malignant.csv',
        print_info=True,
        different_prompt_category=True,
        category_column='category',
        unsafe_prompt_category='jailbreak'
    )
    print(Prompt_Injection_Malignant_dataset.info())


