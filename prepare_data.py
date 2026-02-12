from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import Literal
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

def change_dataset_column_to_necessary_form(dataset:Dataset,
                                            prompt_column:str,
                                            different_prompt_category:bool=False,
                                            is_unsafe:bool=False,
                                            category_column:str=None,
                                            unsafe_prompt_category:str=None):
    dataset=dataset.to_pandas()
    dataset=dataset.rename(columns={prompt_column:'prompt'})
    if different_prompt_category:
        dataset['is_unsafe']=np.where(dataset[category_column]==unsafe_prompt_category,1,0)
    else:
        dataset['is_unsafe']=int(is_unsafe)


if __name__=='__main__':
    Prompt_Injection_Malignant_dataset=load_dataset_from_source(
        path_to_dataset="marycamilainfo/prompt-injection-malignant",
        source_type='kaggle',
        file_name='malignant.csv',

        print_info=True
    )
    prompt_injection_suffix_attack_adv_prompts_dataset=load_dataset_from_source(
        path_to_dataset="arielzilber/prompt-injection-suffix-attack",
        source_type='kaggle',
        file_name="adv_prompts.csv",
        print_info=True
    )

