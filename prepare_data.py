from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import Literal,Union
import re
from dotenv import load_dotenv
load_dotenv()

def load_dataset_from_source(path_to_dataset:str,
                 source_type:Literal['csv','kaggle','Hugging Face'],
                 name:str=None,
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
            pandas_kwargs={
                'encoding': 'latin-1',
                'on_bad_lines': 'skip',
                'engine': 'python'
            }
        )

    if source_type == 'Hugging Face':
        dataset=load_dataset(path_to_dataset,name=name,split=split)
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
                                            unsafe_prompt_category:Union[str,int]=None)->pd.DataFrame:
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

def category_cleaning(data:pd.DataFrame,
                      list_excluded_categories:list=['TextFooler','Bert-Attack','BAE','PWWS',
                                                     'TextBugger','Deletion Characters','Zero Width',
                                                     'Alzantot','Pruthi'],
                      target_col:str='attack_name'
                      )->Dataset:
    data=data[~data[target_col].isin(list_excluded_categories)]
    data=data.reset_index(drop=True)
    data['index']=data.index
    df = data.copy()

    list_attack_name=df[target_col].unique().tolist()
    short_data = pd.DataFrame()

    for i, (attack_name, group) in enumerate(df.groupby(target_col)):

        mask = (group['index'] % len(list_attack_name)) == i
        temp = group[mask]
        short_data = pd.concat(
            [short_data, temp],
            ignore_index=True
        )
    return Dataset.from_pandas(short_data)

def clasteresation_nested_prompts(data:pd.DataFrame,nested_prompt_column:str,prompt_column:str,n_samples:int,n_first_words:int)->Dataset:
    data[nested_prompt_column]=np.where(data[nested_prompt_column].str.contains('image'),data[prompt_column],data[nested_prompt_column])
    data['first_words']=data[nested_prompt_column].apply(lambda x:' '.join(x.split()[:n_first_words]))
    data['count']=data.groupby('first_words')['first_words'].transform('count')
    data=data[data['count']>=n_samples]
    data_sample=data.groupby('first_words').apply(lambda x:x.sample(n_samples)).reset_index(drop=True)
    return Dataset.from_pandas(data_sample)
def complete_process_loading_dataset(path_to_dataset:str,
                 source_type:Literal['csv','kaggle','Hugging Face'],
                 prompt_column: str,

                 dataset_name:str=None,
                 file_name:str=None,
                 name:str=None,
                 split:str=None,
                 print_info:bool=False,
                 different_prompt_category: bool = False,
                 is_unsafe: bool = None,
                 category_column: str = None,
                 unsafe_prompt_category:Union[str,int] = None,
                 is_special_cleaning: bool = False,
                 is_clasteresation: bool = False,
                 nested_prompt_column:str=None,
                 n_samples:int=3,
                 n_first_words:int=5,
                 list_categories:list=None,
                 target_category_column:str=None,
                                     )->pd.DataFrame:
    data=load_dataset_from_source(path_to_dataset=path_to_dataset,
                                  source_type=source_type,
                                  file_name=file_name,
                                  name=name,
                                  split=split,
                                  print_info=print_info)
    if is_special_cleaning:
        data=category_cleaning(data.to_pandas())
    if is_clasteresation:
        data=clasteresation_nested_prompts(data=data.to_pandas(),
                                           nested_prompt_column=nested_prompt_column,
                                           prompt_column=prompt_column,
                                           n_samples=n_samples,
                                           n_first_words=n_first_words)
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
    Prompt_Injection_Malignant_dataset = complete_process_loading_dataset(
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

    prompt_injection_suffix_attack_adv_prompts_dataset = complete_process_loading_dataset(
        path_to_dataset="adv_prompts.csv",
        source_type='csv',
        prompt_column='Prompt',
        dataset_name='prompt_injection_suffix_attack_adv_prompts',

        print_info=True,
        different_prompt_category=False,
        is_unsafe=True
    )
    print(prompt_injection_suffix_attack_adv_prompts_dataset.info())

    prompt_injection_suffix_attack_viccuna_prompts_dataset = complete_process_loading_dataset(
        path_to_dataset="viccuna_prompts.csv",
        source_type='csv',
        prompt_column='Prompt',
        dataset_name='prompt_injection_suffix_attack_viccuna_prompts',

        print_info=True,
        different_prompt_category=False,
        is_unsafe=True
    )
    print(prompt_injection_suffix_attack_viccuna_prompts_dataset.info())

    prompt_injection_suffix_in_the_wild_forbidden_question_set_df_dataset = complete_process_loading_dataset(
        path_to_dataset='forbidden_question_set_df.csv',
        source_type='csv',
        prompt_column='Prompt',
        dataset_name='prompt_injection_suffix_in_the_wild',

        print_info=True,
        different_prompt_category=False,
        is_unsafe=True
    )
    print(prompt_injection_suffix_in_the_wild_forbidden_question_set_df_dataset.info())

    prompt_injection_suffix_in_the_wild_jailbreak_prompts_dataset = complete_process_loading_dataset(
        path_to_dataset='jailbreak_prompts.csv',
        source_type='csv',
        prompt_column='Prompt',
        dataset_name='prompt_injection_suffix_in_the_wild',

        print_info=True,
        different_prompt_category=False,
        is_unsafe=True
    )
    print(prompt_injection_suffix_in_the_wild_jailbreak_prompts_dataset.info())

    train_deepset_prompt_injections_dataset = complete_process_loading_dataset(
        path_to_dataset="deepset/prompt-injections",
        source_type='Hugging Face',
        split='train',
        prompt_column='text',
        dataset_name='train_deepset_prompt_injections',

        print_info=True,
        different_prompt_category=True,
        category_column='label',
        unsafe_prompt_category=1
    )
    print('train_deepset_prompt_injections_dataset:')
    print(train_deepset_prompt_injections_dataset.info())

    test_deepset_prompt_injections_dataset = complete_process_loading_dataset(
        path_to_dataset="deepset/prompt-injections",
        source_type='Hugging Face',
        split='test',
        prompt_column='text',
        dataset_name='train_deepset_prompt_injections',

        print_info=True,
        different_prompt_category=True,
        category_column='label',
        unsafe_prompt_category=1
    )
    print('test_deepset_prompt_injections_dataset:')
    print(test_deepset_prompt_injections_dataset.info())

    JailbreakBench_JBB_Behaviors_dataset = complete_process_loading_dataset(
        path_to_dataset="JailbreakBench/JBB-Behaviors",
        source_type='Hugging Face',
        name='judge_comparison',
        split='test',
        print_info=True,
        dataset_name='JailbreakBench_JBB_Behaviors',
        prompt_column='prompt',
        different_prompt_category=False,
        is_unsafe=True
    )
    print('JailbreakBench_JBB_Behaviors_dataset:')
    print(JailbreakBench_JBB_Behaviors_dataset.info())

    LLM_LAT_harmful_dataset = complete_process_loading_dataset(
        path_to_dataset="LLM-LAT/harmful-dataset",
        source_type='Hugging Face',
        split='train',
        print_info=True,
        dataset_name='LLM_LAT_harmful',
        prompt_column='prompt',
        different_prompt_category=False,
        is_unsafe=True
    )
    print('LLM_LAT_harmful_dataset:')
    print(LLM_LAT_harmful_dataset.info())

    Mindgard_evaded_prompt_injection_and_jailbreak_samples_dataset=complete_process_loading_dataset(
        path_to_dataset="Mindgard/evaded-prompt-injection-and-jailbreak-samples",
        source_type='Hugging Face',
        split='train',
        print_info=True,
        dataset_name='Mindgard_evaded_prompt_injection_and_jailbreak_samples',
        prompt_column='modified_sample',
        different_prompt_category=False,
        is_unsafe=True,
        is_special_cleaning=True
    )
    print('Mindgard_evaded_prompt_injection_and_jailbreak_samples_dataset:')
    print(Mindgard_evaded_prompt_injection_and_jailbreak_samples_dataset.info())

    jackhhao_jailbreak_classification_dataset_train=complete_process_loading_dataset(
        path_to_dataset="jackhhao/jailbreak-classification",
        source_type='Hugging Face',
        split='train',
        prompt_column='prompt',
        dataset_name='jackhhao_jailbreak_classification',

        print_info=True,
        different_prompt_category=True,
        category_column='type',
        unsafe_prompt_category='jailbreak'
    )
    print('jackhhao_jailbreak_classification_dataset_train:')
    print(jackhhao_jailbreak_classification_dataset_train.info())
    jackhhao_jailbreak_classification_dataset_test = complete_process_loading_dataset(
        path_to_dataset="jackhhao/jailbreak-classification",
        source_type='Hugging Face',
        split='test',
        prompt_column='prompt',
        dataset_name='jackhhao_jailbreak_classification',

        print_info=True,
        different_prompt_category=True,
        category_column='type',
        unsafe_prompt_category='jailbreak'
    )
    print('jackhhao_jailbreak_classification_dataset_test:')
    print(jackhhao_jailbreak_classification_dataset_test.info())

    Deep1994_ReNeLLM_Jailbreak_dataset=complete_process_loading_dataset(
        path_to_dataset="Deep1994/ReNeLLM-Jailbreak",
        source_type='Hugging Face',
        split='train',
        print_info=True,
        dataset_name='Deep1994_ReNeLLM_Jailbreak',
        prompt_column='original_harm_behavior',
        different_prompt_category=False,
        is_unsafe=True

    )
    print('Deep1994_ReNeLLM_Jailbreak_dataset:')
    print(Deep1994_ReNeLLM_Jailbreak_dataset.info())

    prompt_injection_suffix_in_the_wild_forbidden_question_set_with_prompts_dataset=complete_process_loading_dataset(
        path_to_dataset="forbidden_question_set_with_prompts.csv",
        source_type='csv',

        print_info=True,
        dataset_name='forbidden_question_set_with_prompts',
        prompt_column='Prompt',
        different_prompt_category=False,
        is_unsafe=True,
        is_clasteresation=True,
        nested_prompt_column='Prompt',
        n_samples=10,
        n_first_words=5

    )
    print('prompt_injection_suffix_in_the_wild_forbidden_question_set_with_prompts_dataset:')
    print(prompt_injection_suffix_in_the_wild_forbidden_question_set_with_prompts_dataset.info())

    JailbreakV_28K_dataset=complete_process_loading_dataset(
        path_to_dataset="JailbreakV-28K/JailBreakV-28k",
        source_type='Hugging Face',
        name='JailBreakV_28K',
        split='JailBreakV_28K',
        print_info=True,
        dataset_name='JailBreakV_28K',
        prompt_column='jailbreak_query',
        different_prompt_category=False,
        is_unsafe=True,
        is_clasteresation=True,
        nested_prompt_column='jailbreak_query',
        n_samples=3,
        n_first_words=5
    )
    print('JailbreakV_28K_dataset:')
    print(JailbreakV_28K_dataset.info())

    tatsu_lab_alpaca=complete_process_loading_dataset(
        path_to_dataset="tatsu-lab/alpaca",
        source_type='Hugging Face',
        split='train',
        prompt_column='instruction',
        print_info=True,
        dataset_name='tatsu_lab_alpaca',
        different_prompt_category=False,
        is_unsafe=False
    ).sample(6000)
    print('tatsu_lab_alpaca:')
    print(tatsu_lab_alpaca.info())

    akoksal_LongForm = complete_process_loading_dataset(
        path_to_dataset="akoksal/LongForm",
        source_type='Hugging Face',
        split='train',
        prompt_column='input',
        print_info=True,
        dataset_name='akoksal_LongForm',
        different_prompt_category=False,
        is_unsafe=False
    )
    akoksal_LongForm=akoksal_LongForm[(akoksal_LongForm['prompt'].str.len()>=1000) & (akoksal_LongForm['prompt'].str.len()<=13000)].sample(1500)
    dataset_list = [
        Prompt_Injection_Malignant_dataset,
        prompt_injection_suffix_attack_adv_prompts_dataset,
        prompt_injection_suffix_attack_viccuna_prompts_dataset,
        prompt_injection_suffix_in_the_wild_forbidden_question_set_df_dataset,
        prompt_injection_suffix_in_the_wild_jailbreak_prompts_dataset,
        train_deepset_prompt_injections_dataset,
        test_deepset_prompt_injections_dataset,
        JailbreakBench_JBB_Behaviors_dataset,
        LLM_LAT_harmful_dataset,
        Mindgard_evaded_prompt_injection_and_jailbreak_samples_dataset,
        jackhhao_jailbreak_classification_dataset_train,
        jackhhao_jailbreak_classification_dataset_test,
        Deep1994_ReNeLLM_Jailbreak_dataset,
        prompt_injection_suffix_in_the_wild_forbidden_question_set_with_prompts_dataset,
        JailbreakV_28K_dataset,
        tatsu_lab_alpaca,
        akoksal_LongForm
    ]

    final_data=pd.concat(dataset_list)
    final_data = final_data.drop_duplicates(subset=['prompt'])
    print(final_data.info())
    print(final_data['is_unsafe'].value_counts())
