from prepare_data_for_prompts_classifier import (load_dataset_from_source,
                                                 clasteresation_nested_prompts,
                                                 change_dataset_column_to_necessary_form,
                                                 detect_english_text,
                                                 clean_prompt_text)
from typing import Literal, List, Tuple,Union
import numpy as np
import pandas as pd
def process_cleaning_dataset_to_SFT(path_to_dataset:str,
                                            source_type:Literal['csv','kaggle','Hugging Face'],
                                            name:str=None,
                                            file_name:str=None,
                                            split:str=None,
                                            print_info:bool=False,
                                            dataset_name:str=None,
                                            prompt_column:str=None,
                                            different_prompt_category: bool = False,
                                            is_unsafe: bool = None,
                                            category_column: str = None,
                                            unsafe_prompt_category: Union[str, int] = None,

                                            is_clasteresation: bool = False,
                                            nested_prompt_column:str=None,
                                            n_samples:int=3,
                                            n_first_words:int=5,
                                            is_detect_english_texts: bool = False,
                                            min_confidence:int=0.95,
                                            is_n_samples_split: bool = False,
                                            n_samples_split:int=None,
                                            ):
    data = load_dataset_from_source(path_to_dataset=path_to_dataset,
                                    source_type=source_type,
                                    file_name=file_name,
                                    name=name,
                                    split=split,
                                    print_info=print_info)
    if is_clasteresation:
        data=clasteresation_nested_prompts(data=data,
                                           nested_prompt_column=nested_prompt_column,
                                           prompt_column=prompt_column,
                                           n_samples=n_samples,
                                           n_first_words=n_first_words)
    data=change_dataset_column_to_necessary_form(dataset=data,
                                                 prompt_column=prompt_column,
                                                 different_prompt_category=different_prompt_category,
                                                 is_unsafe=is_unsafe,
                                                 category_column=category_column,
                                                 unsafe_prompt_category=unsafe_prompt_category

                                   )
    data = data.drop_duplicates(subset=['prompt'])
    if is_n_samples_split:
        data = data.head(n_samples_split)
    if is_detect_english_texts:
        data=data[data['prompt'].apply(lambda x:detect_english_text(x,min_confidence=min_confidence))]
    data['from_dataset'] = f'{dataset_name}'
    return data
def transform_dataset_column_response(data:pd.DataFrame,
                                      response_column:str,
                                      is_drop_nan:bool=False,
                                      columns_to_drop_nan:list=None,
                                      different_response_category:bool=False,
                                      is_unsafe: bool = None,
                                      category_column: str = None,
                                      unsafe_response_category: Union[str, int] = None,):

    data=change_dataset_column_to_necessary_form(dataset=data,
                                            prompt_column=response_column,
                                            name_column_for_rename='response',
                                            different_prompt_category=different_response_category,
                                            is_unsafe=is_unsafe,
                                            category_column=category_column,
                                            unsafe_prompt_category=unsafe_response_category)
    if is_drop_nan:
        data=data.dropna(subset=columns_to_drop_nan)
    data = data.drop_duplicates(subset=['response'])
    return data[['prompt','is_unsafe_prompt','response','is_unsafe_response']]
def split_dataset_on_response(data:pd.DataFrame,
                              number_of_response:int,
                              column_to_split_dataset_on_response:List[List[str]]=None,
                              response_column:str='response',
                              is_drop_nan:bool=False,
                              columns_to_drop_nan:Union[list,List[List[str]]]=None,
                              different_response_category:bool=False,
                              is_unsafe: Union[bool,List[bool]] = None,
                              category_column:Union[list, str] = None,
                              unsafe_response_category: Union[str, int] = None
                              )->Union[pd.DataFrame,Tuple[pd.DataFrame,...]]:
    if number_of_response in (0,None):
        data['response']=np.nan
        data=transform_dataset_column_response(data=data,response_column=response_column,
                                      is_drop_nan=is_drop_nan,
                                      columns_to_drop_nan=columns_to_drop_nan,
                                      different_response_category=different_response_category,
                                      is_unsafe=is_unsafe,
                                      category_column=category_column,
                                      unsafe_response_category=unsafe_response_category)
        return data
    if number_of_response ==1:
        data = transform_dataset_column_response(data=data, response_column=response_column,
                                                 is_drop_nan=is_drop_nan,
                                                 columns_to_drop_nan=columns_to_drop_nan,
                                                 different_response_category=different_response_category,
                                                 is_unsafe=is_unsafe,
                                                 category_column=category_column,
                                                 unsafe_response_category=unsafe_response_category)

        return data
    if number_of_response>=2:

        data_list = []
        for i,columns in enumerate(column_to_split_dataset_on_response):
           _data=transform_dataset_column_response(data=data[columns],
                                                   response_column=response_column,
                                                   is_drop_nan=is_drop_nan,
                                                   columns_to_drop_nan=columns_to_drop_nan,
                                                   different_response_category=different_response_category,
                                                   is_unsafe=is_unsafe,
                                                   category_column=category_column,
                                                   unsafe_response_category=unsafe_response_category)
           data_list.append(_data)
           return tuple(data_list)
if __name__=='__main__':
    '''allenai_wildguardmix_test=process_cleaning_dataset_to_SFT(path_to_dataset="allenai/wildguardmix",
        source_type='Hugging Face',
        name='wildguardtest',
        split='test',
        print_info=True,
        dataset_name='allenai_wildguardmix',
        prompt_column='prompt',
        different_prompt_category=True,
        category_column='prompt_harm_label',
        unsafe_prompt_category='harmful',
        is_clasteresation=True,
        nested_prompt_column='prompt',
        n_samples=1,
        n_first_words=5,
        is_detect_english_texts=True,

        )
    allenai_wildguardmix_test=split_dataset_on_response(data=allenai_wildguardmix_test,
                                 number_of_response=1,
                                 response_column='response',
                                 is_drop_nan=True,
                                 columns_to_drop_nan=['response','prompt_harm_label','response_harm_label'],
                                 different_response_category=True,
                                 category_column='response_harm_label',
                                 unsafe_response_category='harmful')

    print('allenai_wildguardmix_test:')
    print(allenai_wildguardmix_test.info())

    allenai_wildguardmix_train = process_cleaning_dataset_to_SFT(path_to_dataset="allenai/wildguardmix",
                                                                source_type='Hugging Face',
                                                                name='wildguardtrain',
                                                                split='train',
                                                                print_info=True,
                                                                dataset_name='allenai_wildguardmix',
                                                                prompt_column='prompt',
                                                                different_prompt_category=True,
                                                                category_column='prompt_harm_label',
                                                                unsafe_prompt_category='harmful',
                                                                is_clasteresation=True,
                                                                nested_prompt_column='prompt',
                                                                n_samples=1,
                                                                n_first_words=5,
                                                                is_detect_english_texts=True,

                                                                )
    allenai_wildguardmix_train = split_dataset_on_response(data=allenai_wildguardmix_train,
                                                          number_of_response=1,
                                                          response_column='response',
                                                          is_drop_nan=True,
                                                          columns_to_drop_nan=['response', 'prompt_harm_label',
                                                                               'response_harm_label'],
                                                          different_response_category=True,
                                                          category_column='response_harm_label',
                                                          unsafe_response_category='harmful')

    print('allenai_wildguardmix_train:')
    print(allenai_wildguardmix_train.info())'''

    PKU_Alignment_PKU_SafeRLHF_train=process_cleaning_dataset_to_SFT(path_to_dataset="PKU-Alignment/PKU-SafeRLHF",
                                                               source_type='Hugging Face',
                                                               name='default',
                                                               split='train',
                                                               print_info=True,
                                                               dataset_name='PKU-SafeRLHF',
                                                               prompt_column='prompt',
                                                               different_prompt_category=False,
                                                               is_unsafe=False,
                                                               is_clasteresation=True,
                                                               nested_prompt_column='prompt',
                                                               n_samples=1,
                                                               n_first_words=5,
                                                               is_detect_english_texts=True

                                                               )
    PKU_Alignment_PKU_SafeRLHF_train_response_0,PKU_Alignment_PKU_SafeRLHF_train_response_1=split_dataset_on_response(data=PKU_Alignment_PKU_SafeRLHF_train,
                                                                                                                      number_of_response=2,
                                                                                                                      column_to_split_dataset_on_response=[['prompt','response_0','is_response_0_safe'],
                                                                                                                                                           ['prompt','response_1','is_response_1_safe']],
                                                                                                                      is_drop_nan=True,
                                                                                                                      columns_to_drop_nan=[['prompt','response_0','is_response_0_safe'],
                                                                                                                                           ['prompt','response_1','is_response_1_safe']],
                                                                                                                      different_response_category=True,
                                                                                                                      category_column=['is_response_0_safe','is_response_1_safe'],
                                                                                                                      unsafe_response_category='false'
                                                                                                                      )
    print('PKU_Alignment_PKU_SafeRLHF_train_response_0:')
    print(PKU_Alignment_PKU_SafeRLHF_train_response_0.info())
    print('PKU_Alignment_PKU_SafeRLHF_train_response_1:')
    print(PKU_Alignment_PKU_SafeRLHF_train_response_1.info())