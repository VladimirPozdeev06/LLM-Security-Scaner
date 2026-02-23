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
                              columns_to_drop_nan:list=None,
                              different_response_category:bool=False,
                              is_unsafe: bool = None,
                              category_column: str = None,
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
        for columns in column_to_split_dataset_on_response:
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
    ds=process_cleaning_dataset_to_SFT(path_to_dataset="allenai/wildguardmix",
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
    ds=split_dataset_on_response(data=ds,
                                 number_of_response=1,
                                 response_column='response',
                                 is_drop_nan=True,
                                 columns_to_drop_nan=['response','prompt_harm_label'],
                                 different_response_category=True,
                                 category_column='prompt_harm_label',
                                 unsafe_response_category='harmful')
    print(ds.info())
