from prepare_data_for_prompts_classifier import (load_dataset_from_source,
                                                 clasteresation_nested_prompts,
                                                 change_dataset_column_to_necessary_form,
                                                 detect_english_text)
from typing import Literal, List, Tuple,Union
import pandas as pd
def split_dataset_on_response(data:pd.DataFrame,number_of_response:int,
                              column_to_split_dataset_on_response:List[List[str]],
                              )->Tuple[pd.DataFrame,...]:
    if number_of_response in (0,None,1):
        return (data,)
    if number_of_response>=2:

       data_list = []
       for columns in column_to_split_dataset_on_response:
           data_list.append(data[columns])
           return tuple(data_list)

def process_cleaning_dataset_to_SFT(path_to_dataset:str,
                                            source_type:Literal['csv','kaggle','Hugging Face'],
                                            name:str=None,
                                            file_name:str=None,
                                            split:str=None,
                                            print_info:bool=False,
                                            prompt_column:str=None,
                                            different_prompt_category: bool = False,
                                            is_unsafe: bool = None,
                                            category_column: str = None,
                                            unsafe_prompt_category: Union[str, int] = None,
                                            add_column_to_output: bool = False,
                                            list_column_add_to_output: list = None,
                                            number_of_response:int=None,
                                            column_to_split_dataset_on_response:List[List[str]]=None,
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
                                                 unsafe_prompt_category=unsafe_prompt_category,
                                                 add_column_to_output=add_column_to_output,
                                                 list_column_add_to_output=list_column_add_to_output
                                   )
    data = data.drop_duplicates(subset=['prompt'])
    if is_n_samples_split:
        data = data.head(n_samples_split)
    if is_detect_english_texts:
        data=data[data['prompt'].apply(lambda x:detect_english_text(x,min_confidence=min_confidence))]
    return data


