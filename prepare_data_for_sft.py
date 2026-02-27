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
                                      number_of_responses:int,
                                      column_to_split_dataset_on_response:List[str],
                                      response_column:str,
                                      is_drop_nan:bool=False,
                                      columns_to_drop_nan:list=[],
                                      different_response_category:bool=False,
                                      is_unsafe: bool = None,
                                      category_column: str = None,
                                      unsafe_response_category: Union[str, int,bool] = None,):

    data=data[column_to_split_dataset_on_response]

    if number_of_responses==0:
        data['response']=np.nan
    data=change_dataset_column_to_necessary_form(dataset=data,
                                            prompt_column=response_column,
                                            name_column_for_rename='response',
                                            is_drop_nan=is_drop_nan,
                                            columns_to_drop_nan=columns_to_drop_nan,
                                            different_prompt_category=different_response_category,
                                            is_unsafe=is_unsafe,
                                            category_column=category_column,
                                            unsafe_prompt_category=unsafe_response_category)

    data = data.drop_duplicates(subset=['response'])
    return data[['prompt','is_unsafe_prompt','response','is_unsafe_response','from_dataset']]

if __name__=='__main__':
    allenai_wildguardmix_test=process_cleaning_dataset_to_SFT(path_to_dataset="allenai/wildguardmix",
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

    allenai_wildguardmix_test=transform_dataset_column_response(data=allenai_wildguardmix_test,
                                 number_of_responses=1,
                                 column_to_split_dataset_on_response=['prompt','is_unsafe_prompt','response','prompt_harm_label','response_harm_label', 'from_dataset'],
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
    allenai_wildguardmix_train = transform_dataset_column_response(data=allenai_wildguardmix_train,
                                                                  number_of_responses=1,
                                                                  column_to_split_dataset_on_response=['prompt','response','is_unsafe_prompt',
                                                                                                       'prompt_harm_label',
                                                                                                       'response_harm_label', 'from_dataset'],
                                                                  response_column='response',
                                                                  is_drop_nan=True,
                                                                  columns_to_drop_nan=['response', 'prompt_harm_label',
                                                                                       'response_harm_label'],
                                                                  different_response_category=True,
                                                                  category_column='response_harm_label',
                                                                  unsafe_response_category='harmful')

    print('allenai_wildguardmix_train:')
    print(allenai_wildguardmix_train.info())

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

    PKU_Alignment_PKU_SafeRLHF_train_response_0=transform_dataset_column_response(data=PKU_Alignment_PKU_SafeRLHF_train,
                                                                                number_of_responses=2,
                                                                                column_to_split_dataset_on_response=['prompt','is_unsafe_prompt','response_0','is_response_0_safe', 'from_dataset'],
                                                                                response_column='response_0',
                                                                                is_drop_nan=True,
                                                                                columns_to_drop_nan=['prompt','is_response_0_safe'],
                                                                                different_response_category=True,
                                                                                category_column='is_response_0_safe',
                                                                                unsafe_response_category=False)
    print('PKU_Alignment_PKU_SafeRLHF_train_response_0:')
    print(PKU_Alignment_PKU_SafeRLHF_train_response_0.info())

    PKU_Alignment_PKU_SafeRLHF_train_response_1 = transform_dataset_column_response(
        data=PKU_Alignment_PKU_SafeRLHF_train,
        number_of_responses=2,
        column_to_split_dataset_on_response=['prompt', 'is_unsafe_prompt', 'response_1', 'is_response_1_safe', 'from_dataset'],
        response_column='response_1',
        is_drop_nan=True,
        columns_to_drop_nan=['prompt', 'is_response_1_safe'],
        different_response_category=True,
        category_column='is_response_1_safe',
        unsafe_response_category=False)
    print('PKU_Alignment_PKU_SafeRLHF_train_response_1:')
    print(PKU_Alignment_PKU_SafeRLHF_train_response_1.info())

    PKU_Alignment_PKU_SafeRLHF_test = process_cleaning_dataset_to_SFT(path_to_dataset="PKU-Alignment/PKU-SafeRLHF",
                                                                       source_type='Hugging Face',
                                                                       name='default',
                                                                       split='test',
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
    
    PKU_Alignment_PKU_SafeRLHF_test_response_0 = transform_dataset_column_response(
        data=PKU_Alignment_PKU_SafeRLHF_test,
        number_of_responses=2,
        column_to_split_dataset_on_response=['prompt', 'is_unsafe_prompt', 'response_0', 'is_response_0_safe', 'from_dataset'],
        response_column='response_0',
        is_drop_nan=True,
        columns_to_drop_nan=['prompt', 'is_response_0_safe'],
        different_response_category=True,
        category_column='is_response_0_safe',
        unsafe_response_category=False)
    print('PKU_Alignment_PKU_SafeRLHF_test_response_0:')
    print(PKU_Alignment_PKU_SafeRLHF_test_response_0.info())

    PKU_Alignment_PKU_SafeRLHF_test_response_1 = transform_dataset_column_response(
        data=PKU_Alignment_PKU_SafeRLHF_test,
        number_of_responses=2,
        column_to_split_dataset_on_response=['prompt', 'is_unsafe_prompt', 'response_1', 'is_response_1_safe', 'from_dataset'],
        response_column='response_1',
        is_drop_nan=True,
        columns_to_drop_nan=['prompt', 'is_response_1_safe'],
        different_response_category=True,
        category_column='is_response_1_safe',
        unsafe_response_category=False)
    print('PKU_Alignment_PKU_SafeRLHF_test_response_1:')
    print(PKU_Alignment_PKU_SafeRLHF_test_response_1.info())

    LLM_LAT_harmful_dataset=process_cleaning_dataset_to_SFT(path_to_dataset="LLM-LAT/harmful-dataset",
                                                               source_type='Hugging Face',
                                                               split='train',

                                                               print_info=True,
                                                               dataset_name='LLM_LAT_harmful_dataset',
                                                               prompt_column='prompt',
                                                               different_prompt_category=False,
                                                               is_unsafe=True,
                                                               is_clasteresation=True,
                                                               nested_prompt_column='prompt',
                                                               n_samples=1,
                                                               n_first_words=5,
                                                               is_detect_english_texts=True

                                                               )

    LLM_LAT_harmful_dataset_chosen=transform_dataset_column_response(
                                        data=LLM_LAT_harmful_dataset,
                                        number_of_responses=2,
                                        column_to_split_dataset_on_response=['prompt', 'is_unsafe_prompt','chosen', 'from_dataset'],
                                        response_column='chosen',
                                        is_drop_nan=False,

                                        different_response_category=False,
                                        is_unsafe=False
                                        )
    print('LLM_LAT_harmful_dataset_chosen:')
    print(LLM_LAT_harmful_dataset_chosen.info())

    LLM_LAT_harmful_dataset_rejected= transform_dataset_column_response(
        data=LLM_LAT_harmful_dataset,
        number_of_responses=2,
        column_to_split_dataset_on_response=['prompt', 'is_unsafe_prompt','rejected', 'from_dataset'],
        response_column='rejected',
        is_drop_nan=False,

        different_response_category=False,
        is_unsafe=True
    )
    print('LLM_LAT_harmful_dataset_rejected:')
    print(LLM_LAT_harmful_dataset_rejected.info())