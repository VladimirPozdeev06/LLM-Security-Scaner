from prepare_data_for_sft import process_cleaning_dataset_to_SFT
from prompts_classifier import define_prompt_safety,load_pretrained_classification_model
from prepare_data_for_prompts_classifier import change_dataset_column_to_necessary_form
import pandas as pd
def get_chosen_rejected(row,column_1_response:str=None,
                                 column_2_response:str=None,
                                 label_column_1_response:str=None,
                                 label_column_2_response:str=None)->pd.Series:


    if bool(row[label_column_1_response]) and not bool(row[label_column_2_response]):
        return pd.Series({'prompt':row['prompt'],
                          'chosen':row[column_1_response],
                          'rejected':row[column_2_response],
                          'is_unsafe_prompt':row['is_unsafe_prompt'],
                          'confidence':row['predicted_class_confidence']})
    elif not bool(row[label_column_1_response]) and bool(row[label_column_2_response]):
        return pd.Series({'prompt':row['prompt'],
                          'chosen':row[column_2_response],
                          'rejected':row[column_1_response],
                          'is_unsafe_prompt':row['is_unsafe_prompt'],
                          'confidence':row['predicted_class_confidence']})
    else:
        return pd.Series({'prompt':row['prompt'],
                          'chosen':None,'rejected':None,
                          'is_unsafe_prompt':row['is_unsafe_prompt'],
                          'confidence':row['predicted_class_confidence']})
def process_cleaning_data_to_DPO(data:pd.DataFrame,prompt_column='prompt',
                                 name_column_for_rename='prompt',
                                 is_drop_nan:bool=False,
                                 columns_to_drop_nan:list=None,
                                 is_clean_text:bool=False,
                                 columns_to_clean_text:list[str]=None,
                                 has_response_label_column:bool=False,
                                 column_1_response:str=None,
                                 column_2_response:str=None,
                                 label_column_1_response:str=None,
                                 label_column_2_response:str=None,):
    data=change_dataset_column_to_necessary_form(dataset=data,
                                                 prompt_column=prompt_column,
                                                 name_column_for_rename=name_column_for_rename,
                                                 is_drop_nan=is_drop_nan,
                                                 columns_to_drop_nan=columns_to_drop_nan,
                                                 is_clean_text=is_clean_text,
                                                 columns_to_clean_text=columns_to_clean_text,
                                                 is_define_prompt_category=False

                                                 )

    if has_response_label_column:
        data=data[data[label_column_1_response].astype(int)+data[label_column_2_response].astype(int)==1]

        data=data.apply(lambda row: get_chosen_rejected(row,
                                                        column_1_response=column_1_response,
                                                        column_2_response=column_2_response,
                                                        label_column_1_response=label_column_1_response,
                                                        label_column_2_response=label_column_2_response),axis=1)


        data=data.dropna(subset=['chosen','rejected'])
    else:
        data=data.rename(columns={'predicted_class_confidence':'confidence'})


    return data[['prompt','chosen','rejected','is_unsafe_prompt','confidence']]




if __name__ == '__main__':
    PKU_Alignment_PKU_SafeRLHF_train = process_cleaning_dataset_to_SFT(path_to_dataset="PKU-Alignment/PKU-SafeRLHF",
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
    model, tokenizer = load_pretrained_classification_model('prompts classifier')
    PKU_Alignment_PKU_SafeRLHF_train = define_prompt_safety(PKU_Alignment_PKU_SafeRLHF_train,
                                                                model, tokenizer,
                                                                min_confidence=None,save_confidence=True)

    # add define prompt safety in process_cleaning_dataset_to_SFT


    PKU_Alignment_PKU_SafeRLHF_train = process_cleaning_data_to_DPO(data=PKU_Alignment_PKU_SafeRLHF_train,
                                                                        is_drop_nan=True,
                                                                        columns_to_drop_nan=['prompt', 'response_0',
                                                                                             'response_1',
                                                                                             'is_response_0_safe',
                                                                                             'is_response_1_safe'],
                                                                        is_clean_text=True,
                                                                        columns_to_clean_text=['prompt', 'response_0',
                                                                                               'response_1'],
                                                                        has_response_label_column=True,
                                                                        column_1_response='response_0',
                                                                        column_2_response='response_1',
                                                                        label_column_1_response='is_response_0_safe',
                                                                        label_column_2_response='is_response_1_safe')
    print('PKU_Alignment_PKU_SafeRLHF_train:')
    print(PKU_Alignment_PKU_SafeRLHF_train.info())


    LLM_LAT_harmful_dataset = process_cleaning_dataset_to_SFT(path_to_dataset="LLM-LAT/harmful-dataset",
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

    LLM_LAT_harmful_dataset = define_prompt_safety(LLM_LAT_harmful_dataset, model, tokenizer,
                                                                min_confidence=None,save_confidence=True)

    LLM_LAT_harmful_dataset=process_cleaning_data_to_DPO(data=LLM_LAT_harmful_dataset,
                                                             is_drop_nan=True,
                                                             columns_to_drop_nan=['prompt', 'chosen','rejected'],
                                                             is_clean_text=True,
                                                             columns_to_clean_text=['prompt', 'chosen','rejected'],
                                                             has_response_label_column=False)

    print('LLM_LAT_harmful_dataset:')
    print(LLM_LAT_harmful_dataset.info())

    claude_dataset=process_cleaning_dataset_to_SFT(
        path_to_dataset='dpo_200_pairs.csv',
        source_type='csv',
        print_info=True,
        dataset_name='claude_dataset',
        prompt_column='prompt'

    )
    model, tokenizer = load_pretrained_classification_model('prompts classifier')
    claude_dataset=define_prompt_safety(claude_dataset, model, tokenizer,
                                                            min_confidence=None,save_confidence=True)
    claude_dataset=process_cleaning_data_to_DPO(data=claude_dataset,
                                                         is_drop_nan=True,
                                                         columns_to_drop_nan=['prompt', 'chosen','rejected'],
                                                         is_clean_text=True,
                                                         columns_to_clean_text=['prompt', 'chosen','rejected'],
                                                         has_response_label_column=False)
    print('claude_dataset:')
    print(claude_dataset.info())


    simple_dpo_data_without_response_from_sft_model=pd.concat([PKU_Alignment_PKU_SafeRLHF_train,LLM_LAT_harmful_dataset,claude_dataset])
    simple_dpo_data_without_response_from_sft_model=simple_dpo_data_without_response_from_sft_model.drop_duplicates(subset='prompt')
    simple_dpo_data_without_response_from_sft_model.to_csv('simple_dpo_data_without_response_from_sft_model.csv')
    print('simple_dpo_data_without_response_from_sft_model:')
    print(simple_dpo_data_without_response_from_sft_model.info())
    print(simple_dpo_data_without_response_from_sft_model['is_unsafe_prompt'].value_counts())