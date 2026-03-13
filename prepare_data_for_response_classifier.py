from prepare_data_for_sft import process_cleaning_dataset_to_SFT,transform_dataset_column_response





if __name__ == '__main__':
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
    LLM_LAT_harmful_dataset_chosen = transform_dataset_column_response(
        data=LLM_LAT_harmful_dataset,
        number_of_responses=2,
        column_to_split_dataset_on_response=['prompt', 'is_unsafe_prompt', 'chosen', 'from_dataset'],
        response_column='chosen',
        is_drop_nan=False,

        different_response_category=False,
        is_unsafe=False
    )
    print('LLM_LAT_harmful_dataset_chosen:')
    print(LLM_LAT_harmful_dataset_chosen.info())

    LLM_LAT_harmful_dataset_rejected = transform_dataset_column_response(
        data=LLM_LAT_harmful_dataset,
        number_of_responses=2,
        column_to_split_dataset_on_response=['prompt', 'is_unsafe_prompt', 'rejected', 'from_dataset'],
        response_column='rejected',
        is_drop_nan=False,

        different_response_category=False,
        is_unsafe=True
    )
    print('LLM_LAT_harmful_dataset_rejected:')
    print(LLM_LAT_harmful_dataset_rejected.info())