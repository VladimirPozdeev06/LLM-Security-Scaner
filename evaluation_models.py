import pandas as pd

from prepare_data_for_dpo import extract_response_and_analysis, extract_analysis_fields

def prepare_data_to_necessary_from_for_evaluation(path:str='final_test_data/test_data_final_all_models.csv')->pd.DataFrame:
    data = pd.read_csv('/content/drive/MyDrive/test_data_final_all_models.csv', index_col=0)
    data['response_part'], data['analysis_part'] = zip(
        *data['text'].apply(extract_response_and_analysis)
    )
    fields = data['text'].apply(extract_analysis_fields)
    fields_df = pd.DataFrame(fields.tolist(), index=data.index)  # ← явно передай индекс
    data = data.join(fields_df)
    data = data.rename(columns={'is_unsafe_prompt': 'is_unsafe_prompt_real',
                                'response_part': 'response_part_real',
                                'analysis_part': 'analysis_part_real',

                                'attack_type': 'attack_type_real',
                                'confidence': 'confidence_real',
                                'recommendation': 'recommendation_real'})
    data = data.drop(columns=['is_unsafe'])
    print(data.info())
    return data