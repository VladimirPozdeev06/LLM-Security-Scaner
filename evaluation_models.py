import pandas as pd
import re
from sklearn.metrics import accuracy_score, classification_report,f1_score,precision_score,recall_score
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


from evaluate import load

from bert_score import score
import time
from prepare_data_for_dpo import extract_response_and_analysis, extract_analysis_fields
from prompts_classifier import predict_batch

def prepare_data_to_necessary_from_for_evaluation(path:str='final_test_data/test_data_final_all_models.csv',text_column:str='text')->pd.DataFrame:
    data = pd.read_csv('/content/drive/MyDrive/test_data_final_all_models.csv', index_col=0)
    data = data.reset_index(drop=True)
    data['response_part'], data['analysis_part'] = zip(
        *data[text_column].apply(extract_response_and_analysis)
    )
    fields = data['analysis_part'].apply(extract_analysis_fields)
    fields_df = pd.DataFrame(fields.tolist(), index=data.index)
    data = data.join(fields_df)
    if 'is_unsafe' in data.columns:
        data = data.drop(columns=['is_unsafe'])
    print(data.info())
    return data

def compute_format_metrics(data:pd.DataFrame)->None:
    format_compliance = 1 - round(len(data[data['analysis_part'].isna()]) / len(data), 3)
    print(f'Наличие аналитической части:{format_compliance:.1%}')
    data['all_analysis'] = (
            data['response_part'].notna() &
            data['analysis_part'].notna() &
            data['is_unsafe_prompt'].notna() &
            data['attack_type'].notna() &
            data['confidence'].notna() &
            data['recommendation'].notna()
    ).astype(int)
    print(f'Полное соответствие формату {data['all_analysis'].mean():.1%}', )

def compute_numeric_metrics(data:pd.DataFrame):
    data_calssification_metrics = data[data['is_unsafe_prompt'].notna()].reset_index(drop=True)
    data_calssification_metrics['is_unsafe_prompt'] = data_calssification_metrics['is_unsafe_prompt'].astype('int')
    y_true = data_calssification_metrics['is_unsafe_prompt_real']
    y_pred = data_calssification_metrics['is_unsafe_prompt']

    accuracy = round(accuracy_score(y_true, y_pred), 3)
    print(f'Acccuracy: {accuracy}')

    f1 = round(f1_score(y_true, y_pred, average='binary'), 3)
    print(f'F1: {f1}')

    precision = round(precision_score(y_true, y_pred, average='binary'), 3)
    print(f'Precision: {precision}')

    recall = round(recall_score(y_true, y_pred, average='binary'), 3)
    print(f'Recall: {recall}')

    # print(classification_report(data_calssification_metrics['is_unsafe_prompt_real'],data_calssification_metrics['is_unsafe_prompt']))


def compute_safety_metrics(data:pd.DataFrame,model,tokenizer,return_confidence:bool=False):
    results = predict_batch(data['response_part'].tolist(), model, tokenizer,
                            target_prediction_name='is_unsafe_response')
    results = pd.DataFrame(results)
    print()
    print(f'Процент плохих ответов: {results['is_unsafe_response'].sum() / len(results):.3%}')
    print(f'Среднее качество ответов: {results['confidence_chosen_class'].mean()}')
    if return_confidence:
        return results['confidence_chosen_class']
    return data
def compute_text_metrics(data:pd.DataFrame)->None:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = load('rouge')
    bleu = load('bleu')

    results_rouge = rouge.compute(
        predictions=data['response_part'].tolist(),
        references=data['response_part_real'].tolist()
    )

    print(f"ROUGE-1: {results_rouge['rouge1']:.4f}")
    print(f"ROUGE-2: {results_rouge['rouge2']:.4f}")
    print(f"ROUGE-L: {results_rouge['rougeL']:.4f}")

    results_bleu = bleu.compute(
        predictions=data['response_part'].tolist(),
        references=[[ref] for ref in data['response_part_real'].tolist()]
    )

    print(f"BLEU: {results_bleu['bleu']:.4f}")


def compute_semantic_metrics(data:pd.DataFrame)->None:
    P, R, F1 = score(
        data['response_part'].tolist(),
        data['response_part_real'].tolist(),
        model_type='bert-base-uncased',
        lang='en',
        verbose=True,
        device='cuda'

    )
    print(f"BERTScore Precision: {P.mean().item():.4f}")
    print(f"BERTScore Recall:    {R.mean().item():.4f}")
    print(f"BERTScore F1:        {F1.mean().item():.4f}")

def compute_quantitative_metrics(data:pd.DataFrame)->None:
    print(f"Средняя длина ответа: {data['response_part'].str.len().mean()}")

def compute_all_metrics_responses(test_data:pd.DataFrame,path_data:str,
                                 is_format_metrics:bool=True,
                                 is_numeric_metrics:bool=True,
                                 is_safety_metrics:bool=True,
                                 return_confidence:bool=False,
                                 is_text_metrics:bool=True,
                                 is_semantic_metrics:bool=True,
                                 is_quantitative_metrics:bool=True):
    responses=prepare_data_to_necessary_from_for_evaluation(path_data,text_column='0')
    responses = responses.rename(columns={'0': 'response'})
    data = pd.concat([test_data, responses], axis=1)
    if is_format_metrics:
        compute_format_metrics(data)
    if is_numeric_metrics:
        compute_numeric_metrics(data)
    if is_safety_metrics:
        if return_confidence:
            return compute_safety_metrics(data,return_confidence)
        else: compute_safety_metrics(data)
    if is_text_metrics:
        compute_text_metrics(data)
    if is_semantic_metrics:
        compute_semantic_metrics(data)
    if is_quantitative_metrics:
        compute_quantitative_metrics(data)





