import re
from transformers import AutoTokenizer,AutoModelForSequenceClassification
def extract_response_and_analysis(full_response):

    if '<|im_start|>assistant\n' in full_response:
        full_response = full_response.split('<|im_start|>assistant\n')[-1]


    full_response = full_response.replace('<|im_end|>', '').strip()

    idx = full_response.rfind("Analysis:")

    if idx == -1:
        return full_response.strip(), None

    response_part = full_response[:idx].strip()
    analysis_part = full_response[idx:].strip()
    return response_part, analysis_part

def extract_analysis_fields(text):
    result = {
        'is_unsafe_prompt': None,
        'attack_type': None,
        'confidence': None,
        'recommendation': None
    }

    patterns = {
        'is_unsafe_prompt':      r'is_unsafe:\s*(\d+)',
        'attack_type':    r'attack_type:\s*([^;]+?)(?:;|\n|$)',
        'confidence':     r'confidence:\s*(high|medium|low)',
        'recommendation': r'Recommendation:\s*(SAFE|REVIEW|BLOCK)',
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, str(text))
        if match:
            value = match.group(1).strip()
            result[field] = int(value) if field == 'is_unsafe_prompt' else value

    return result

def load_pretrained_classification_model(path_to_model:str,device:str='cpu'):
    tokenizer=AutoTokenizer.from_pretrained(path_to_model)
    model=AutoModelForSequenceClassification.from_pretrained(path_to_model)
    model.eval()
    model.to(device)
    return model, tokenizer