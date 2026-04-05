import torch
from transformers import BitsAndBytesConfig, AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from utils import extract_response_and_analysis,extract_analysis_fields,load_pretrained_classification_model

import os
from dotenv import load_dotenv
load_dotenv()
CLASSIFIER_PATH='prompts_classifier'
BASE_MODEL = "Qwen/Qwen3-4B"
DPO_EXTENDED_PATH='dpo_model_extended'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a security-focused AI assistant. "
    "Always respond in English. "
    "After your response, add analysis in this exact format:\n"
    "Analysis: [explanation]; is_unsafe: [0 or 1]; "
    "attack_type: [type]; confidence: [high/medium/low]; "
    "Recommendation: [SAFE/REVIEW/BLOCK]"
)
HF_TOKEN=os.getenv('HF_TOKEN')

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
def load_llm(base_model_path,peft_path:str,token,device:str='cpu'):
    tokenizer=AutoTokenizer.from_pretrained(base_model_path,token=token,trust_remote_code=True)
    if device=='cuda':
        base_model=AutoModelForCausalLM.from_pretrained(base_model_path,
                                                    quantization_config=BNB_CONFIG,
                                                    dtype=torch.float16,
                                                    device_map='auto',
                                                    token=token
                                                    )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float32,
            token=token
        )
    model=PeftModel.from_pretrained(base_model,peft_path)
    model.eval()
    return model,tokenizer

def predict_classifier(prompt,model,tokenizer):
    inputs=tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512

    )

    with torch.no_grad():
        output=model(**inputs)
        probs=torch.softmax(output.logits,dim=1)
        prediction=torch.argmax(output.logits).item()

    return {
        "prompt":prompt,
        'is_unsafe_prompt':prediction,
        'predicted_confidence': (probs[0][0] if prediction==0 else probs[0][1]).item()
    }
def generate_response(prompt,model,tokenizer,max_new_tokens:int=512):
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs=tokenizer(text,return_tensors='pt').to(model.device)
    with torch.no_grad():
        output=model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    new_tokens = output[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def hybrid_system(prompt,
                  classifier_tokenizer,classifier_model,
                  llm_tokenizer,llm_model,
                  threshold:float=0.85,
                  max_new_tokens:int=512):
    predictions_safety_results=predict_classifier(prompt,classifier_model,classifier_tokenizer)
    if predictions_safety_results['is_unsafe_prompt']==1 and predictions_safety_results['predicted_confidence']>threshold:
        return {
            "prompt": prompt,
            "response": "I cannot assist with this request as it has been identified as potentially harmful.",
            "analysis": (
                "Analysis: Blocked by safety classifier before generation.; "
                "is_unsafe: 1; attack_type: detected_by_classifier; "
                "confidence: high; Recommendation: BLOCK"
            ),
            'is_unsafe_prompt': 1,
            "blocked_by_classifier": True,
            'predicted_classifier_confidence': predictions_safety_results['predicted_confidence'],
        }
    response=generate_response(prompt,llm_model,llm_tokenizer,max_new_tokens)
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    response_part,analysis_part=extract_response_and_analysis(response)
    fields = extract_analysis_fields(analysis_part)

    return {
        'prompt': prompt,
        'response':response_part,
        "analysis":analysis_part,
        'is_unsafe_prompt': predictions_safety_results['is_unsafe_prompt']==1 ,
        "blocked_by_classifier": False,
        'predicted_classifier_confidence': predictions_safety_results['predicted_confidence']
    }




def get_models():
    global _classifier,_clf_tokenizer,_llm_model,_llm_tokenizer
    if _classifier is None:
        _classifier,_clf_tokenizer=load_pretrained_classification_model(CLASSIFIER_PATH,DEVICE)
    if _llm_model is None:
        _llm_model,_llm_tokenizer=load_llm(BASE_MODEL,DPO_EXTENDED_PATH,HF_TOKEN,DEVICE)
    return _classifier, _clf_tokenizer, _llm_model, _llm_tokenizer