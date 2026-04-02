from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import contextmanager

from create_hybrid_system import get_models, hybrid_system

_classifier = None
_clf_tokenizer = None
_llm_model = None
_llm_tokenizer = None
@contextmanager
def lifespan(app:FastAPI):
    global _classifier, _clf_tokenizer, _llm_model, _llm_tokenizer
    print('Load models')
    _classifier, _clf_tokenizer, _llm_model, _llm_tokenizer = get_models()

    yield
    print('END')
app = FastAPI(
    title="LLM Security Scanner",
    description="Hybrid safety classifier + generative model pipeline",
    version="1.0.0",
    lifespan=lifespan
)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    threshold: float = 0.85




@app.post('/analyze')
def analyze(prompt_request: PromptRequest):
    prompt=prompt_request.prompt
    if not (prompt.strip()):
        return HTTPException(status_code=400,detail='Empty prompt')

    max_new_tokens=prompt_request.max_new_tokens
    threshold=prompt_request.threshold

    response = hybrid_system(
        prompt,
        _clf_tokenizer,
        _classifier,
        _llm_tokenizer,
        _llm_model,
        threshold,
        max_new_tokens
    )
    return response

@app.get('/healthcheck')
def health():
    return {
        'status':'ok',
        'models_loaded': _classifier is not None
    }
