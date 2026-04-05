from fastapi import FastAPI, HTTPException,Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
from create_hybrid_system import get_models, hybrid_system, DEFAULT_MAX_NEW_TOKENS, DEFAULT_CLASSIFIER_THRESHOLD



MAX_PROMPT_LENGTH = 2000



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):

    logger.info("Loading models...")
    classifier, clf_tokenizer, llm_model, llm_tokenizer = get_models()
    app.state.classifier = classifier
    app.state.clf_tokenizer = clf_tokenizer
    app.state.llm_model = llm_model
    app.state.llm_tokenizer = llm_tokenizer

    yield
    logger.info('Turning off')
app = FastAPI(
    title="LLM Security Scanner",
    description="Hybrid safety classifier + generative model pipeline",
    version="1.0.0",
    lifespan=lifespan
)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    threshold: float = DEFAULT_CLASSIFIER_THRESHOLD




@app.post('/analyze')
async def analyze(prompt_request: PromptRequest,request: Request):
    prompt=prompt_request.prompt
    if not (prompt.strip()):
        raise HTTPException(status_code=400,detail='Empty prompt')
    if len(prompt.strip()) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=400,detail='Prompt too long')
    logger.info(f"Analyzing prompt, length={len(prompt)}")
    max_new_tokens=prompt_request.max_new_tokens
    threshold=prompt_request.threshold

    response = hybrid_system(
        prompt,
        request.app.state.clf_tokenizer,
        request.app.state.classifier,
        request.app.state.llm_tokenizer,
        request.app.state.llm_model,
        threshold,
        max_new_tokens
    )
    return response

@app.get('/healthcheck')
def health(request:Request):
    return {
        'status':'ok',
        'models_loaded': request.app.state.classifier is not None and request.app.state.llm_model is not None,
    }
