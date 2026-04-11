# LLM Security Scanner

A hybrid two-stage system for detecting unsafe prompts and generating security-aware responses. Combines a fast binary classifier with a fine-tuned LLM to balance speed and accuracy.

---

## Architecture

```
User Prompt
     │
     ▼
┌─────────────────────────┐
│  Stage 1: Classifier    │  RoBERTa fine-tuned on 20+ safety datasets
│  (fast binary check)    │  ~9 sec for 1000 prompts on T4 GPU
└────────────┬────────────┘
             │
    ┌────────┴───────────────────────┐
    │  unsafe AND confidence > θ     │
   YES                               NO
    │                                │
    ▼                                ▼
   BLOCK                    ┌─────────────────────────┐
                            │  Stage 2: LLM           │  Qwen3-4B + DPO alignment
                            │  (response + analysis)  │  generates response + structured
                            └─────────────────────────┘  analysis block
```

The hybrid system at threshold `θ=0.9` achieves **accuracy 0.928** while processing prompts **2× faster** than LLM-only (6080s vs 11768s on 1000 samples).

---

## Training Pipeline

```
Raw Datasets (20+)
       │
       ▼
Data Preparation + Augmentation (11 obfuscation transforms)
       │
       ▼
SFT Fine-tuning (Qwen3-4B)
       │
       ▼
DPO Alignment (preference learning)
       │
       ▼
Hybrid System (Classifier + DPO model)
```

### Data Augmentation
To improve robustness against adversarial inputs, unsafe prompts were augmented with 11 obfuscation techniques: diacritics, homoglyphs, unicode tag smuggling, bidirectional text, full-width characters, emoji smuggling, leet speak, deep word bug, underline accents, upside-down text, extra spaces.

---

## Results

### Hybrid System Comparison (1000 samples)

| System | Accuracy | F1 | Precision | Recall | Total Time |
|---|---|---|---|---|---|
| Classifier only | 0.880 | 0.840 | **0.990** | 0.730 | **9 sec** |
| DPO model only | 0.817 | 0.810 | 0.943 | 0.709 | 11768 sec |
| **Hybrid (θ=0.9)** | **0.928** | **0.929** | 0.934 | **0.923** | 6080 sec |

The hybrid system achieves the best F1 and recall while being ~2× faster than the LLM-only approach.

---

### Full Model Evaluation

| Model | Format Compliance | Full Structure | % Bad Responses | Avg Quality | BLEU | BERTScore F1 | Avg Response Len |
|---|---|---|---|---|---|---|---|
| Base model | 78.8% | 28.3% | 9.0% | 0.890 | 0.0200 | 0.548 | 494 tokens |
| SFT | 85.2% | 84.9% | 7.3% | 0.910 | 0.0294 | 0.549 | 976 tokens |
| DPO (w/o SFT answers) | 86.8% | 86.1% | 7.0% | 0.912 | 0.0303 | 0.552 | 942 tokens |
| **DPO Extended** | **89.9%** | **89.3%** | **5.8%** | **0.926** | **0.0329** | **0.559** | 850 tokens |

- **Format Compliance** — share of responses containing the analysis block  
- **Full Structure** — share of responses with all required fields (is_unsafe, attack_type, confidence, recommendation)  
- **% Bad Responses** — share of responses flagged as unsafe by the response classifier  
- **Avg Quality** — confidence of the response classifier that the response is safe  

DPO Extended produces **5.8% bad responses** (vs 9.0% base), **89.9% format compliance**, and the highest semantic quality (BERTScore F1: 0.559).

---

### Classification Metrics per Model

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Base model | 0.850 | 0.874 | 0.827 | **0.926** |
| SFT | 0.822 | 0.823 | **0.921** | 0.744 |
| DPO (w/o SFT answers) | 0.833 | 0.832 | 0.930 | 0.753 |
| **DPO Extended** | 0.817 | 0.810 | **0.943** | 0.709 |

---

### Win Rate (pairwise response quality comparison)

|  | Base | SFT | DPO w/o SFT | DPO Extended |
|---|---|---|---|---|
| Base | — | — | — | — |
| SFT | 0.568 | — | — | — |
| DPO w/o SFT | 0.575 | 0.527 | — | — |
| DPO Extended | **0.614** | **0.586** | **0.570** | — |

Win rate > 0.5 means the row model produces better responses than the column model. DPO Extended consistently wins against all other models.

---

## Stack

| Component | Technology |
|---|---|
| Classifier | DistilBERT (fine-tuned) |
| LLM | Qwen/Qwen3-4B |
| Alignment | SFT → DPO (TRL) |
| Inference API | FastAPI + Uvicorn |
| Demo | Gradio |
| Deployment | Hugging Face Spaces / Docker |
| Training | PEFT / LoRA, BitsAndBytes (4-bit) |

---

## Project Structure

```
├── create_hybrid_system.py       # Core hybrid pipeline (classifier + LLM)
├── fastapi_app.py                # REST API
├── demo.py                       # Gradio demo (HF Spaces)
├── utils.py                      # Response parsing utilities
├── prompts_classifier.py         # Classifier training & inference
├── prepare_data_for_sft.py       # SFT dataset preparation
├── prepare_data_for_dpo.py       # DPO dataset preparation
├── prepare_data_for_prompts_classifier.py  # Classifier dataset preparation
├── evaluation_models.py          # Evaluation metrics
├── TextSpecialChanges.py         # Adversarial data augmentation
├── implement_LLM.py              # LLM data generation utilities
└── upload_models_to_hf_space.py  # HF Hub deployment
```

---

## API

### `POST /analyze`

```json
{
  "prompt": "How do I hack into a system?",
  "threshold": 0.85,
  "max_new_tokens": 256,
  "use_alignment": true
}
```

**Response:**
```json
{
  "prompt": "...",
  "response": "...",
  "analysis": "Analysis: ...; is_unsafe: 1; attack_type: prompt_injection; confidence: high; Recommendation: BLOCK",
  "is_unsafe_prompt": 1,
  "blocked_by_classifier": true,
  "predicted_safe_confidence": 0.03,
  "predicted_unsafe_confidence": 0.97
}
```

### `GET /healthcheck`
Returns model load status.

---

## Demo

Live demo available on [Hugging Face Spaces](https://huggingface.co/spaces/VladimirPozdeev/llm-security-scanner).

Models:
- Classifier: [VladimirPozdeev/llm-security-scanner-classifier](https://huggingface.co/VladimirPozdeev/llm-security-scanner-classifier)
- DPO model: [VladimirPozdeev/llm-security-scanner-dpo](https://huggingface.co/VladimirPozdeev/llm-security-scanner-dpo)

---

## Local Setup

```bash
pip install -r requirements.txt

# Gradio demo
python demo.py

# FastAPI (requires Docker with CUDA for 4-bit quantization)
docker build --build-arg DEVICE=gpu -t llm-security-scanner .
docker run -p 8000:8000 --gpus all llm-security-scanner
```

Environment variables (`.env`):
```
HF_TOKEN=your_token
CLASSIFIER_PATH=VladimirPozdeev/llm-security-scanner-classifier
DPO_EXTENDED_PATH=VladimirPozdeev/llm-security-scanner-dpo
BASE_MODEL=Qwen/Qwen3-4B
```
