FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

ARG DEVICE=cpu
RUN pip3 install --no-cache-dir --upgrade pip \
    if [ "$DEVICE" = "gpu" ]; then \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    else \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    && pip3 install --no-cache-dir -r requirements.txt


COPY app.py .
COPY create_hybrid_system.py .
COPY utils.py .
COPY prompts_classifier ./prompts_classifier
COPY dpo_model_extended ./dpo_model_extended

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]