# ── Base: CUDA 12.1 + cuDNN 8 + Ubuntu 22.04 ─────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevents interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 \
       --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source ────────────────────────────────────────────────────────────────
COPY src/ src/
COPY experiments/ experiments/
COPY setup.py .
COPY train.py .
COPY evaluate.py .

RUN pip install -e . --no-deps

# ── Kaggle credentials mount point ────────────────────────────────────────────
# Mount your ~/.kaggle/kaggle.json at /root/.kaggle/kaggle.json at runtime
RUN mkdir -p /root/.kaggle

# ── Ports (MLflow UI) ─────────────────────────────────────────────────────────
EXPOSE 5000

# ── Default command ────────────────────────────────────────────────────────────
CMD ["python", "train.py", "--config", "experiments/configs/base_config.yaml"]
