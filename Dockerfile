FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar ffmpeg 5.1 (compatible 100% con PyAV 11.0.0) y sus librerías de desarrollo
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    pkg-config \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .

# Forzamos las versiones de CPU para evitar la descarga de CUDA
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=user app.py .
EXPOSE 7860
CMD ["python", "app.py"]
