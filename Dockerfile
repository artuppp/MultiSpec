FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# ── Sistema base ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        wget curl git \
        v4l-utils libv4l-dev \
        gphoto2 libgphoto2-dev \
        ffmpeg \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
RUN /opt/conda/bin/pip install --no-cache-dir \
        flask \
        opencv-python-headless \
        "numpy<2.0.0" \
        "scipy<1.13.0"

# Instalamos basicsr y realesrgan ignorando sus dependencias estrictas de numpy
RUN /opt/conda/bin/pip install --no-cache-dir --no-deps \
        basicsr \
        realesrgan

RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /opt/conda/lib/python3.10/site-packages/basicsr/data/degradations.py

# ── Precarga modelo Real-ESRGAN en tiempo de build ────────────────────────────
RUN mkdir -p /app && /opt/conda/bin/python -c "\
from basicsr.archs.rrdbnet_arch import RRDBNet; \
from realesrgan import RealESRGANer; \
import torch; \
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4); \
upsampler = RealESRGANer( \
    scale=4, \
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', \
    model=model, \
    tile=0, \
    half=torch.cuda.is_available() \
); \
print('Modelo Real-ESRGAN descargado y listo')" && \
    find /root -name "RealESRGAN_x4plus.pth" -exec cp {} /app/ \; 2>/dev/null || true

# ── Aplicación ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY server.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/status || exit 1

CMD ["python3", "server.py"]