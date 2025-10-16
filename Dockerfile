FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod diffusers transformers accelerate safetensors pillow xformers torch torchvision requests

WORKDIR /app
COPY app.py /app/app.py

CMD ["python", "-u", "app.py"]
