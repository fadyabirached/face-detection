FROM python:3.11-slim

# System libs OpenCV needs — without these you get "ImportError: libGL.so.1"
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, in its own layer.
# Without --index-url this pulls the CUDA build and the image balloons past 6GB.
RUN pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the VGGFace2 weights into the image (~110MB).
# Otherwise every cold start re-downloads them and the container takes minutes to boot.
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"

COPY . .

ENV SERVER_NAME=0.0.0.0
ENV SERVER_PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]