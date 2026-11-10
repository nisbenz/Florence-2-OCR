# Stage 1: export. PyTorch lives here only, to turn the original weights into ONNX graphs.
FROM python:3.12-slim AS export
WORKDIR /build
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY export/requirements.txt export/
RUN pip install --no-cache-dir -r export/requirements.txt
COPY export/ export/
RUN cd export && python export_encoders.py --out /models && python export_decoder.py --out /models \
    && python quantize.py --models /models \
    && rm -rf /root/.cache/huggingface

# Stage 2: server. No torch: only ONNX Runtime and the server dependencies.
FROM python:3.12-slim
WORKDIR /app
COPY server/requirements.txt server/
RUN pip install --no-cache-dir -r server/requirements.txt
COPY server/ server/
COPY --from=export /models /models
ENV MODELS_DIR=/models PYTHONUNBUFFERED=1
EXPOSE 8000
WORKDIR /app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
