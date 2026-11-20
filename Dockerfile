# Inference image. CPU by default; build with
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
# for a CUDA image (Cloud Run with an L4 GPU, or Vertex AI).
#
#   docker build -t brats-seg .
#   docker run -p 8080:8080 brats-seg
#   curl -F flair=@x_flair.nii.gz -F t1=@x_t1.nii.gz -F t1ce=@x_t1ce.nii.gz \
#        -F t2=@x_t2.nii.gz "localhost:8080/predict?output=json"
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY serve/ serve/
# Checkpoints are not in git; build from a checkout where they exist, or mount
# them and set MODEL_CHECKPOINTS. The build fails loudly if none are present.
COPY checkpoints/nnunet_style_best.pt checkpoints/

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
