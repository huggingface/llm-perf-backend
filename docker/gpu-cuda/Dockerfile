FROM ghcr.io/huggingface/optimum-benchmark:latest-cuda

WORKDIR /workspace

COPY setup.py .

RUN pip install -e .[cuda] \
    && pip install flash-attn --no-build-isolation

COPY . .

ENTRYPOINT ["llm-perf", "run-benchmark", "--hardware", "cuda", "--backend", "pytorch"]
