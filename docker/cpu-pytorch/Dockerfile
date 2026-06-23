FROM ghcr.io/huggingface/optimum-benchmark:latest-cpu

WORKDIR /workspace

COPY setup.py .

RUN pip install -e .

COPY . .

ENTRYPOINT ["llm-perf", "run-benchmark", "--hardware", "cpu", "--backend", "pytorch"]
