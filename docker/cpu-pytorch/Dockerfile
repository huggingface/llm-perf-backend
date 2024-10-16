FROM ghcr.io/huggingface/optimum-benchmark:latest-cpu

WORKDIR /workspace

COPY . .

RUN pip install -e .

CMD ["llm-perf", "run-benchmark", "--hardware", "cpu", "--backend", "pytorch"]
