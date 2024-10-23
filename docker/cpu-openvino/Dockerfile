FROM ghcr.io/huggingface/optimum-benchmark:latest-cpu

WORKDIR /workspace

COPY setup.py .
# COPY pyproject.toml .

RUN pip install -e .[openvino]

COPY . .

CMD ["llm-perf", "run-benchmark", "--hardware", "cpu", "--backend", "openvino"]
