FROM ghcr.io/huggingface/optimum-benchmark:latest-cpu

WORKDIR /workspace

COPY setup.py .

RUN pip install -e .[openvino]

COPY . .

ENTRYPOINT ["llm-perf", "run-benchmark", "--hardware", "cpu", "--backend", "openvino"]
