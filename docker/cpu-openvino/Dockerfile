FROM ghcr.io/huggingface/optimum-benchmark:latest-cpu

WORKDIR /workspace

# Copy the entire "optimum-benchmark" directory into the container
COPY optimum-benchmark /workspace/optimum-benchmark

# Install the package in editable mode with extras
RUN pip install -e /workspace/optimum-benchmark[openvino]

WORKDIR /workspace

COPY setup.py .
# COPY pyproject.toml .

RUN pip install -e .[openvino]

COPY . .

ENV DEBUG_MODE=1

CMD ["llm-perf", "run-benchmark", "--hardware", "cpu", "--backend", "openvino"]
