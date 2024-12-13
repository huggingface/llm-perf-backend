# Style and Quality checks
.PHONY: style quality install install-dev run_cpu_container run_cuda_container run_rocm_container cpu-pytorch-container cpu-openvino-container collector-container

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

install:
	pip install .

install-dev:
	DEBUG=1 uv pip install -e .

# Running optimum-benchmark containers
run_cpu_container:
	docker run -it --rm --pid host --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-cpu

run_cuda_container:
	docker run -it --rm --pid host --gpus all --shm-size 64G --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-cuda

run_rocm_container:
	docker run -it --rm --shm-size 64G --device /dev/kfd --device /dev/dri --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-rocm

# Running llm-perf backend containers
cpu-pytorch-container:
	docker build -t cpu-pytorch -f docker/cpu-pytorch/Dockerfile .
	docker run -it --rm --pid host cpu-pytorch

cpu-openvino-container:
	docker build -t cpu-openvino -f docker/cpu-openvino/Dockerfile .
	docker run -it --rm --pid host cpu-openvino

collector-container:
	docker build -t collector -f docker/collector/Dockerfile .
	docker run -it --rm --pid host collector
