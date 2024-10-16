# Style and Quality checks
.PHONY: style quality

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

.PHONY: install

install:
	pip install .

install-dev:
	DEBUG=1 uv pip install -e .

# Running containers
.PHONY: run_cpu_container run_cuda_container run_rocm_container

run_cpu_container:
	docker run -it --rm --pid host --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-cpu

run_cuda_container:
	docker run -it --rm --pid host --gpus all --shm-size 64G --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-cuda

run_rocm_container:
	docker run -it --rm --shm-size 64G --device /dev/kfd --device /dev/dri --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-rocm

cpu-pytorch-container:
	docker build -t cpu-pytorch -f docker/cpu-pytorch/Dockerfile .
	docker run -it --rm --pid host cpu-pytorch /bin/bash
	# docker run -it --rm --pid host 
