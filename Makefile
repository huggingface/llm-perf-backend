# Load environment variables
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Style and Quality checks
.PHONY: style quality run_cpu_container run_cuda_container run_rocm_container cpu-pytorch-container cpu-openvino-container collector-container help

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

# Running optimum-benchmark containers
run-optimum-benchmark-cpu-container:
	docker run -it --rm --pid host --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-cpu

run-optimum-benchmark-cuda-container:
	docker run -it --rm --pid host --gpus all --shm-size 64G --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-cuda

run-optimum-benchmark-rocm-container:
	docker run -it --rm --shm-size 64G --device /dev/kfd --device /dev/dri --volume .:/llm-perf-backend --workdir /llm-perf-backend ghcr.io/huggingface/optimum-benchmark:latest-rocm

# Running llm-perf-leaderboard benchmarks
run-llm-perf-benchmark-cpu-pytorch:
	docker build -t llm-perf-backend-cpu-pytorch -f docker/cpu-pytorch/Dockerfile .
	docker run -it --rm --pid host llm-perf-backend-cpu-pytorch

run-llm-perf-benchmark-cpu-openvino:
	docker build -t llm-perf-backend-cpu-openvino -f docker/cpu-openvino/Dockerfile .
	docker run -it --rm --pid host llm-perf-backend-cpu-openvino

run-llm-perf-benchmark-cuda-pytorch:
	docker build -t llm-perf-backend-cuda-pytorch -f docker/gpu-cuda/Dockerfile .
	docker run -it --rm --pid host --gpus all --shm-size 64G --volume .:/llm-perf-backend --workdir /llm-perf-backend llm-perf-backend-cuda-pytorch

run-llm-perf-benchmark-collector:
	docker build -t llm-perf-backend-collector -f docker/collector/Dockerfile .
	docker run -it --rm --pid host llm-perf-backend-collector

help:
	@echo "Commands:"
	@echo "  style                    - Format code and fix style issues"
	@echo "  quality                  - Run style checks without fixing"
	@echo ""
	@echo "Optimum Benchmark Containers:"
	@echo "  run-optimum-benchmark-cpu-container   - Run CPU container"
	@echo "  run-optimum-benchmark-cuda-container  - Run CUDA container"
	@echo "  run-optimum-benchmark-rocm-container  - Run ROCm container"
	@echo ""
	@echo "LLM Performance Backend Containers:"
	@echo "  run-llm-perf-benchmark-cpu-pytorch   - Run the llm-perf-leaderboard Benchmark CPU PyTorch"
	@echo "  run-llm-perf-benchmark-cpu-openvino  - Run the llm-perf-leaderboard Benchmark CPU OpenVINO"
	@echo "  run-llm-perf-benchmark-cuda-pytorch  - Run the llm-perf-leaderboard Benchmark CUDA PyTorch"
	@echo "  run-llm-perf-benchmark-collector     - Run the llm-perf-leaderboard Collector container"

