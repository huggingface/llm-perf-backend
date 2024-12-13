# LLM-perf Backend 🏋️

The official backend system powering the [LLM-perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard). This repository contains the infrastructure and tools needed to run standardized benchmarks for Large Language Models (LLMs) across different hardware configurations and optimization backends.

## About 📝

LLM-perf Backend is designed to:
- Run automated benchmarks for the LLM-perf leaderboard
- Ensure consistent and reproducible performance measurements
- Support multiple hardware configurations and optimization backends
- Generate standardized performance metrics for latency, throughput, memory usage, and energy consumption

## Key Features 🔑

- Standardized benchmarking pipeline using [Optimum-Benchmark](https://github.com/huggingface/optimum-benchmark)
- Support for multiple hardware configurations (CPU, GPU)
- Multiple backend implementations (PyTorch, Onnxruntime, etc.)
- Automated metric collection:
  - Latency and throughput measurements
  - Memory usage tracking
  - Energy consumption monitoring
  - Quality metrics integration with Open LLM Leaderboard

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/huggingface/llm-perf-backend
cd llm-perf-backend
```

2. Create a python env
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the package with required dependencies:
```bash
pip install -e "." 
# or
pip install -e ".[all]" # to install optional dependency like Onnxruntime
```

## Usage 📋

### Command Line Interface

Run benchmarks using the CLI tool:

```bash
llm-perf run-benchmark --hardware cpu --backend pytorch
```

### Configuration Options

View all the options with
```bash
llm-perf run-benchmark --help
```

- `--hardware`: Target hardware platform (cpu, cuda)
- `--backend`: Backend framework to use (pytorch, onnxruntime, etc.)

## Benchmark Dataset 📊

Results are published to the official dataset:
[optimum-benchmark/llm-perf-leaderboard](https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard)

## Benchmark Specifications 📑

All benchmarks follow these standardized settings:
- Single GPU usage to avoid communication-dependent results
- Energy monitoring via CodeCarbon
- Memory tracking:
  - Maximum allocated memory
  - Maximum reserved memory
  - Maximum used memory (via PyNVML for GPU)