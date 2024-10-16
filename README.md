# llm-perf-backend
The backend of [the LLM-perf leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)

## Why
this runs all the benchmarks to get results for the leaderboard

## How to install
git clone 
pip install -e .[openvino]

## How to use the cli 
llm-perf run-benchmark --hardware cpu --backend openvino
llm-perf run-benchmark --hardware cpu --backend pytorch

https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard
