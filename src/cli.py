import typer
import os
import subprocess
from enum import Enum
from dotenv import load_dotenv

from src.benchmark_runners.cpu.update_llm_perf_cpu_onnxruntime import CPUOnnxRuntimeBenchmarkRunner
from src.benchmark_runners.cpu.update_llm_perf_cpu_pytorch import CPUPyTorchBenchmarkRunner
from src.benchmark_runners.cpu.update_llm_perf_cpu_openvino import CPUOpenVINOBenchmarkRunner
from src.benchmark_runners.cuda.update_llm_perf_cuda_pytorch import CUDAPyTorchBenchmarkRunner


import warnings
if os.environ.get("DISABLE_WARNINGS", "0") == "1":
    warnings.filterwarnings("ignore")

app = typer.Typer()

class Hardware(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"

class Backend(str, Enum):
    ONNXRUNTIME = "onnxruntime"
    PYTORCH = "pytorch"
    OPENVINO = "openvino"

@app.command()
def run_benchmark(
    hardware: Hardware = typer.Option(..., help="Hardware to run on: CPU or CUDA"),
    backend: Backend = typer.Option(..., help="Backend to use: ONNXRUNTIME, PYTORCH, or OPENVINO"),
):
    env_vars = load_dotenv()
    if env_vars:
        print("Environment variables loaded successfully")
    else:
        print("No environment variables loaded")

    if hardware == Hardware.CPU:
        if backend == Backend.ONNXRUNTIME:
            runner = CPUOnnxRuntimeBenchmarkRunner()
        elif backend == Backend.PYTORCH:
            runner = CPUPyTorchBenchmarkRunner()
        elif backend == Backend.OPENVINO:
            runner = CPUOpenVINOBenchmarkRunner()
    elif hardware == Hardware.CUDA:
        if backend == Backend.PYTORCH:
            runner = CUDAPyTorchBenchmarkRunner()
        else:
            typer.echo(f"CUDA is not supported for {backend} backend")
            raise typer.Exit(code=1)
        
    runner.run_benchmarks()

if __name__ == "__main__":
    app()
