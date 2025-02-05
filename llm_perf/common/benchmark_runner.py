import os
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import subprocess
import traceback
import time

from loguru import logger
from optimum_benchmark import Benchmark, BenchmarkConfig, BenchmarkReport

from llm_perf.common.utils import (
    CANONICAL_PRETRAINED_OPEN_LLM_LIST,
    OPEN_LLM_LIST,
    PRETRAINED_OPEN_LLM_LIST,
)
from llm_perf.common.memory_utils import log_memory_usage

class LLMPerfBenchmarkManager(ABC):
    def __init__(
        self,
        backend: str,
        device: str,
        subset: Optional[str] = None,
        machine: Optional[str] = None,
    ):
        self.backend = backend
        self.device = device
        self.subset = subset or os.getenv("SUBSET", None)
        self.machine = machine or os.getenv("MACHINE", None)
        
        if self.machine is None and self.subset is None:
            self.push_repo_id = (
                f"optimum-benchmark/llm-perf-{self.backend}-{self.device}-debug"
            )
            self.canonical_pretrained_open_llm_list = ["gpt2"]
            self.subset = "unquantized"
        elif self.machine is not None and self.subset is not None:
            self.push_repo_id = f"optimum-benchmark/llm-perf-{self.backend}-{self.device}-{self.subset}-{self.machine}"
        else:
            raise ValueError(
                "Either both MACHINE and SUBSET should be set for benchmarking or neither for debugging"
            )

        logger.info(f"Starting benchmark runner with backend: {self.backend}, device: {self.device}, subset: {self.subset}, machine: {self.machine}")

    @abstractmethod
    def _get_weights_configs(self, subset: str) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    @abstractmethod
    def _get_attention_configs(self) -> List[str]:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def is_benchmark_supported(self, **kwargs) -> bool:
        """
        Can be overridden by child classes to exclude unsupported configurations
        """
        return True

    @abstractmethod
    def get_list_of_benchmarks_to_run(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def run_single_benchmark_in_subprocess(self, model: str, **kwargs) -> bool:
        """Run a single benchmark in a separate process"""
        try:
            # Create the Python script to run in subprocess
            script = f"""
import sys
import os
from {self.__class__.__module__} import {self.__class__.__name__}
from loguru import logger
import traceback

try:
    runner = {self.__class__.__name__}()
    
    runner.run_benchmark(model="{model}", **{kwargs})
    sys.exit(0)
except Exception:
    logger.error("Error in subprocess:" + "\\n" + traceback.format_exc())
    sys.exit(1)
"""

            # Run the subprocess with timeout
            result = subprocess.run(
                [sys.executable, "-c", script],
                text=True,
                env={**os.environ, 
                     "PYTHONUNBUFFERED": "1",
                     "LOG_TO_FILE": "0"  # Disable file logging for optimum-benchmark
                },
                timeout=3600  # 1 hour timeout
            )
            
            return result.returncode == 0
        
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out for model {model}")
            return False
        except Exception:
            logger.error("Failed to run benchmark process:" + "\n" + traceback.format_exc())
            return False

    def run_benchmarks(self):
        """Run all benchmarks sequentially with process isolation"""
        benchmarks_to_run = self.get_list_of_benchmarks_to_run()
                
        logger.info(
            f"Running a total of {len(benchmarks_to_run)} benchmarks, "
            f"with {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)} models"
        )
        
        logger.info(f"Models that are being benchmarked: {CANONICAL_PRETRAINED_OPEN_LLM_LIST}")
        
        rerun_already_conducted_benchmarks = os.getenv("RERUN_ALREADY_CONDUCTED_BENCHMARKS", "false") == "true"

        total_benchmarks = len(benchmarks_to_run)
        completed_benchmarks = 0
        failed_benchmarks = 0
        skipped_benchmarks = 0
        failed_models = []
        start_time = time.time()

        for benchmark_config in benchmarks_to_run:
            try:
                # Log memory before benchmark
                logger.info("Memory usage before benchmark:")
                log_memory_usage("before")
                
                model = benchmark_config.pop("model")  # Remove model from kwargs
                benchmark_name = self.get_benchmark_name(model, **benchmark_config)
                subfolder = f"{benchmark_name}/{model.replace('/', '--')}"
                
                if not rerun_already_conducted_benchmarks:
                    if self.is_benchmark_conducted(self.push_repo_id, subfolder):
                        logger.info(f"Skipping already conducted benchmark: {benchmark_name}")
                        benchmark_config["model"] = model  # Restore model key
                        completed_benchmarks += 1
                        skipped_benchmarks += 1
                        success_rate = ((completed_benchmarks - failed_benchmarks) / completed_benchmarks) * 100 if completed_benchmarks > 0 else 100
                        logger.info(f"Progress: {completed_benchmarks}/{total_benchmarks} benchmarks completed ({(completed_benchmarks/total_benchmarks)*100:.1f}%) - Current success rate: {success_rate:.1f}%")
                        continue
                
                logger.info(f"Starting benchmark for model {model} with config: {benchmark_config}")
                
                # Run the benchmark in a separate process
                success = self.run_single_benchmark_in_subprocess(
                    model=model,
                    **benchmark_config
                )
                
                if not success:
                    logger.error(f"Benchmark failed for model {model}")
                    failed_benchmarks += 1
                    failed_models.append(model)
                
                completed_benchmarks += 1
                success_rate = ((completed_benchmarks - failed_benchmarks) / completed_benchmarks) * 100 if completed_benchmarks > 0 else 100
                logger.info(f"Progress: {completed_benchmarks}/{total_benchmarks} benchmarks completed ({(completed_benchmarks/total_benchmarks)*100:.1f}%) - Current success rate: {success_rate:.1f}%")
                
                # Log memory after benchmark
                logger.info("Memory usage after benchmark:")
                log_memory_usage("after")
                
            except Exception as e:
                logger.error(f"Failed to run benchmark for {model}: {str(e)}")
                logger.error(traceback.format_exc())
                failed_benchmarks += 1
                failed_models.append(model)
            finally:
                # Restore model key in case the config is reused
                benchmark_config["model"] = model

        # Calculate execution time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        # Print summary
        logger.info("\n" + "="*50)
        logger.info("BENCHMARK EXECUTION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total execution time: {hours}h {minutes}m {seconds}s")
        logger.info(f"Total benchmarks: {total_benchmarks}")
        logger.info(f"Successfully completed: {completed_benchmarks - failed_benchmarks}")
        logger.info(f"Failed: {failed_benchmarks}")
        logger.info(f"Skipped (already conducted): {skipped_benchmarks}")
        logger.info(f"Success rate: {((completed_benchmarks - failed_benchmarks) / total_benchmarks) * 100:.1f}%")
        
        if failed_models:
            logger.info("\nFailed models:")
            for model in failed_models:
                logger.info(f"  - {model}")
        
        logger.info("\nConfiguration:")
        logger.info(f"  Backend: {self.backend}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Subset: {self.subset}")
        logger.info(f"  Machine: {self.machine}")
        logger.info(f"  Rerun already conducted: {rerun_already_conducted_benchmarks}")
        logger.info("="*50 + "\n")

    def is_benchmark_conducted(self, push_repo_id, subfolder):
        try:
            report = BenchmarkReport.from_pretrained(
                repo_id=push_repo_id, subfolder=subfolder
            )
            if "traceback" in report.to_dict():
                return False
            else:
                return True
        except Exception:
            return False

    @abstractmethod
    def get_benchmark_name(self, model: str, **kwargs) -> str:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def run_benchmark(self, rerun_already_conducted_benchmarks: bool = False, **kwargs):
        model = kwargs.pop("model")

        benchmark_name = self.get_benchmark_name(model, **kwargs)
        subfolder = f"{benchmark_name}/{model.replace('/', '--')}"

        if not self.is_benchmark_supported(**kwargs):
            logger.info(
                f"Skipping benchmark {benchmark_name} with model {model} since it is not supported"
            )
            return

        if not rerun_already_conducted_benchmarks and self.is_benchmark_conducted(self.push_repo_id, subfolder):
            logger.info(
                f"Skipping benchmark {benchmark_name} with model {model} since it was already conducted"
            )
            return

        benchmark_config = self.get_benchmark_config(model, **kwargs)
        benchmark_config.push_to_hub(
            repo_id=self.push_repo_id, subfolder=subfolder, private=True
        )
        self.execute_and_log_benchmark(benchmark_config, subfolder)

    @abstractmethod
    def get_benchmark_config(self, model: str, **kwargs) -> BenchmarkConfig:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def execute_and_log_benchmark(
        self, benchmark_config: BenchmarkConfig, subfolder: str
    ):
        try:
            logger.info("Memory usage before execution:")
            log_memory_usage("before")
            
            logger.info(
                f"Running benchmark {benchmark_config.name} with model {benchmark_config.backend.model}"
            )
            benchmark_report = Benchmark.launch(benchmark_config)
            benchmark_report.push_to_hub(
                repo_id=self.push_repo_id, subfolder=subfolder, private=True
            )
            benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
            benchmark.push_to_hub(
                repo_id=self.push_repo_id, subfolder=subfolder, private=True
            )
            
            logger.info("Memory usage after execution:")
            log_memory_usage("after")
            
        except Exception as e:
            logger.error(
                f"Benchmark {benchmark_config.name} failed with model {benchmark_config.backend.model}, error:\n{e}"
            )
            benchmark_report = BenchmarkReport.from_dict(
                {"traceback": traceback.format_exc()}
            )
            benchmark_report.push_to_hub(
                repo_id=self.push_repo_id, subfolder=subfolder, private=True
            )
            benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
            benchmark.push_to_hub(
                repo_id=self.push_repo_id, subfolder=subfolder, private=True
            )
