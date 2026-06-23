from dataclasses import dataclass


@dataclass
class BenchmarkRunDetails:
    machine: str
    hardware: str
    subsets: str
    backends: str
    model: str
    success: bool
    traceback: str
    last_updated: str
    run_id: str
    run_start_time: str
