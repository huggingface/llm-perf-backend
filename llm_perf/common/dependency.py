from dotenv import load_dotenv
import os

load_dotenv()


def is_debug_mode():
    return os.environ.get("DEBUG_MODE", "0") == "1"


def get_benchmark_top_n():
    return int(os.environ.get("BENCHMARK_TOP_N", "10"))
