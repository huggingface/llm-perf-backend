import subprocess
from glob import glob

import pandas as pd
from huggingface_hub import create_repo, snapshot_download, upload_file, repo_exists
from optimum_benchmark import Benchmark
import json

from llm_perf.common.hardware_config import load_hardware_configs
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

REPO_TYPE = "dataset"
MAIN_REPO_ID = "optimum-benchmark/llm-perf-leaderboard"
PERF_REPO_ID = "optimum-benchmark/llm-perf-{backend}-{hardware}-{subset}-{machine}"

PERF_DF = "perf-df-{backend}-{hardware}-{subset}-{machine}.csv"
LLM_DF = "llm-df.csv"


def patch_json(file):
    """
    Patch a JSON file by adding a 'stdev_' key with the same value as 'stdev' for all occurrences,
    but only if 'stdev_' doesn't already exist at the same level.
    This is to make the old optimum benchmark compatible with the new one.

    This function reads a JSON file, recursively traverses the data structure,
    and for each dictionary that contains a 'stdev' key without a corresponding 'stdev_' key,
    it adds a 'stdev_' key with the same value. The modified data is then written back to the file.

    Args:
        file (str): The path to the JSON file to be patched.

    Returns:
        None
    """
    with open(file, "r") as f:
        data = json.load(f)

    def add_stdev_(obj):
        if isinstance(obj, dict):
            new_items = []
            for key, value in obj.items():
                if key == "stdev" and "stdev_" not in obj:
                    new_items.append(("stdev_", value))
                if isinstance(value, (dict, list)):
                    add_stdev_(value)
            for key, value in new_items:
                obj[key] = value
        elif isinstance(obj, list):
            for item in obj:
                add_stdev_(item)

    add_stdev_(data)

    with open(file, "w") as f:
        json.dump(data, f, indent=4)


def gather_benchmarks(subset: str, machine: str, backend: str, hardware: str):
    """
    Gather the benchmarks for a given machine
    """
    perf_repo_id = PERF_REPO_ID.format(
        subset=subset, machine=machine, backend=backend, hardware=hardware
    )
    snapshot = snapshot_download(
        repo_type=REPO_TYPE,
        repo_id=perf_repo_id,
        allow_patterns=["**/benchmark.json"],
    )

    dfs = []
    for file in glob(f"{snapshot}/**/benchmark.json", recursive=True):
        patch_json(file)
        dfs.append(Benchmark.from_json(file).to_dataframe())
    benchmarks = pd.concat(dfs, ignore_index=True)

    perf_df = PERF_DF.format(
        subset=subset, machine=machine, backend=backend, hardware=hardware
    )
    benchmarks.to_csv(perf_df, index=False)
    create_repo(repo_id=MAIN_REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)
    upload_file(
        repo_id=MAIN_REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo=perf_df,
        path_or_fileobj=perf_df,
    )
    print(f"Uploaded {perf_df} to {MAIN_REPO_ID}")


# def check_if_url_exists(url: str):
#     """
#     Check if a URL exists
#     """
#     repo_exists
#     print(f"response: {response}")
#     return response.status_code == 200


def update_perf_dfs():
    """
    Update the performance dataframes for all machines
    """
    hardware_configs = load_hardware_configs("llm_perf/hardware.yaml")

    for hardware_config in hardware_configs:
        for subset in hardware_config.subsets:
            for backend in hardware_config.backends:
                try:
                    gather_benchmarks(
                        subset,
                        hardware_config.machine,
                        backend,
                        hardware_config.hardware,
                    )
                except Exception:
                    print("Dataset not found for:")
                    print(f"  • Backend: {backend}")
                    print(f"  • Subset: {subset}")
                    print(f"  • Machine: {hardware_config.machine}")
                    print(f"  • Hardware Type: {hardware_config.hardware}")
                    url = f"{PERF_REPO_ID.format(subset=subset, machine=hardware_config.machine, backend=backend, hardware=hardware_config.hardware)}"

                    does_exist = repo_exists(url, repo_type="dataset")

                    if does_exist:
                        print(f"Dataset exists: {url} but could not be processed")


scrapping_script = """
git clone https://github.com/Weyaxi/scrape-open-llm-leaderboard.git
pip install -r scrape-open-llm-leaderboard/requirements.txt -q
python scrape-open-llm-leaderboard/main.py
rm -rf scrape-open-llm-leaderboard
"""


def update_llm_df():
    """
    Scrape the open-llm-leaderboard and update the leaderboard dataframe
    """
    subprocess.run(scrapping_script, shell=True)
    create_repo(repo_id=MAIN_REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)
    upload_file(
        repo_id=MAIN_REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo=LLM_DF,
        path_or_fileobj="open-llm-leaderboard.csv",
    )


def update_llm_perf_leaderboard():
    update_llm_df()
    update_perf_dfs()


if __name__ == "__main__":
    update_llm_perf_leaderboard()
