import gradio as gr
import pandas as pd
from huggingface_hub import repo_exists, snapshot_download
from llm_perf.common.hardware_config import load_hardware_configs
from glob import glob
from llm_perf.update_llm_perf_leaderboard import patch_json
from optimum_benchmark import Benchmark
import json
from huggingface_hub.errors import RepositoryNotFoundError

PERF_REPO_ID = "optimum-benchmark/llm-perf-{backend}-{hardware}-{subset}-{machine}"

def create_status_df():
    hardware_configs = load_hardware_configs("llm_perf/hardware.yaml")
    
    rows = []
    for hardware_config in hardware_configs:
        for subset in hardware_config.subsets:
            for backend in hardware_config.backends:
                repo_id = PERF_REPO_ID.format(
                    subset=subset,
                    machine=hardware_config.machine,
                    backend=backend, 
                    hardware=hardware_config.hardware
                )
                
                exists = repo_exists(repo_id, repo_type="dataset")
                status = "✅" if exists else "⛔️"
                
                rows.append({
                    "Backend": backend,
                    "Hardware": hardware_config.hardware,
                    "Subset": subset,
                    "Machine": hardware_config.machine,
                    "Status": status
                })
    
    df = pd.DataFrame(rows)
    return df

def create_benchmark_status_df():
    hardware_configs = load_hardware_configs("llm_perf/hardware.yaml")
    
    rows = []
    for hardware_config in hardware_configs:
        for subset in hardware_config.subsets:
            for backend in hardware_config.backends:
                repo_id = PERF_REPO_ID.format(
                    subset=subset,
                    machine=hardware_config.machine,
                    backend=backend,
                    hardware=hardware_config.hardware
                )
                
                try:
                    snapshot = snapshot_download(
                        repo_type="dataset",
                        repo_id=repo_id,
                        allow_patterns=["**/benchmark.json"],
                    )
                except RepositoryNotFoundError as e:
                    print(f"Repository {repo_id} not found")
                    continue
                
                for file in glob(f"{snapshot}/**/benchmark.json", recursive=True):
                    patch_json(file)
                    
                    with open(file, "r") as f:
                        data = json.load(f)
                        benchmark = Benchmark.from_json(file)
                        df = benchmark.to_dataframe()
                        
                        # print("hello")
                        
                        for _, row in df.iterrows():
                            if "report.traceback" in row:
                                traceback = row["report.traceback"]
                            else:
                                traceback = ""
                                # print(f"No traceback for {row['config.name']} {row['config.backend.model']}")
                            rows.append({
                                "Backend": backend,
                                "Hardware": hardware_config.hardware,
                                "Subset": subset,
                                "Machine": hardware_config.machine,
                                "Status": "✅" if traceback == "" else "⛔️",
                                "Model": row["config.backend.model"],
                                "Experiment": row["config.name"],
                                "Traceback": traceback,
                                "Full Data": json.dumps(row.to_dict()),
                                # "Markdown": f"### Model: {row['config.backend.model']}\n### Experiment: {row['config.name']}\n\n```json\n{json.dumps(row.to_dict(), indent=2)}\n```"
                            })
                # except:
                #     rows.append({
                #         "Backend": backend,
                #         "Hardware": hardware_config.hardware,
                #         "Subset": subset,
                #         "Machine": hardware_config.machine,
                #         "Status": "⛔️",
                #         "Model": "N/A",
                #         "Experiment": "N/A"
                #     })
    
    df = pd.DataFrame(rows)
    return df

def create_status_table():
    df = create_status_df()
    return gr.DataFrame(
        value=df,
        headers=["Backend", "Hardware", "Subset", "Machine", "Status"],
        row_count=(len(df), "fixed"),
        col_count=(5, "fixed"),
        wrap=True
    )

def create_benchmark_table(df_benchmark_status):

    return gr.DataFrame(
        value=df_benchmark_status,
        headers=["Backend", "Hardware", "Subset", "Machine", "Status", "Model", "Experiment", "Traceback", "Full Data"],
        row_count=(len(df_benchmark_status), "fixed"),
        col_count=(9, "fixed"),
        column_widths=[100, 100, 100, 100, 100, 200, 100, 100, 100],
    )
    
def compute_machine_stats(df_benchmark_status):
    """
    Compute statistics about failed benchmarks per machine
    Args:
        df_benchmark_status (pd.DataFrame): DataFrame containing benchmark status information
    Returns:
        gr.DataFrame: Gradio DataFrame with machine failure statistics
    """
    # Stats per machine
    stats_by_machine = df_benchmark_status.groupby(['Machine']).agg(
        Total_Benchmarks=('Status', 'count'),
        Failed_Benchmarks=('Status', lambda x: (x == '⛔️').sum())
    ).reset_index()
    
    stats_by_machine['Success_Rate'] = ((stats_by_machine['Total_Benchmarks'] - stats_by_machine['Failed_Benchmarks']) / 
                            stats_by_machine['Total_Benchmarks'] * 100).round(2)
    stats_by_machine['Success_Rate'] = stats_by_machine['Success_Rate'].astype(str) + '%'
    
    machine_stats = gr.DataFrame(
        value=stats_by_machine,
        headers=["Machine", "Total_Benchmarks", "Failed_Benchmarks", "Success_Rate"],
        row_count=(len(stats_by_machine), "fixed"),
        col_count=(4, "fixed"),
        wrap=True
    )
    
    return machine_stats

def compute_config_stats(df_benchmark_status):
    """
    Compute statistics about failed benchmarks per configuration
    Args:
        df_benchmark_status (pd.DataFrame): DataFrame containing benchmark status information
    Returns:
        gr.DataFrame: Gradio DataFrame with configuration failure statistics
    """
    # Stats per configuration
    stats_by_config = df_benchmark_status.groupby(['Backend', 'Hardware', 'Subset', 'Machine']).agg(
        Total_Benchmarks=('Status', 'count'),
        Failed_Benchmarks=('Status', lambda x: (x == '⛔️').sum())
    ).reset_index()
    
    stats_by_config['Success_Rate'] = ((stats_by_config['Total_Benchmarks'] - stats_by_config['Failed_Benchmarks']) / 
                            stats_by_config['Total_Benchmarks'] * 100).round(2)
    stats_by_config['Success_Rate'] = stats_by_config['Success_Rate'].astype(str) + '%'
    
    config_stats = gr.DataFrame(
        value=stats_by_config,
        headers=["Backend", "Hardware", "Subset", "Machine", "Total_Benchmarks", "Failed_Benchmarks", "Success_Rate"],
        row_count=(len(stats_by_config), "fixed"),
        col_count=(7, "fixed"),
        wrap=True
    )
    
    return config_stats

def main():
    
    df_benchmark_status = create_benchmark_status_df()
    
    with gr.Blocks() as demo:
        with gr.Tab("Hardware status"):    
            gr.Markdown("# LLM Performance Dashboard")
            gr.Markdown("Status of benchmark results across different configurations")
            create_status_table()
        with gr.Tab("Benchmark status"):
            gr.Markdown("# Benchmark Results Status")
            gr.Markdown("Status of individual benchmark runs with model and experiment details")
            create_benchmark_table(df_benchmark_status)
        with gr.Tab("Stats"):
            gr.Markdown("# Stats")
            gr.Markdown("## Stats by Machine")
            gr.Markdown("Overall statistics per machine")
            compute_machine_stats(df_benchmark_status)
            gr.Markdown("## Stats by Configuration")
            gr.Markdown("Detailed statistics for each configuration")
            compute_config_stats(df_benchmark_status)
        with gr.Tab("Trends"):
            gr.Markdown("## Trends")
            gr.Markdown("Trends in benchmark results")
            gr.Markdown("TODO")
    
    demo.launch()

if __name__ == "__main__":
    main()
