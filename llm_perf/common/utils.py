import pandas as pd

from llm_perf.common.dependency import is_debug_mode

INPUT_SHAPES = {"batch_size": 1, "sequence_length": 256}
GENERATE_KWARGS = {"max_new_tokens": 64, "min_new_tokens": 64}


OPEN_LLM_LEADERBOARD = pd.read_csv(
    "hf://datasets/optimum-benchmark/llm-perf-leaderboard/llm-df.csv"
)
OPEN_LLM_LIST = OPEN_LLM_LEADERBOARD.drop_duplicates(subset=["Model"])["Model"].tolist()
PRETRAINED_OPEN_LLM_LIST = (
    OPEN_LLM_LEADERBOARD[OPEN_LLM_LEADERBOARD["Type"] == "pretrained"]
    .drop_duplicates(subset=["Model"])["Model"]
    .tolist()
)


def get_top_llm_list(n: int = 10) -> list[str]:
    """
    Fetches the top n text generation models from the Hugging Face dataset.

    Args:
        n (int): Number of top models to retrieve. Defaults to 10.

    Returns:
        list: A list of strings representing the top n models in the format "organization/model_name".
    """
    try:
        # Download the dataset from the Hugging Face Hub
        from datasets import load_dataset

        ds = load_dataset("optimum-benchmark/top-text-generation-models")

        # Get the data from the dataset
        models_data = ds["train"].to_pandas().to_dict("records")

        # sort by downloads
        models_data = sorted(models_data, key=lambda x: x["downloads"], reverse=True)

        # Create the list of top models
        top_models = [
            f"{model['organization']}/{model['model_name']}"
            for model in models_data[:n]
        ]

        return top_models
    except Exception as e:
        print(f"Error fetching top LLM list: {e}")
        return []


if is_debug_mode():
    CANONICAL_PRETRAINED_OPEN_LLM_LIST = ["gpt2"]
else:
    CANONICAL_PRETRAINED_OPEN_LLM_LIST = get_top_llm_list(n=10)
    print(
        f"Benchamrking the following {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)} models: {CANONICAL_PRETRAINED_OPEN_LLM_LIST}"
    )
