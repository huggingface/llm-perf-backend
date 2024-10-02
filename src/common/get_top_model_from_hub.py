import json
import os
from collections import defaultdict
from typing import Dict, List

import requests
from datasets import Dataset


def get_top_text_generation_models(n: int, sort: str = "downloads", direction: int = -1) -> List[Dict]:
    base_url = "https://huggingface.co/api/models"
    params = {
        "sort": sort,
        "direction": direction,
        "limit": n,
        "filter": "text-generation",
        "full": "false",
    }

    headers = {}
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    if huggingface_token:
        headers["Authorization"] = f"Bearer {huggingface_token}"

    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()  # Raise an exception for bad responses

    models = response.json()
    return [
        {
            "organization": model["id"].split("/")[0],
            "model_name": model["id"].split("/")[-1],
            "downloads": model.get("downloads", 0),
        }
        for model in models
        if "downloads" in model
    ]


def save_to_json(data: List[Dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {filename}")


def compute_org_downloads(models: List[Dict]) -> Dict[str, int]:
    org_downloads = defaultdict(int)
    for model in models:
        org_downloads[model["organization"]] += model["downloads"]
    return dict(org_downloads)


def upload_to_hf_dataset(data: List[Dict], dataset_name: str):
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(dataset_name)
    print(f"Data uploaded to Hugging Face dataset: {dataset_name}")


def main():
    # Set up authentication (optional, but recommended)
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    if huggingface_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = huggingface_token
    else:
        print("Warning: HUGGINGFACE_TOKEN not found in environment variables. Running without authentication.")

    n = 100
    top_models = get_top_text_generation_models(n)

    print(f"\nTop {n} text generation models on Hugging Face Hub:")
    for i, model in enumerate(top_models, 1):
        print(f"{i}. {model['organization']}/{model['model_name']}: {model['downloads']:,} downloads")

    # Upload to Hugging Face dataset
    dataset_name = "optimum-benchmark/top-text-generation-models"
    upload_to_hf_dataset(top_models, dataset_name)

    # Display top 10 organizations by downloads
    print("\nTop 10 organizations by total downloads:")
    org_downloads = compute_org_downloads(top_models)
    sorted_orgs = sorted(org_downloads.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (org, downloads) in enumerate(sorted_orgs, 1):
        print(f"{i}. {org}: {downloads:,} downloads")


if __name__ == "__main__":
    main()
