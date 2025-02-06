import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo, HfApi
from loguru import logger
from typing import List, Optional
import time

from llm_perf.common.dashboard import BenchmarkRunDetails

DASHBOARD_REPO_ID = "optimum-benchmark/llm-perf-dashboard"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class DashboardManager:
    def __init__(self):
        # Ensure the dataset repository exists
        create_repo(repo_id=DASHBOARD_REPO_ID, repo_type="dataset", exist_ok=True)
        self._current_commit = None
        self._api = HfApi()
        self._is_first_upload = False

    def _get_current_commit(self) -> Optional[str]:
        """Get the current commit hash of the main branch."""
        try:
            repo_info = self._api.repo_info(
                repo_id=DASHBOARD_REPO_ID, repo_type="dataset"
            )
            return repo_info.sha
        except Exception as e:
            logger.error(f"Failed to get current commit: {str(e)}")
            return None

    def _load_existing_dataset(self) -> Optional[Dataset]:
        """Load the existing dataset from the hub."""
        try:
            dataset = load_dataset(DASHBOARD_REPO_ID, split="train")
            if isinstance(dataset, Dataset):
                self._current_commit = self._get_current_commit()
                return dataset
            else:
                logger.error("Loaded dataset is not of type Dataset")
                return None
        except Exception as e:
            if "doesn't contain any data files" in str(e):
                logger.info("No existing dataset found, this will be the first upload")
                self._is_first_upload = True
                self._current_commit = self._get_current_commit()
                return None
            logger.error(f"Failed to load existing dataset: {str(e)}")
            return None

    def _verify_commit(self) -> bool:
        """Verify that the current commit hasn't changed."""
        if self._is_first_upload:
            # For first upload, we don't need to verify commit
            return True

        current = self._get_current_commit()
        if current != self._current_commit:
            logger.error("Dataset has been updated since last read. Aborting upload.")
            return False
        return True

    def _convert_to_dict(self, run_details: BenchmarkRunDetails) -> dict:
        """Convert BenchmarkRunDetails to a dictionary format suitable for the dataset."""
        return {
            "machine": run_details.machine,
            "hardware": run_details.hardware,
            "subsets": run_details.subsets,
            "backends": run_details.backends,
            "model": run_details.model,
            "success": run_details.success,
            "traceback": run_details.traceback,
            "last_updated": run_details.last_updated,
            "run_id": run_details.run_id,
            "run_start_time": run_details.run_start_time,
        }

    def upload_run_details(self, run_details: BenchmarkRunDetails):
        """Upload a single benchmark run details to the dashboard dataset."""
        for attempt in range(MAX_RETRIES):
            try:
                # Reset first upload flag on each attempt
                self._is_first_upload = False

                # Load existing dataset
                existing_dataset = self._load_existing_dataset()
                if existing_dataset is None and not self._is_first_upload:
                    # Failed to load for reasons other than being first upload
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        logger.error(
                            "Max retries reached. Failed to upload run details."
                        )
                        return

                # Get existing data or empty list for first upload
                existing_data = existing_dataset.to_list() if existing_dataset else []

                # Convert the new run details to a dictionary
                new_run = self._convert_to_dict(run_details)

                # Combine existing data with new run
                combined_data = existing_data + [new_run]

                # Create new dataset
                dataset = Dataset.from_list(combined_data)

                # Verify commit hasn't changed (skipped for first upload)
                if not self._verify_commit():
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        logger.error(
                            "Max retries reached. Failed to upload run details."
                        )
                        return

                # Push to hub
                dataset.push_to_hub(repo_id=DASHBOARD_REPO_ID, split="train")
                logger.info(
                    f"Successfully uploaded run details for {run_details.run_id} to dashboard"
                )
                break

            except Exception as e:
                logger.error(
                    f"Failed to upload run details (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                break

    def upload_multiple_run_details(self, run_details_list: List[BenchmarkRunDetails]):
        """Upload multiple benchmark run details to the dashboard dataset."""
        for attempt in range(MAX_RETRIES):
            try:
                # Load existing dataset
                existing_dataset = self._load_existing_dataset()
                if existing_dataset is None:
                    existing_data = []
                else:
                    existing_data = existing_dataset.to_list()

                # Convert all new run details to dictionaries
                new_runs = [self._convert_to_dict(rd) for rd in run_details_list]

                # Combine existing data with new runs
                combined_data = existing_data + new_runs

                # Create new dataset
                dataset = Dataset.from_list(combined_data)

                # Verify commit hasn't changed
                if not self._verify_commit():
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        logger.error(
                            "Max retries reached. Failed to upload run details."
                        )
                        return

                # Push to hub
                dataset.push_to_hub(repo_id=DASHBOARD_REPO_ID, split="train")
                logger.info(
                    f"Successfully uploaded {len(run_details_list)} run details to dashboard"
                )
                break

            except Exception as e:
                logger.error(
                    f"Failed to upload run details (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                break

    def get_latest_runs(
        self,
        machine: Optional[str] = None,
        hardware: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Retrieve the latest benchmark runs from the dashboard dataset.

        Args:
            machine: Filter by machine name
            hardware: Filter by hardware type
            model: Filter by model name
            limit: Maximum number of runs to return

        Returns:
            DataFrame containing the latest runs
        """
        try:
            # Load the dataset
            dataset = load_dataset(DASHBOARD_REPO_ID, split="train")
            if not isinstance(dataset, Dataset):
                logger.error("Failed to load dataset: not a Dataset instance")
                return pd.DataFrame()

            # Convert to pandas DataFrame using dictionary
            data_dict = {col: dataset[col] for col in dataset.column_names}
            df = pd.DataFrame(data_dict)

            # Apply filters
            if machine:
                df = df[df["machine"] == machine]
            if hardware:
                df = df[df["hardware"] == hardware]
            if model:
                df = df[df["model"] == model]

            # Sort by last_updated and take the most recent runs
            df["last_updated"] = pd.to_datetime(df["last_updated"])
            df = df.sort_values("last_updated", ascending=False).head(limit)

            return df

        except Exception as e:
            logger.error(f"Failed to retrieve latest runs: {str(e)}")
            return pd.DataFrame()
