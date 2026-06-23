import gradio as gr
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from plotly.graph_objs._figure import Figure
from loguru import logger
from typing import Optional, Tuple, List

from llm_perf.common.dashboard_manager import DashboardManager


def create_status_plot(df: pd.DataFrame) -> Optional[Figure]:
    """Create a status plot showing success/failure over time."""
    if df.empty:
        return None

    # Ensure last_updated is datetime
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df["success_str"] = df["success"].map({True: "Success", False: "Failure"})

    # Create hover text with more details
    df["hover_text"] = df.apply(
        lambda row: f"Model: {row['model']}<br>"
        + f"Hardware: {row['hardware']}<br>"
        + f"Machine: {row['machine']}<br>"
        + f"Status: {row['success_str']}<br>"
        + f"Time: {row['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}",
        axis=1,
    )

    fig = px.scatter(
        df,
        x="last_updated",
        y="model",
        color="success_str",
        title="Benchmark Status Over Time",
        labels={"last_updated": "Time", "model": "Model", "success_str": "Status"},
        hover_data=["hover_text"],
        height=600,
    )  # Make plot taller to accommodate more models

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Model",
        showlegend=True,
        legend_title="Status",
        hovermode="closest",
    )

    return fig


def create_hardware_stats(df: pd.DataFrame) -> Optional[Figure]:
    """Create statistics about hardware usage."""
    if df.empty:
        return None

    stats = (
        df.groupby(["hardware", "machine"])["success"]
        .agg(["count", "mean"])
        .reset_index()
    )
    # Calculate success rate as percentage
    stats["success_rate"] = (stats["mean"] * 100).round(2)
    # Drop the mean column since we've converted it to success_rate
    stats = stats.drop("mean", axis=1)
    stats = stats.rename(columns={"count": "total_runs"})

    fig = px.bar(
        stats,
        x="hardware",
        y="total_runs",
        color="success_rate",
        title="Hardware Usage and Success Rate",
        labels={
            "hardware": "Hardware Type",
            "total_runs": "Total Runs",
            "success_rate": "Success Rate (%)",
        },
    )
    return fig


class DashboardApp:
    def __init__(self):
        self.dashboard_manager = DashboardManager()

    def refresh_data(
        self,
        time_range: str,
        machine: str = "All",
        hardware: str = "All",
        model: str = "All",
    ) -> Tuple[Optional[Figure], Optional[Figure], Optional[List[List[str]]]]:
        """
        Refresh dashboard data based on filters.

        Args:
            time_range: Time range to filter (e.g., '1d', '7d', '30d', 'all')
            machine: Machine name filter
            hardware: Hardware type filter
            model: Model name filter

        Returns:
            Tuple of (status plot, hardware stats plot, data table)
        """
        try:
            # Get the data
            df = self.dashboard_manager.get_latest_runs(
                machine=machine if machine != "All" else None,
                hardware=hardware if hardware != "All" else None,
                model=model if model != "All" else None,
            )

            if df.empty:
                return None, None, None

            # Apply time range filter
            if time_range != "all":
                days = int(time_range[:-1])
                cutoff = datetime.now() - timedelta(days=days)
                df = df[df["last_updated"] >= cutoff]

            # Create visualizations
            status_plot = create_status_plot(df)
            hardware_plot = create_hardware_stats(df)

            # Prepare table data
            table_df = df[
                ["model", "hardware", "machine", "success", "last_updated"]
            ].copy()
            table_df["last_updated"] = table_df["last_updated"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            table_data: List[List[str]] = [
                [str(val) for val in row] for row in table_df.values.tolist()
            ]

            return status_plot, hardware_plot, table_data

        except Exception as e:
            logger.error(f"Error refreshing dashboard data: {str(e)}")
            return None, None, None

    def launch(self, port: int = 7860, share: bool = False):
        """Launch the Gradio interface.

        Args:
            port: Port to run the dashboard on
            share: Whether to create a public URL
        """
        with gr.Blocks(title="LLM Performance Dashboard") as interface:
            gr.Markdown("# 🚀 LLM Performance Dashboard")
            gr.Markdown(
                "Monitor the status and performance of LLM benchmarks across different hardware configurations."
            )

            with gr.Row():
                time_range = gr.Dropdown(
                    choices=["1d", "7d", "30d", "all"], value="7d", label="Time Range"
                )
                machine = gr.Dropdown(
                    choices=["All"],  # Will be populated dynamically
                    value="All",
                    label="Machine",
                )
                hardware = gr.Dropdown(
                    choices=["All"],  # Will be populated dynamically
                    value="All",
                    label="Hardware",
                )
                model = gr.Dropdown(
                    choices=["All"],  # Will be populated dynamically
                    value="All",
                    label="Model",
                )
                refresh_btn = gr.Button("🔄 Refresh")

            with gr.Row():
                status_plot = gr.Plot(label="Benchmark Status")
                hardware_plot = gr.Plot(label="Hardware Statistics")

            with gr.Row():
                results_table = gr.Dataframe(
                    headers=["Model", "Hardware", "Machine", "Success", "Last Updated"],
                    label="Recent Benchmark Results",
                )

            # Update function
            def update_dashboard(
                time_range: str, machine: str, hardware: str, model: str
            ):
                return self.refresh_data(time_range, machine, hardware, model)

            # Register update function
            refresh_btn.click(
                fn=update_dashboard,
                inputs=[time_range, machine, hardware, model],
                outputs=[status_plot, hardware_plot, results_table],
            )

            # Auto-refresh on load
            interface.load(
                fn=update_dashboard,
                inputs=[time_range, machine, hardware, model],
                outputs=[status_plot, hardware_plot, results_table],
            )

        # Launch the interface with specified parameters
        interface.launch(server_port=port, share=share)


if __name__ == "__main__":
    app = DashboardApp()
    app.launch()
