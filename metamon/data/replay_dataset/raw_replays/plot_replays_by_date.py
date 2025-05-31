import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import defaultdict


def load_time_series(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def parse_month_str(month_str: str) -> datetime:
    return datetime.strptime(month_str, "%Y-%m")


def aggregate_all_formats(data: dict) -> list:
    # Skip metadata key
    formats = [k for k in data.keys() if not k.startswith("_")]

    # Aggregate counts by month
    monthly_totals = defaultdict(int)
    for format_str in formats:
        for entry in data[format_str]:
            monthly_totals[entry["month"]] += entry["count"]

    # Convert to sorted list of (month, count) pairs
    return sorted((month, count) for month, count in monthly_totals.items())


def plot_aggregate_time_series(data: dict, ax: plt.Axes) -> None:
    monthly_data = aggregate_all_formats(data)
    dates = [parse_month_str(month) for month, _ in monthly_data]
    counts = [count for _, count in monthly_data]

    # Plot bars
    ax.bar(dates, counts, width=25, alpha=0.7)

    # Enhance visual elements
    ax.set_title("Total Replays Across All Formats", fontsize=16, pad=20)
    ax.set_xlabel("Date", fontsize=14, labelpad=10)
    ax.set_ylabel("Number of Replays", fontsize=14, labelpad=10)

    # Format x-axis to show dates nicely
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # Increase tick label size
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Set x-axis limits from Jan 1 2014 to present
    ax.set_xlim(datetime(2014, 1, 1), datetime.now())

    # Rotate and align the tick labels so they look better
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add gridlines and thicken spines
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def plot_format_subplots(data: dict, fig: plt.Figure) -> None:
    # Skip metadata key and sort formats
    formats = sorted([k for k in data.keys() if not k.startswith("_")])
    n_formats = len(formats)

    # Calculate subplot grid dimensions
    n_cols = min(3, n_formats)
    n_rows = (n_formats + n_cols - 1) // n_cols

    # Create subplots with spacing
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # Create subplots
    for idx, format_str in enumerate(formats, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)

        # Extract and plot data for this format
        format_data = data[format_str]
        dates = [parse_month_str(entry["month"]) for entry in format_data]
        counts = [entry["count"] for entry in format_data]

        # Plot bars
        ax.bar(dates, counts, width=25, alpha=0.7)
        ax.set_title(format_str, fontsize=14, pad=10)

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Increase tick label size
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Rotate and align the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add gridlines and thicken spines
        ax.grid(True, linestyle="--", alpha=0.7, axis="y")
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Add overall title
    fig.suptitle("Replay Counts by Format", fontsize=20, y=1.02)


def main():
    parser = argparse.ArgumentParser(description="Plot replay time series data")
    parser.add_argument(
        "json_paths", type=str, nargs="+", help="Path(s) to time series JSON file"
    )
    parser.add_argument("--output", type=str, help="Output figure names")
    args = parser.parse_args()

    # Load data
    all_data = {}
    for json_path in args.json_paths:
        data = load_time_series(json_path)
        for tier, tier_data in data.items():
            if tier not in all_data:
                all_data[tier] = []
            all_data[tier].extend(tier_data)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create figure for aggregate plot
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_aggregate_time_series(all_data, ax1)
    fig1.tight_layout()

    # Create figure for format subplots
    fig2 = plt.figure(figsize=(15, 10))
    plot_format_subplots(all_data, fig2)
    fig2.tight_layout()

    # Save plots
    fig1.savefig(f"{args.output}_aggregate.png", bbox_inches="tight", dpi=300)
    fig2.savefig(f"{args.output}_by_format.png", bbox_inches="tight", dpi=300)
    print(f"Saved plots to {args.output}_aggregate.png and {args.output}_by_format.png")
    plt.close()


if __name__ == "__main__":
    main()
