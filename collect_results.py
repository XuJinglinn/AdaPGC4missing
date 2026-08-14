import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd


METRIC_FILES = {
    "accuracy": "result.csv",
    "precision": "precision.csv",
    "recall": "recall.csv",
    "f1": "f1.csv",
}
AVERAGES = ("macro", "micro", "weighted")


def get_first_non_empty_line(file_path):
    """Return the first non-empty line from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def _safe_file_component(value):
    """Make an args.json value safe to use as part of an output filename."""
    value = str(value).strip() or "unknown"
    return re.sub(r'[<>:"/\\|?*\s]+', "_", value)


def _read_accuracy(file_path):
    """Read result.csv and return {corruption: accuracy}."""
    table = pd.read_csv(file_path, header=None)
    if table.empty or table.shape[1] < 2:
        raise ValueError("expected at least two columns: corruption, accuracy")

    values = {}
    for corruption, accuracy in table.iloc[:, :2].itertuples(index=False, name=None):
        if pd.isna(corruption):
            continue
        values[str(corruption).strip()] = accuracy
    if not values:
        raise ValueError("no result rows found")
    return values


def _read_aggregate_metric(file_path):
    """Read precision/recall/F1 CSV and flatten its corruption/average columns."""
    table = pd.read_csv(file_path)
    if table.empty:
        raise ValueError("no metric rows found")

    # append_metric_results writes an unnamed first column for corruption.
    corruption_column = table.columns[0]
    normalized_columns = {
        str(column).strip().lower(): column for column in table.columns[1:]
    }
    missing = [average for average in AVERAGES if average not in normalized_columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    values = {}
    for _, row in table.iterrows():
        corruption = row[corruption_column]
        if pd.isna(corruption):
            continue
        corruption = str(corruption).strip()
        for average in AVERAGES:
            values[f"{corruption}_{average}"] = row[normalized_columns[average]]
    if not values:
        raise ValueError("no metric rows found")
    return values


def _read_metric(metric_name, file_path):
    if metric_name == "accuracy":
        return _read_accuracy(file_path)
    return _read_aggregate_metric(file_path)


def collect_and_group_results(root="exp_logs", out_dir="collected_results"):
    """Collect all experiment metrics by dataset and corruption modality.

    Each experiment becomes one row. Files are written as::

        collected_results/<metric>/results_<dataset>_<modality>.csv

    Accuracy columns are named by corruption. Precision, recall, and F1 columns
    are named ``<corruption>_<average>`` for macro, micro, and weighted values.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"experiment root does not exist: {root}")

    grouped_data = {
        metric_name: defaultdict(list) for metric_name in METRIC_FILES
    }

    for exp_name in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        args_path = os.path.join(exp_dir, "args.json")
        if not os.path.exists(args_path):
            print(f"Skipping {exp_dir}: args.json not found")
            continue
        try:
            with open(args_path, "r", encoding="utf-8") as file:
                args = json.load(file)
            dataset = args.get("dataset", "unknown")
            modality = args.get("corruption_modality", "unknown")
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping {exp_dir}: failed to parse args.json: {error}")
            continue

        remark_path = os.path.join(exp_dir, "remark.md")
        remark = (
            get_first_non_empty_line(remark_path)
            if os.path.exists(remark_path)
            else ""
        )
        group_key = (str(dataset), str(modality))

        for metric_name, source_name in METRIC_FILES.items():
            source_path = os.path.join(exp_dir, source_name)
            if not os.path.exists(source_path) or os.path.getsize(source_path) == 0:
                print(
                    f"Skipping {metric_name} for {exp_name}: "
                    f"{source_name} is missing or empty"
                )
                continue
            try:
                metric_values = _read_metric(metric_name, source_path)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                print(f"Skipping {metric_name} for {exp_name}: {error}")
                continue

            row = {"exp_name": exp_name, **metric_values, "remark": remark}
            grouped_data[metric_name][group_key].append(pd.DataFrame([row]))

    for metric_name, groups in grouped_data.items():
        metric_dir = os.path.join(out_dir, metric_name)
        os.makedirs(metric_dir, exist_ok=True)
        for (dataset, modality), tables in groups.items():
            group_table = pd.concat(tables, ignore_index=True, sort=False)
            file_name = "results_{}_{}.csv".format(
                _safe_file_component(dataset),
                _safe_file_component(modality),
            )
            output_path = os.path.join(metric_dir, file_name)
            group_table.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Saved {output_path} with {len(group_table)} rows")

    return grouped_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect accuracy, precision, recall, and F1 experiment CSVs."
    )
    parser.add_argument(
        "--root",
        default="exp_logs",
        help="directory containing experiment subdirectories (default: exp_logs)",
    )
    parser.add_argument(
        "--out-dir",
        default="collected_results",
        help="output directory (default: collected_results)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    collect_and_group_results(root=cli_args.root, out_dir=cli_args.out_dir)
