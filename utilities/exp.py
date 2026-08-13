import os
import subprocess
from datetime import datetime, timedelta
import sys
import shutil
import json
import csv

import numpy as np
from sklearn import metrics as sklearn_metrics


def get_time_prefix():
    time_prefix = (datetime.now()).strftime('%Y%m%d_%H%M%S')
    return time_prefix

def init_experiment_dir(log_dir, exp_name):
    time_prefix = get_time_prefix()
    exp_dir = os.path.join(log_dir, f"{time_prefix}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)
    backup_code_dir = os.path.join(exp_dir, 'backup')
    os.makedirs(backup_code_dir, exist_ok=True)
    files = [
        'output.log',
        'result.csv',
        'precision.csv',
        'recall.csv',
        'f1.csv',
        'predictions.csv',
        'commit_hash.txt',
        'remark.md',
    ]
    for f in files:
        file_path = os.path.join(exp_dir, f)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as fp:
                pass
    return exp_dir


def add_remark(exp_dir, remark):
    file_path = os.path.join(exp_dir, 'remark.md')
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(remark + '\n')


def save_commit_hash(exp_dir):
    try:
        hash_str = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        hash_str = "Not a git repo"
    with open(os.path.join(exp_dir, 'commit_hash.txt'), 'w') as f:
        f.write(hash_str)

def redirect_output_to_log(log_path):
    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    return logger

def get_result_csv_path(exp_dir):
    return os.path.join(exp_dir, 'result.csv')

def get_metric_csv_paths(exp_dir):
    return {
        'precision': os.path.join(exp_dir, 'precision.csv'),
        'recall': os.path.join(exp_dir, 'recall.csv'),
        'f1': os.path.join(exp_dir, 'f1.csv'),
    }

def get_predictions_csv_path(exp_dir):
    return os.path.join(exp_dir, 'predictions.csv')

def calculate_classification_metrics(logits, targets, n_class):
    """Calculate aggregate single-label metrics as percentages."""
    logits = logits.detach().cpu().numpy() if hasattr(logits, 'detach') else np.asarray(logits)
    targets = targets.detach().cpu().numpy() if hasattr(targets, 'detach') else np.asarray(targets)

    predicted_labels = np.argmax(logits, axis=1)
    true_labels = np.argmax(targets, axis=1) if targets.ndim > 1 else targets.astype(int)
    class_labels = np.arange(n_class)
    metric_functions = {
        'precision': sklearn_metrics.precision_score,
        'recall': sklearn_metrics.recall_score,
        'f1': sklearn_metrics.f1_score,
    }

    results = {}
    for metric_name, metric_function in metric_functions.items():
        values = {
            average: metric_function(
                true_labels,
                predicted_labels,
                labels=class_labels,
                average=average,
                zero_division=0,
            ) * 100.0
            for average in ('macro', 'micro', 'weighted')
        }
        results[metric_name] = values
    return results

def append_metric_results(metric_csv_paths, corruption, metric_results):
    """Write one aggregate-metric row per corruption, replacing duplicates."""
    header = ['', 'macro', 'micro', 'weighted']
    for metric_name, csv_path in metric_csv_paths.items():
        values = metric_results[metric_name]
        new_row = [
            corruption,
            f"{values['macro']:.2f}",
            f"{values['micro']:.2f}",
            f"{values['weighted']:.2f}",
        ]

        rows = []
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, 'r', encoding='utf-8', newline='') as csv_file:
                existing_rows = list(csv.reader(csv_file))
            if existing_rows and existing_rows[0] == header:
                rows = [row for row in existing_rows[1:] if row and row[0] != corruption]

        rows.append(new_row)
        with open(csv_path, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(rows)

def get_audio_sample_names(dataset, sample_indices):
    """Resolve shuffled DataLoader indices back to extension-free audio names."""
    if hasattr(sample_indices, 'detach'):
        sample_indices = sample_indices.detach().cpu().tolist()
    names = []
    for sample_index in sample_indices:
        datum = dataset.decode_data(dataset.data[int(sample_index)])
        names.append(os.path.splitext(os.path.basename(datum['wav']))[0])
    return names

def append_prediction_results(
        predictions_csv, corruption, severity, epoch, sample_names, logits, targets):
    """Append one CSV row for every prediction, retaining the complete logits."""
    logits = logits.detach().cpu().numpy() if hasattr(logits, 'detach') else np.asarray(logits)
    targets = targets.detach().cpu().numpy() if hasattr(targets, 'detach') else np.asarray(targets)
    true_labels = np.argmax(targets, axis=1) if targets.ndim > 1 else targets.astype(int)
    write_header = not os.path.exists(predictions_csv) or os.path.getsize(predictions_csv) == 0

    with open(predictions_csv, 'a', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(['corruption', 'severity', 'epoch', 'sample_name', 'logits', 'true_label'])
        for sample_name, sample_logits, true_label in zip(sample_names, logits, true_labels):
            writer.writerow([
                corruption,
                severity,
                epoch,
                sample_name,
                json.dumps(sample_logits.tolist(), separators=(',', ':')),
                int(true_label),
            ])

def get_backup_code_path(exp_dir):
    return os.path.join(exp_dir, 'backup')

def backup_files(target_dir):
    file_list = [
        'run_adapgc.py',
        'run_diag_adapgc.py',
        'TTA/ADAPGC.py',
        'TTA/ADAPGC_diag.py',
        'models/cav_mae.py',
        'dataloader.py',
        'utilities/exp.py',
    ]
    os.makedirs(target_dir, exist_ok=True)
    for file_path in file_list:
        if os.path.exists(file_path):
            dest_path = os.path.join(target_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
        else:
            print(f"Warning: File does not exist, not copied {file_path}")

def save_args(args, target_dir, filename="args.json"):
    os.makedirs(target_dir, exist_ok=True)
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = dict(args)
    with open(os.path.join(target_dir, filename), "w", encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
