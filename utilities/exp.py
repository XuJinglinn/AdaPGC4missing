import os
import subprocess
from datetime import datetime, timedelta
import sys
import shutil
import json
import csv

import numpy as np
import torch
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

def get_recovered_features_records_dir(exp_dir):
    records_dir = os.path.join(exp_dir, 'recovered_features_records')
    os.makedirs(records_dir, exist_ok=True)
    return records_dir

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

def save_recovered_features_records(
        records_dir, corruption, batch_index, sample_names, records):
    """Save forward features and each predict_x2f trace for one test batch."""
    if records is None:
        return

    sample_names = list(sample_names)
    forward_results = records['forward_results']
    batch_size = int(forward_results['logits'].shape[0])
    if len(sample_names) != batch_size:
        raise ValueError(
            f'Expected {batch_size} sample names, but received {len(sample_names)}.'
        )
    if int(forward_results['feat'].shape[0]) != batch_size:
        raise ValueError('The first dimension of feat does not match logits.')
    for mask_name, mask_value in forward_results['mask'].items():
        if int(mask_value.shape[0]) != batch_size:
            raise ValueError(
                f'The {mask_name} mask does not match the forward batch size.'
            )

    corruption_dir = os.path.join(records_dir, corruption)
    os.makedirs(corruption_dir, exist_ok=True)

    full_mask = forward_results['mask']['full'].bool().tolist()
    full_sample_names = [
        sample_name
        for sample_name, is_full in zip(sample_names, full_mask)
        if is_full
    ]
    num_full_samples = len(full_sample_names)
    for feature_name in ('ca', 'cv'):
        feature_value = forward_results[feature_name]
        if feature_value is None:
            if num_full_samples != 0:
                raise ValueError(
                    f'{feature_name} is None for {num_full_samples} full samples.'
                )
        elif int(feature_value.shape[0]) != num_full_samples:
            raise ValueError(
                f'{feature_name} contains {feature_value.shape[0]} samples, but '
                f'the full mask selects {num_full_samples}.'
            )
    forward_payload = {
        'sample_names': sample_names,
        'logits': forward_results['logits'],
        'ca': forward_results['ca'],
        'cv': forward_results['cv'],
        'feat': forward_results['feat'],
        'mask': forward_results['mask'],
        'full_sample_names': full_sample_names,
    }
    forward_filename = f'batch_{batch_index:05d}_forward_results.pt'
    forward_path = os.path.join(corruption_dir, forward_filename)
    torch.save(forward_payload, forward_path)

    index_rows = [[
        corruption,
        batch_index,
        'forward_results',
        'all',
        batch_size,
        '',
        os.path.relpath(forward_path, records_dir).replace(os.sep, '/'),
    ]]

    for recovery_record in records['predict_x2f']:
        source = recovery_record['source']
        if source == 'a':
            mask_key = 'audio_only'
            observed_modality = 'audio'
            missing_modality = 'video'
        elif source == 'v':
            mask_key = 'video_only'
            observed_modality = 'video'
            missing_modality = 'audio'
        else:
            raise ValueError(f'Unsupported predict_x2f source: {source}')

        source_mask = forward_results['mask'][mask_key].bool().tolist()
        recovered_sample_names = [
            sample_name
            for sample_name, selected in zip(sample_names, source_mask)
            if selected
        ]
        num_samples = int(recovery_record['predict_x2f_output'].shape[0])
        if len(recovered_sample_names) != num_samples:
            raise ValueError(
                f'{source} predict_x2f returned {num_samples} samples, but its '
                f'mask selected {len(recovered_sample_names)} sample names.'
            )
        for value_name in ('alpha', 'cond_means', 'logits_gda'):
            value = recovery_record[value_name]
            if value is not None and int(value.shape[0]) != num_samples:
                raise ValueError(
                    f'{source} {value_name} contains {value.shape[0]} samples, '
                    f'but predict_x2f_output contains {num_samples}.'
                )

        recovery_payload = {
            'sample_names': recovered_sample_names,
            'source': source,
            'observed_modality': observed_modality,
            'missing_modality': missing_modality,
            'warmup_fallback': recovery_record['warmup_fallback'],
            'alpha': recovery_record['alpha'],
            'cond_means': recovery_record['cond_means'],
            'predict_x2f_output': recovery_record['predict_x2f_output'],
            'logits_gda': recovery_record['logits_gda'],
        }
        recovery_filename = f'batch_{batch_index:05d}_x2f_{source}.pt'
        recovery_path = os.path.join(corruption_dir, recovery_filename)
        torch.save(recovery_payload, recovery_path)
        index_rows.append([
            corruption,
            batch_index,
            'predict_x2f',
            source,
            num_samples,
            recovery_record['warmup_fallback'],
            os.path.relpath(recovery_path, records_dir).replace(os.sep, '/'),
        ])

    index_path = os.path.join(records_dir, 'index.csv')
    write_header = not os.path.exists(index_path) or os.path.getsize(index_path) == 0
    with open(index_path, 'a', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow([
                'corruption',
                'batch_index',
                'record_type',
                'source',
                'num_samples',
                'warmup_fallback',
                'file',
            ])
        writer.writerows(index_rows)

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
