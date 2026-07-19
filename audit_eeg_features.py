"""Audit the cached ZuCo sentence-level EEG features without training a model.

This script intentionally uses the same feature order and seeded split function
as ``main_new.py``. Its output is diagnostic evidence, not a paper-comparable
model result.
"""

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
from types import SimpleNamespace

import numpy as np

from split_utils import split_eeg_dict


BANDS = ('t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2')
CLASS_NAMES = ('neutral', 'positive', 'negative')
CLASS_NUM = len(CLASS_NAMES)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_cache', required=True)
    parser.add_argument('--eeg_dir', default='data/SR')
    parser.add_argument(
        '--labels_csv', default='data/sentiment_labels_clean.csv'
    )
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_or_build_eeg_dict(args):
    if os.path.exists(args.eeg_cache):
        with open(args.eeg_cache, 'rb') as cache_file:
            eeg_dict = pickle.load(cache_file)
        return eeg_dict, 'existing cache'

    import pandas as pd

    from dataset_new import prepare_sr_eeg_data

    labels = pd.read_csv(args.labels_csv)
    eeg_dict = prepare_sr_eeg_data(
        args.eeg_dir,
        labels.sentence.tolist(),
        labels.sentiment_label.tolist(),
        labels.sentence_id.tolist(),
        SimpleNamespace(dev=0),
    )
    with open(args.eeg_cache, 'wb') as cache_file:
        pickle.dump(eeg_dict, cache_file)
    return eeg_dict, 'built from raw .mat files'


def feature_vector(item):
    return np.concatenate([
        np.asarray(item[band], dtype=np.float64) for band in BANDS
    ])


def split_arrays(split):
    ids = list(split.keys())
    features = np.stack([feature_vector(split[key]) for key in ids])
    labels = np.asarray(
        [int(split[key]['label']) for key in ids], dtype=np.int64
    )
    return ids, features, labels


def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def feature_summary(features, labels):
    finite = np.isfinite(features)
    finite_values = features[finite]
    safe = np.where(finite, features, np.nan)
    feature_std = np.nanstd(safe, axis=0)
    sample_norm = np.sqrt(np.nansum(safe ** 2, axis=1))

    band_means = []
    band_stds = []
    for band_index in range(len(BANDS)):
        start = band_index * 104
        band = safe[:, start:start + 104]
        band_means.append(np.nanmean(band, axis=1))
        band_stds.append(np.nanstd(band, axis=1))
    band_means = np.stack(band_means, axis=1)
    band_stds = np.stack(band_stds, axis=1)

    return {
        'shape': list(features.shape),
        'class_counts': np.bincount(
            labels, minlength=CLASS_NUM
        ).astype(int).tolist(),
        'nonfinite_values': int((~finite).sum()),
        'samples_with_nonfinite': int((~finite).any(axis=1).sum()),
        'global': {
            'min': finite_number(np.min(finite_values)),
            'max': finite_number(np.max(finite_values)),
            'mean': finite_number(np.mean(finite_values)),
            'std': finite_number(np.std(finite_values)),
        },
        'per_feature_std_across_samples': {
            'min': finite_number(np.nanmin(feature_std)),
            'median': finite_number(np.nanmedian(feature_std)),
            'max': finite_number(np.nanmax(feature_std)),
            'near_constant_count_le_1e-8': int(
                np.sum(feature_std <= 1e-8)
            ),
        },
        'sample_l2_norm': {
            'min': finite_number(np.nanmin(sample_norm)),
            'mean': finite_number(np.nanmean(sample_norm)),
            'max': finite_number(np.nanmax(sample_norm)),
        },
        'per_sample_band_zscore_check': {
            'max_abs_band_mean': finite_number(
                np.nanmax(np.abs(band_means))
            ),
            'max_abs_band_std_minus_one': finite_number(
                np.nanmax(np.abs(band_stds - 1.0))
            ),
        },
    }


def row_hash(row):
    contiguous = np.ascontiguousarray(row)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def duplicate_summary(split_data):
    hashes = {}
    duplicate_label_conflicts = []
    for split_name, (_, features, labels) in split_data.items():
        for index, (row, label) in enumerate(zip(features, labels)):
            digest = row_hash(row)
            record = {
                'split': split_name,
                'index': index,
                'label': int(label),
            }
            if digest in hashes:
                if any(
                    previous['label'] != int(label)
                    for previous in hashes[digest]
                ):
                    duplicate_label_conflicts.append({
                        'hash': digest,
                        'previous': hashes[digest],
                        'current': record,
                    })
                hashes[digest].append(record)
            else:
                hashes[digest] = [record]

    duplicate_groups = [
        records for records in hashes.values() if len(records) > 1
    ]
    cross_split_groups = [
        records for records in duplicate_groups
        if len({record['split'] for record in records}) > 1
    ]
    return {
        'unique_feature_rows': len(hashes),
        'duplicate_groups': len(duplicate_groups),
        'cross_split_duplicate_groups': len(cross_split_groups),
        'duplicate_label_conflicts': duplicate_label_conflicts,
    }


def confusion_matrix(labels, predictions):
    matrix = np.zeros((CLASS_NUM, CLASS_NUM), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def metric_summary(labels, predictions):
    matrix = confusion_matrix(labels, predictions)
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(
        diagonal,
        predicted_counts,
        out=np.zeros(CLASS_NUM, dtype=float),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        diagonal,
        true_counts,
        out=np.zeros(CLASS_NUM, dtype=float),
        where=true_counts != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(CLASS_NUM, dtype=float),
        where=(precision + recall) != 0,
    )
    return {
        'accuracy': float(diagonal.sum() / matrix.sum()),
        'macro_f1': float(f1.mean()),
        'confusion_matrix': matrix.tolist(),
        'predicted_class_counts': predicted_counts.astype(int).tolist(),
    }


def squared_distances(left, right):
    distances = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * np.matmul(left, right.T)
    )
    return np.maximum(distances, 0.0)


def nearest_centroid_predictions(train_x, train_y, target_x):
    centroids = np.stack([
        train_x[train_y == label].mean(axis=0)
        for label in range(CLASS_NUM)
    ])
    return np.argmin(squared_distances(target_x, centroids), axis=1)


def nearest_neighbor_predictions(train_x, train_y, target_x):
    nearest = np.argmin(squared_distances(target_x, train_x), axis=1)
    return train_y[nearest]


def simple_baselines(split_data):
    _, train_x, train_y = split_data['train']
    train_mean = train_x.mean(axis=0)
    train_std = train_x.std(axis=0)
    train_scale = np.where(train_std <= 1e-8, 1.0, train_std)

    standardized = {}
    for name, (ids, features, labels) in split_data.items():
        standardized[name] = (
            ids,
            (features - train_mean) / train_scale,
            labels,
        )

    results = {
        'notes': (
            'These are diagnostic non-neural baselines, not tuned paper '
            'reproductions. Standardization uses training statistics only.'
        ),
        'majority_class': int(np.bincount(train_y).argmax()),
    }
    for representation_name, representation in (
        ('authors_band_zscore_features', split_data),
        ('additional_train_feature_standardization', standardized),
    ):
        _, representation_train_x, representation_train_y = (
            representation['train']
        )
        representation_results = {}
        for target_name in ('validation', 'test'):
            _, target_x, target_y = representation[target_name]
            representation_results[target_name] = {
                'nearest_centroid': metric_summary(
                    target_y,
                    nearest_centroid_predictions(
                        representation_train_x,
                        representation_train_y,
                        target_x,
                    ),
                ),
                'one_nearest_neighbor': metric_summary(
                    target_y,
                    nearest_neighbor_predictions(
                        representation_train_x,
                        representation_train_y,
                        target_x,
                    ),
                ),
                'majority': metric_summary(
                    target_y,
                    np.full_like(target_y, results['majority_class']),
                ),
            }

        train_distances = squared_distances(
            representation_train_x, representation_train_x
        )
        np.fill_diagonal(train_distances, np.inf)
        leave_one_out_predictions = representation_train_y[
            np.argmin(train_distances, axis=1)
        ]
        representation_results['train_leave_one_out_1nn'] = metric_summary(
            representation_train_y, leave_one_out_predictions
        )
        results[representation_name] = representation_results

    return results


def main():
    args = get_args()
    np.random.seed(args.seed)

    eeg_dict, cache_source = load_or_build_eeg_dict(args)
    (
        train_split,
        validation_split,
        test_split,
        split_manifest,
    ) = split_eeg_dict(eeg_dict, args.seed)
    split_objects = {
        'train': train_split,
        'validation': validation_split,
        'test': test_split,
    }
    split_data = {
        name: split_arrays(split)
        for name, split in split_objects.items()
    }

    all_ids = set(eeg_dict.keys())
    split_id_sets = {
        name: set(ids) for name, (ids, _, _) in split_data.items()
    }
    finite_everywhere = all(
        np.isfinite(features).all()
        for _, features, _ in split_data.values()
    )

    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = 'unknown'

    result = {
        'audit_type': 'ZuCo sentence-level EEG feature and split audit',
        'scientific_model_result': False,
        'code_commit': git_commit,
        'seed': args.seed,
        'split_algorithm': split_manifest['algorithm'],
        'cache_source': cache_source,
        'construction_observations_from_released_code': {
            'unit_before_split': 'one vector per sentence_id',
            'subject_recordings': (
                'all matching .mat records for a sentence_id are averaged '
                'band-by-band before the seeded sentence split'
            ),
            'normalization': (
                'scipy.stats.zscore is applied separately to each averaged '
                '104-channel frequency-band vector'
            ),
            'feature_order': list(BANDS),
            'feature_width': 832,
            'recording_counts_per_sentence': (
                'not retained in existing post-aggregation cache'
            ),
        },
        'dataset': {
            'total_sentences': len(eeg_dict),
            'class_names': list(CLASS_NAMES),
            'overall_class_counts': np.bincount(
                [int(item['label']) for item in eeg_dict.values()],
                minlength=CLASS_NUM,
            ).astype(int).tolist(),
        },
        'splits': {
            name: {
                'size': len(ids),
                'sentence_ids': [
                    value.item() if hasattr(value, 'item') else value
                    for value in ids
                ],
                'class_counts': np.bincount(
                    labels, minlength=CLASS_NUM
                ).astype(int).tolist(),
                'feature_summary': feature_summary(features, labels),
            }
            for name, (ids, features, labels) in split_data.items()
        },
        'split_integrity': {
            'train_validation_id_overlap': len(
                split_id_sets['train'] & split_id_sets['validation']
            ),
            'train_test_id_overlap': len(
                split_id_sets['train'] & split_id_sets['test']
            ),
            'validation_test_id_overlap': len(
                split_id_sets['validation'] & split_id_sets['test']
            ),
            'union_matches_all_sentence_ids': (
                set().union(*split_id_sets.values()) == all_ids
            ),
            'all_features_finite': finite_everywhere,
            'duplicates': duplicate_summary(split_data),
        },
        'simple_baselines': (
            simple_baselines(split_data)
            if finite_everywhere
            else {'skipped': 'non-finite feature values detected'}
        ),
        'runtime': {
            'python': platform.python_version(),
            'numpy': np.__version__,
        },
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w') as output_file:
        json.dump(result, output_file, indent=2)

    print(json.dumps({
        'output_json': args.output_json,
        'dataset': result['dataset'],
        'split_integrity': result['split_integrity'],
        'simple_baselines': result['simple_baselines'],
    }, indent=2))


if __name__ == '__main__':
    main()
