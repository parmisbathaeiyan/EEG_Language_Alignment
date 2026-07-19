"""Audit unaveraged ZuCo subject-sentence EEG records without model training.

The released sentence-level loader averages recordings across subjects before
splitting. This diagnostic retains each valid subject-sentence recording while
assigning splits by sentence ID, so no sentence can leak across train,
validation, and test.
"""

import argparse
import json
import os
import pickle
import platform
import random
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as io
from scipy.stats import zscore

from audit_eeg_features import (
    BANDS,
    CLASS_NAMES,
    CLASS_NUM,
    metric_summary,
    squared_distances,
)


ELECTRODES_PER_BAND = 104
FEATURE_WIDTH = len(BANDS) * ELECTRODES_PER_BAND
RELEASED_EXCLUDED_FILE = 'resultsZDN_SR.mat'
SENTENCE_CORRECTIONS = {
    (
        'Ultimately feels emp11111ty and unsatisfying, like swallowing '
        'a Communion wafer without the wine.'
    ): (
        'Ultimately feels empty and unsatisfying, like swallowing '
        'a Communion wafer without the wine.'
    ),
    "Bullock's complete lack of focus and ability quickly derails the film.1":
        "Bullock's complete lack of focus and ability quickly derails the film.",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_dir', default='data/SR')
    parser.add_argument(
        '--labels_csv', default='data/sentiment_labels_clean.csv'
    )
    parser.add_argument('--released_cache', default=None)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def sentence_lookup(labels):
    lookup = {}
    for index, sentence in enumerate(labels.sentence.tolist()):
        # Match list.index() in the released loader: keep the first occurrence.
        lookup.setdefault(sentence, index)
    return lookup


def subject_id_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    if stem.startswith('results'):
        stem = stem[len('results'):]
    if stem.endswith('_SR'):
        stem = stem[:-len('_SR')]
    return stem


def load_subject_records(eeg_dir, labels_csv):
    labels = pd.read_csv(labels_csv)
    lookup = sentence_lookup(labels)
    label_values = labels.sentiment_label.tolist()
    sentence_ids = labels.sentence_id.tolist()

    records = []
    excluded_files = []
    file_summaries = {}
    unmatched_sentences = []

    for filename in sorted(os.listdir(eeg_dir)):
        if not filename.endswith('.mat'):
            continue
        if filename == RELEASED_EXCLUDED_FILE:
            excluded_files.append({
                'filename': filename,
                'reason': 'explicitly excluded by released loader',
            })
            continue

        file_path = os.path.join(eeg_dir, filename)
        subject_id = subject_id_from_filename(filename)
        sentence_data = np.atleast_1d(
            io.loadmat(
                file_path, squeeze_me=True, struct_as_record=False
            )['sentenceData']
        )
        valid_count = 0
        skipped_full_band_nan = 0
        partial_nan_values_zero_filled = 0

        for sentence_record in sentence_data:
            band_arrays = []
            full_band_nan = False
            partial_nan_count = 0
            for band in BANDS:
                values = np.asarray(
                    getattr(sentence_record, f'mean_{band}')[
                        :ELECTRODES_PER_BAND
                    ],
                    dtype=np.float32,
                ).copy()
                nan_mask = np.isnan(values)
                if np.all(nan_mask):
                    full_band_nan = True
                    break
                partial_nan_count += int(nan_mask.sum())
                values[nan_mask] = 0.0
                band_arrays.append(values)

            if full_band_nan:
                skipped_full_band_nan += 1
                continue

            sentence = str(sentence_record.content)
            sentence = SENTENCE_CORRECTIONS.get(sentence, sentence)
            if sentence not in lookup:
                unmatched_sentences.append({
                    'filename': filename,
                    'subject_id': subject_id,
                    'sentence': sentence,
                })
                continue

            label_index = lookup[sentence]
            label = int(label_values[label_index])
            if label == -1:
                label = 2
            sentence_id = json_scalar(sentence_ids[label_index])
            features = np.concatenate(band_arrays).astype(
                np.float32, copy=False
            )
            if features.shape != (FEATURE_WIDTH,):
                raise ValueError(
                    f'Unexpected feature shape {features.shape} for '
                    f'{subject_id}/{sentence_id}'
                )

            records.append({
                'subject_id': subject_id,
                'sentence_id': sentence_id,
                'label': label,
                'features': features,
            })
            valid_count += 1
            partial_nan_values_zero_filled += partial_nan_count

        file_summaries[subject_id] = {
            'filename': filename,
            'sentence_records_in_file': int(len(sentence_data)),
            'valid_records': valid_count,
            'skipped_full_band_nan': skipped_full_band_nan,
            'partial_nan_values_zero_filled':
                partial_nan_values_zero_filled,
        }

    if unmatched_sentences:
        examples = unmatched_sentences[:3]
        raise ValueError(
            f'{len(unmatched_sentences)} raw sentences did not match the '
            f'label CSV; examples: {examples}'
        )
    if not records:
        raise ValueError('No valid subject-sentence EEG records were found')

    return records, {
        'included_subject_files': file_summaries,
        'excluded_files': excluded_files,
    }


def sentence_group_split(records, seed):
    sentence_labels = {}
    for record in records:
        sentence_id = record['sentence_id']
        label = record['label']
        if (
            sentence_id in sentence_labels
            and sentence_labels[sentence_id] != label
        ):
            raise ValueError(
                f'Conflicting labels for sentence {sentence_id}'
            )
        sentence_labels[sentence_id] = label

    rng = random.Random(seed)
    label_sentence_ids = defaultdict(list)
    for sentence_id, label in sentence_labels.items():
        label_sentence_ids[label].append(sentence_id)
    for label in label_sentence_ids:
        rng.shuffle(label_sentence_ids[label])

    sentence_split = {}
    split_sentence_ids = {
        'train': set(),
        'validation': set(),
        'test': set(),
    }
    for label, ids in label_sentence_ids.items():
        train_count = int(0.8 * len(ids))
        validation_count = int(0.10 * len(ids))
        for index, sentence_id in enumerate(ids):
            if index < train_count:
                split = 'train'
            elif index < train_count + validation_count:
                split = 'validation'
            else:
                split = 'test'
            sentence_split[sentence_id] = split
            split_sentence_ids[split].add(sentence_id)

    split_indices = {}
    for split in ('train', 'validation', 'test'):
        split_indices[split] = np.asarray([
            index for index, record in enumerate(records)
            if sentence_split[record['sentence_id']] == split
        ], dtype=np.int64)

    return sentence_split, split_sentence_ids, split_indices


def records_to_arrays(records):
    return {
        'features': np.stack([
            record['features'] for record in records
        ]).astype(np.float64),
        'labels': np.asarray([
            record['label'] for record in records
        ], dtype=np.int64),
        'subjects': np.asarray([
            record['subject_id'] for record in records
        ], dtype=object),
        'sentence_ids': np.asarray([
            record['sentence_id'] for record in records
        ], dtype=object),
    }


def safe_scale(values, axis, keepdims):
    std = np.std(values, axis=axis, keepdims=keepdims)
    return np.where(std <= 1e-12, 1.0, std)


def normalization_variants(
    raw_features, subjects, train_indices
):
    reshaped = raw_features.reshape(
        len(raw_features), len(BANDS), ELECTRODES_PER_BAND
    )
    train_bands = reshaped[train_indices]
    variants = {}
    metadata = {}

    variants['raw_zero_filled'] = raw_features.copy()
    metadata['raw_zero_filled'] = {
        'description': (
            'Released NaN filtering/zero-fill only; no Z-score'
        ),
        'statistics_source': None,
        'preserves_between_record_band_magnitude': True,
    }

    within_band_mean = reshaped.mean(axis=2, keepdims=True)
    within_band_scale = safe_scale(
        reshaped, axis=2, keepdims=True
    )
    variants[
        'within_record_per_band_across_electrodes_zscore'
    ] = ((reshaped - within_band_mean) / within_band_scale).reshape(
        len(raw_features), FEATURE_WIDTH
    )
    metadata[
        'within_record_per_band_across_electrodes_zscore'
    ] = {
        'description': (
            'For each subject-sentence record and each band separately, '
            'Z-score its 104 electrode values'
        ),
        'statistics_source': (
            'each record/band independently, including evaluation records'
        ),
        'preserves_between_record_band_magnitude': False,
        'relationship_to_released_code': (
            'same across-electrode normalization, but before rather than '
            'after cross-subject averaging'
        ),
    }

    whole_mean = raw_features.mean(axis=1, keepdims=True)
    whole_scale = safe_scale(
        raw_features, axis=1, keepdims=True
    )
    variants['within_record_all_832_zscore'] = (
        raw_features - whole_mean
    ) / whole_scale
    metadata['within_record_all_832_zscore'] = {
        'description': (
            'For each subject-sentence record, one Z-score across all '
            '8 bands x 104 electrodes'
        ),
        'statistics_source': (
            'each complete record independently, including evaluation records'
        ),
        'preserves_between_record_band_magnitude': (
            'relative band differences within a record only'
        ),
    }

    global_band_mean = train_bands.mean(
        axis=(0, 2), keepdims=True
    )
    global_band_scale = safe_scale(
        train_bands, axis=(0, 2), keepdims=True
    )
    variants['train_global_per_band_all_electrodes_zscore'] = (
        (reshaped - global_band_mean) / global_band_scale
    ).reshape(len(raw_features), FEATURE_WIDTH)
    metadata['train_global_per_band_all_electrodes_zscore'] = {
        'description': (
            'One mean/std per frequency band, estimated jointly from all '
            'training records and all 104 electrodes'
        ),
        'statistics_source': 'training split only',
        'preserves_between_record_band_magnitude': True,
        'parameters': {
            'band_means': global_band_mean.reshape(-1).tolist(),
            'band_stds': global_band_scale.reshape(-1).tolist(),
        },
    }

    subject_normalized = np.empty_like(reshaped)
    subject_parameters = {}
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        subject_train_mask = subject_mask[train_indices]
        subject_train_bands = train_bands[subject_train_mask]
        if len(subject_train_bands) == 0:
            raise ValueError(
                f'Subject {subject} has no training records'
            )
        subject_band_mean = subject_train_bands.mean(
            axis=(0, 2), keepdims=True
        )
        subject_band_scale = safe_scale(
            subject_train_bands, axis=(0, 2), keepdims=True
        )
        subject_normalized[subject_mask] = (
            reshaped[subject_mask] - subject_band_mean
        ) / subject_band_scale
        subject_parameters[str(subject)] = {
            'band_means':
                subject_band_mean.reshape(-1).tolist(),
            'band_stds':
                subject_band_scale.reshape(-1).tolist(),
            'training_record_count':
                int(len(subject_train_bands)),
        }
    variants[
        'train_subject_per_band_all_electrodes_zscore'
    ] = subject_normalized.reshape(
        len(raw_features), FEATURE_WIDTH
    )
    metadata[
        'train_subject_per_band_all_electrodes_zscore'
    ] = {
        'description': (
            'For each participant and band, one mean/std estimated jointly '
            'from that participant training records and all 104 electrodes'
        ),
        'statistics_source': 'training split only, separately by subject',
        'preserves_between_record_band_magnitude': True,
        'parameters_by_subject': subject_parameters,
    }

    coordinate_mean = raw_features[train_indices].mean(
        axis=0, keepdims=True
    )
    coordinate_scale = safe_scale(
        raw_features[train_indices], axis=0, keepdims=True
    )
    variants['train_per_band_electrode_coordinate_zscore'] = (
        raw_features - coordinate_mean
    ) / coordinate_scale
    metadata['train_per_band_electrode_coordinate_zscore'] = {
        'description': (
            'One mean/std for each of the 832 band-electrode coordinates'
        ),
        'statistics_source': 'training split only',
        'preserves_between_record_band_magnitude': (
            'not directly; every coordinate is rescaled independently'
        ),
        'near_constant_training_coordinates': int(
            np.sum(
                np.std(
                    raw_features[train_indices], axis=0
                ) <= 1e-12
            )
        ),
    }

    return variants, metadata


def feature_distribution_summary(features):
    bands = features.reshape(
        len(features), len(BANDS), ELECTRODES_PER_BAND
    )
    sample_norms = np.linalg.norm(features, axis=1)
    per_record_band_mean = bands.mean(axis=2)
    per_record_band_std = bands.std(axis=2)
    return {
        'global_min': float(np.min(features)),
        'global_max': float(np.max(features)),
        'global_mean': float(np.mean(features)),
        'global_std': float(np.std(features)),
        'sample_l2_norm': {
            'min': float(np.min(sample_norms)),
            'mean': float(np.mean(sample_norms)),
            'max': float(np.max(sample_norms)),
        },
        'per_record_band_mean': {
            'min': float(np.min(per_record_band_mean)),
            'median': float(np.median(per_record_band_mean)),
            'max': float(np.max(per_record_band_mean)),
        },
        'per_record_band_std': {
            'min': float(np.min(per_record_band_std)),
            'median': float(np.median(per_record_band_std)),
            'max': float(np.max(per_record_band_std)),
        },
        'nonfinite_values': int((~np.isfinite(features)).sum()),
    }


def class_centroids(features, labels):
    return np.stack([
        features[labels == label].mean(axis=0)
        for label in range(CLASS_NUM)
    ])


def global_centroid_predictions(train_x, train_y, target_x):
    centroids = class_centroids(train_x, train_y)
    return np.argmin(
        squared_distances(target_x, centroids), axis=1
    )


def subject_centroid_predictions(
    train_x, train_y, train_subjects,
    target_x, target_subjects,
    exclude_target_subject=False,
):
    predictions = np.empty(len(target_x), dtype=np.int64)
    global_sums = np.stack([
        train_x[train_y == label].sum(axis=0)
        for label in range(CLASS_NUM)
    ])
    global_counts = np.asarray([
        np.sum(train_y == label) for label in range(CLASS_NUM)
    ])

    subject_class_sums = {}
    subject_class_counts = {}
    for subject in np.unique(train_subjects):
        subject_mask = train_subjects == subject
        subject_class_sums[subject] = np.stack([
            train_x[subject_mask & (train_y == label)].sum(axis=0)
            for label in range(CLASS_NUM)
        ])
        subject_class_counts[subject] = np.asarray([
            np.sum(subject_mask & (train_y == label))
            for label in range(CLASS_NUM)
        ])

    for subject in np.unique(target_subjects):
        target_mask = target_subjects == subject
        if subject not in subject_class_sums:
            raise ValueError(
                f'Subject {subject} has no training records'
            )
        if exclude_target_subject:
            sums = global_sums - subject_class_sums[subject]
            counts = global_counts - subject_class_counts[subject]
        else:
            sums = subject_class_sums[subject]
            counts = subject_class_counts[subject]
        if np.any(counts == 0):
            raise ValueError(
                f'Missing class centroid for subject mode: {subject}'
            )
        centroids = sums / counts[:, None]
        predictions[target_mask] = np.argmin(
            squared_distances(target_x[target_mask], centroids),
            axis=1,
        )
    return predictions


def nearest_neighbor_predictions(
    train_x, train_y, train_subjects,
    target_x, target_subjects,
):
    distances = squared_distances(target_x, train_x)
    if np.any(np.all(np.isinf(distances), axis=1)):
        raise ValueError('A target record has no training neighbor')

    global_neighbor_index = np.argmin(distances, axis=1)
    global_predictions = train_y[global_neighbor_index]

    same_subject_distances = distances.copy()
    same_subject_distances[
        target_subjects[:, None] != train_subjects[None, :]
    ] = np.inf
    if np.any(np.all(np.isinf(same_subject_distances), axis=1)):
        raise ValueError('A target subject has no same-subject train records')
    same_subject_index = np.argmin(
        same_subject_distances, axis=1
    )

    different_subject_distances = distances.copy()
    different_subject_distances[
        target_subjects[:, None] == train_subjects[None, :]
    ] = np.inf
    if np.any(np.all(np.isinf(different_subject_distances), axis=1)):
        raise ValueError(
            'A target record has no different-subject train records'
        )
    different_subject_index = np.argmin(
        different_subject_distances, axis=1
    )

    return {
        'global_1nn': {
            'predictions': global_predictions,
            'nearest_subject_same_fraction': float(np.mean(
                train_subjects[global_neighbor_index]
                == target_subjects
            )),
        },
        'same_subject_1nn': {
            'predictions': train_y[same_subject_index],
            'nearest_subject_same_fraction': 1.0,
        },
        'different_subject_1nn': {
            'predictions': train_y[different_subject_index],
            'nearest_subject_same_fraction': 0.0,
        },
    }


def same_subject_train_leave_sentence_out_predictions(
    train_x, train_y, train_subjects, train_sentence_ids
):
    predictions = np.empty(len(train_x), dtype=np.int64)
    for subject in np.unique(train_subjects):
        subject_indices = np.flatnonzero(
            train_subjects == subject
        )
        distances = squared_distances(
            train_x[subject_indices], train_x[subject_indices]
        )
        subject_sentence_ids = train_sentence_ids[subject_indices]
        distances[
            subject_sentence_ids[:, None]
            == subject_sentence_ids[None, :]
        ] = np.inf
        if np.any(np.all(np.isinf(distances), axis=1)):
            raise ValueError(
                f'Subject {subject} has no other training sentence'
            )
        predictions[subject_indices] = train_y[subject_indices][
            np.argmin(distances, axis=1)
        ]
    return predictions


def sentence_vote_metrics(sentence_ids, labels, predictions):
    grouped_predictions = defaultdict(list)
    grouped_labels = {}
    for sentence_id, label, prediction in zip(
        sentence_ids, labels, predictions
    ):
        grouped_predictions[sentence_id].append(int(prediction))
        if (
            sentence_id in grouped_labels
            and grouped_labels[sentence_id] != int(label)
        ):
            raise ValueError(
                f'Conflicting target labels for sentence {sentence_id}'
            )
        grouped_labels[sentence_id] = int(label)

    ordered_ids = list(grouped_predictions.keys())
    sentence_labels = np.asarray([
        grouped_labels[sentence_id] for sentence_id in ordered_ids
    ])
    sentence_predictions = np.asarray([
        int(np.bincount(
            grouped_predictions[sentence_id], minlength=CLASS_NUM
        ).argmax())
        for sentence_id in ordered_ids
    ])
    metrics = metric_summary(sentence_labels, sentence_predictions)
    metrics['sentence_count'] = len(ordered_ids)
    metrics['vote_tie_break'] = 'lowest class index'
    return metrics


def evaluate_predictions(
    labels, subjects, sentence_ids, prediction_info
):
    results = {}
    for method, info in prediction_info.items():
        predictions = info['predictions']
        results[method] = {
            'record_level': metric_summary(labels, predictions),
            'record_level_by_subject': {
                str(subject): metric_summary(
                    labels[subjects == subject],
                    predictions[subjects == subject],
                )
                for subject in np.unique(subjects)
            },
            'sentence_majority_vote': sentence_vote_metrics(
                sentence_ids, labels, predictions
            ),
            **{
                key: value for key, value in info.items()
                if key != 'predictions'
            },
        }
    return results


def evaluate_subject_geometry(
    features, arrays, split_indices
):
    train_index = split_indices['train']
    train_x = features[train_index]
    train_y = arrays['labels'][train_index]
    train_subjects = arrays['subjects'][train_index]
    majority_class = int(np.bincount(train_y).argmax())
    results = {}

    for split in ('validation', 'test'):
        target_index = split_indices[split]
        target_x = features[target_index]
        target_y = arrays['labels'][target_index]
        target_subjects = arrays['subjects'][target_index]
        target_sentence_ids = arrays['sentence_ids'][target_index]

        nearest = nearest_neighbor_predictions(
            train_x, train_y, train_subjects,
            target_x, target_subjects,
        )
        prediction_info = {
            'majority': {
                'predictions': np.full_like(
                    target_y, majority_class
                ),
            },
            'global_class_centroid': {
                'predictions': global_centroid_predictions(
                    train_x, train_y, target_x
                ),
            },
            'same_subject_class_centroid': {
                'predictions': subject_centroid_predictions(
                    train_x, train_y, train_subjects,
                    target_x, target_subjects,
                    exclude_target_subject=False,
                ),
            },
            'leave_subject_out_class_centroid': {
                'predictions': subject_centroid_predictions(
                    train_x, train_y, train_subjects,
                    target_x, target_subjects,
                    exclude_target_subject=True,
                ),
            },
            **nearest,
        }
        results[split] = evaluate_predictions(
            target_y, target_subjects, target_sentence_ids,
            prediction_info
        )

    train_sentence_ids = arrays['sentence_ids'][train_index]
    train_nearest = {
        'same_subject_1nn': {
            'predictions':
                same_subject_train_leave_sentence_out_predictions(
                    train_x, train_y, train_subjects,
                    train_sentence_ids,
                ),
            'nearest_subject_same_fraction': 1.0,
        },
    }
    results[
        'train_leave_sentence_out_same_subject_1nn'
    ] = evaluate_predictions(
        train_y, train_subjects, train_sentence_ids,
        train_nearest
    )
    return results


def aggregate_subjects_by_sentence(records, sentence_split):
    grouped = defaultdict(list)
    labels = {}
    for record in records:
        sentence_id = record['sentence_id']
        grouped[sentence_id].append(
            record['features'].reshape(
                len(BANDS), ELECTRODES_PER_BAND
            )
        )
        labels[sentence_id] = record['label']

    split_data = {}
    aggregate_vectors = {}
    for sentence_id, subject_bands in grouped.items():
        mean_bands = np.mean(
            np.asarray(subject_bands, dtype=np.float32), axis=0
        )
        normalized = np.stack([
            zscore(mean_bands[band_index])
            for band_index in range(len(BANDS))
        ]).reshape(FEATURE_WIDTH)
        aggregate_vectors[sentence_id] = normalized

    for split in ('train', 'validation', 'test'):
        ids = [
            sentence_id for sentence_id in aggregate_vectors
            if sentence_split[sentence_id] == split
        ]
        split_data[split] = {
            'sentence_ids': np.asarray(ids, dtype=object),
            'features': np.stack([
                aggregate_vectors[sentence_id]
                for sentence_id in ids
            ]).astype(np.float64),
            'labels': np.asarray([
                labels[sentence_id] for sentence_id in ids
            ], dtype=np.int64),
        }
    return split_data, aggregate_vectors


def evaluate_aggregated_geometry(split_data):
    train = split_data['train']
    majority_class = int(
        np.bincount(train['labels']).argmax()
    )
    results = {}
    for split in ('validation', 'test'):
        target = split_data[split]
        centroid_predictions = global_centroid_predictions(
            train['features'], train['labels'], target['features']
        )
        distances = squared_distances(
            target['features'], train['features']
        )
        nearest_predictions = train['labels'][
            np.argmin(distances, axis=1)
        ]
        results[split] = {
            'majority': metric_summary(
                target['labels'],
                np.full_like(target['labels'], majority_class),
            ),
            'nearest_centroid': metric_summary(
                target['labels'], centroid_predictions
            ),
            'one_nearest_neighbor': metric_summary(
                target['labels'], nearest_predictions
            ),
        }
    return results


def compare_released_cache(cache_path, aggregate_vectors):
    if not cache_path or not os.path.exists(cache_path):
        return {
            'available': False,
            'reason': 'released post-aggregation cache not found',
        }
    with open(cache_path, 'rb') as cache_file:
        cached = pickle.load(cache_file)

    common_ids = set(cached.keys()) & set(aggregate_vectors.keys())
    missing_in_reconstruction = sorted(
        str(value) for value in set(cached.keys()) - common_ids
    )
    missing_in_cache = sorted(
        str(value) for value in set(aggregate_vectors.keys()) - common_ids
    )
    max_differences = []
    for sentence_id in common_ids:
        cached_vector = np.concatenate([
            np.asarray(cached[sentence_id][band])
            for band in BANDS
        ])
        max_differences.append(float(np.max(np.abs(
            cached_vector - aggregate_vectors[sentence_id]
        ))))
    return {
        'available': True,
        'cache_sentence_count': len(cached),
        'reconstructed_sentence_count': len(aggregate_vectors),
        'common_sentence_count': len(common_ids),
        'missing_in_reconstruction': missing_in_reconstruction,
        'missing_in_cache': missing_in_cache,
        'maximum_absolute_feature_difference': (
            max(max_differences) if max_differences else None
        ),
    }


def count_summary(values):
    values = np.asarray(values, dtype=np.int64)
    return {
        'min': int(values.min()),
        'median': float(np.median(values)),
        'mean': float(values.mean()),
        'max': int(values.max()),
    }


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    records, raw_file_metadata = load_subject_records(
        args.eeg_dir, args.labels_csv
    )
    arrays = records_to_arrays(records)
    if not np.all(np.isfinite(arrays['features'])):
        raise ValueError(
            'Raw features still contain non-finite values after filtering'
        )
    subject_sentence_keys = [
        (record['subject_id'], record['sentence_id'])
        for record in records
    ]
    if len(set(subject_sentence_keys)) != len(subject_sentence_keys):
        raise ValueError(
            'Duplicate subject-sentence records were found'
        )
    (
        sentence_split,
        split_sentence_ids,
        split_indices,
    ) = sentence_group_split(records, args.seed)

    variants, normalization_metadata = normalization_variants(
        arrays['features'], arrays['subjects'], split_indices['train']
    )
    variant_results = {}
    for name, features in variants.items():
        print(f'Evaluating subject-level geometry: {name}')
        variant_results[name] = {
            'normalization': normalization_metadata[name],
            'feature_distribution': feature_distribution_summary(
                features
            ),
            'geometry': evaluate_subject_geometry(
                features, arrays, split_indices
            ),
        }

    aggregate_split_data, aggregate_vectors = (
        aggregate_subjects_by_sentence(records, sentence_split)
    )
    aggregate_evaluation = {
        'description': (
            'Reconstructed released behavior: mean available subject '
            'records per sentence, then Z-score each 104-electrode band'
        ),
        'geometry': evaluate_aggregated_geometry(
            aggregate_split_data
        ),
        'cache_comparison': compare_released_cache(
            args.released_cache, aggregate_vectors
        ),
    }

    sentence_record_counts = defaultdict(int)
    for record in records:
        sentence_record_counts[record['sentence_id']] += 1
    per_subject_counts = {
        subject: int(np.sum(arrays['subjects'] == subject))
        for subject in np.unique(arrays['subjects'])
    }

    split_metadata = {}
    sentence_labels = {
        record['sentence_id']: record['label'] for record in records
    }
    for split in ('train', 'validation', 'test'):
        index = split_indices[split]
        labels = arrays['labels'][index]
        sentence_class_counts = np.zeros(CLASS_NUM, dtype=np.int64)
        for sentence_id in split_sentence_ids[split]:
            sentence_class_counts[sentence_labels[sentence_id]] += 1
        split_metadata[split] = {
            'sentence_count': len(split_sentence_ids[split]),
            'sentence_ids': [
                json_scalar(sentence_id) for sentence_id in sorted(
                    split_sentence_ids[split], key=str
                )
            ],
            'sentence_class_counts':
                sentence_class_counts.astype(int).tolist(),
            'record_count': int(len(index)),
            'record_class_counts': np.bincount(
                labels, minlength=CLASS_NUM
            ).astype(int).tolist(),
            'subject_count': int(len(np.unique(
                arrays['subjects'][index]
            ))),
        }

    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = 'unknown'

    result = {
        'audit_type': (
            'ZuCo raw subject-sentence EEG and normalization audit'
        ),
        'scientific_model_result': False,
        'code_commit': git_commit,
        'seed': args.seed,
        'class_names': list(CLASS_NAMES),
        'split_policy': {
            'unit': 'sentence_id',
            'stratified_proportions': '80/10/remainder',
            'sentence_leakage_prevented': True,
            'note': (
                'All subject recordings for one sentence are assigned to '
                'the same split'
            ),
        },
        'raw_records': {
            'record_count': len(records),
            'subject_count': len(np.unique(arrays['subjects'])),
            'unique_sentence_count': len(
                np.unique(arrays['sentence_ids'])
            ),
            'records_per_sentence': count_summary(
                list(sentence_record_counts.values())
            ),
            'records_per_subject': per_subject_counts,
            'file_processing': raw_file_metadata,
        },
        'splits': split_metadata,
        'normalization_variants': variant_results,
        'geometry_methods': {
            'global_1nn': (
                'Nearest training recording from any subject'
            ),
            'same_subject_1nn': (
                'Nearest training sentence recording from the target '
                'recording participant'
            ),
            'different_subject_1nn': (
                'Nearest training recording after excluding the target '
                'participant'
            ),
            'global_class_centroid': (
                'One training-record centroid per sentiment class'
            ),
            'same_subject_class_centroid': (
                'One training centroid per sentiment class, estimated only '
                'from the target participant'
            ),
            'leave_subject_out_class_centroid': (
                'One training centroid per class after excluding all '
                'recordings from the target participant'
            ),
            'train_leave_sentence_out_same_subject_1nn': (
                'Within-participant training-set 1-NN after excluding every '
                'recording with the target sentence ID'
            ),
        },
        'released_cross_subject_aggregate_reconstruction':
            aggregate_evaluation,
        'interpretation_guardrails': [
            (
                'Nearest-neighbor and centroid scores are diagnostic '
                'geometry probes, not paper-comparable model results.'
            ),
            (
                'Record-level scores weight sentences with more valid '
                'subjects more heavily; sentence-majority scores give each '
                'held-out sentence one vote.'
            ),
            (
                'No test score is used to select a normalization variant.'
            ),
            (
                'Keeping subjects separate is not allowed to introduce '
                'sentence overlap across splits.'
            ),
        ],
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

    compact = {
        'output_json': args.output_json,
        'code_commit': git_commit,
        'raw_records': {
            key: value for key, value in result['raw_records'].items()
            if key not in ('file_processing', 'records_per_subject')
        },
        'splits': split_metadata,
        'released_cache_comparison':
            aggregate_evaluation['cache_comparison'],
        'variant_score_summary': {},
    }
    for variant_name, variant in variant_results.items():
        compact['variant_score_summary'][variant_name] = {}
        for split in ('validation', 'test'):
            compact['variant_score_summary'][variant_name][split] = {
                method: {
                    'record_accuracy':
                        metrics['record_level']['accuracy'],
                    'record_macro_f1':
                        metrics['record_level']['macro_f1'],
                    'sentence_vote_accuracy':
                        metrics['sentence_majority_vote']['accuracy'],
                    'sentence_vote_macro_f1':
                        metrics['sentence_majority_vote']['macro_f1'],
                }
                for method, metrics in variant[
                    'geometry'
                ][split].items()
            }
        compact['variant_score_summary'][variant_name][
            'train_leave_sentence_out_same_subject_1nn'
        ] = {
            method: {
                'record_accuracy':
                    metrics['record_level']['accuracy'],
                'record_macro_f1':
                    metrics['record_level']['macro_f1'],
                'sentence_vote_accuracy':
                    metrics['sentence_majority_vote']['accuracy'],
                'sentence_vote_macro_f1':
                    metrics['sentence_majority_vote']['macro_f1'],
            }
            for method, metrics in variant['geometry'][
                'train_leave_sentence_out_same_subject_1nn'
            ].items()
        }
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
