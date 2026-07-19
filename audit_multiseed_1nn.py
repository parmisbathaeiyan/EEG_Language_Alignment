"""Predeclared multi-seed audit of three promising ZuCo EEG 1-NN probes.

This is a diagnostic, not a trained model or a paper-comparable result. The
three setups were frozen after the v14 geometry audit. Every seed uses the same
canonical, order-independent sentence splitter. Sentence-label permutation
controls preserve class counts separately in train, validation, and test.
"""

import argparse
import json
import os
import pickle
import platform
import random
import subprocess
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from audit_eeg_features import BANDS, CLASS_NAMES, CLASS_NUM, metric_summary
from audit_subject_level_eeg import (
    ELECTRODES_PER_BAND,
    FEATURE_WIDTH,
    aggregate_subjects_by_sentence,
    load_subject_records,
    records_to_arrays,
    safe_scale,
)
from split_utils import (
    canonical_stratified_sentence_split,
    json_scalar,
    stable_id_key,
)


SETUP_A = 'released_averaged_1nn'
SETUP_B = 'unaveraged_released_norm_same_subject_1nn_vote'
SETUP_C = 'unaveraged_subject_train_norm_different_subject_1nn_vote'
SETUP_NAMES = (SETUP_A, SETUP_B, SETUP_C)
TARGET_SPLITS = ('validation', 'test')
METRIC_NAMES = ('accuracy', 'macro_f1')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_dir', default='data/SR')
    parser.add_argument(
        '--labels_csv', default='data/sentiment_labels_clean.csv'
    )
    parser.add_argument('--released_cache', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=50)
    parser.add_argument(
        '--permutations', type=int, default=200,
        help='Sentence-label permutations per split seed',
    )
    parser.add_argument('--permutation_seed', type=int, default=20260719)
    parser.add_argument(
        '--device', default='auto', choices=('auto', 'cpu', 'cuda'),
        help='Use CUDA for distance calculations when available',
    )
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


class NearestNeighborBackend:
    """Chunked NumPy/PyTorch nearest-neighbor index calculation."""

    def __init__(self, requested_device):
        self.torch = None
        self.device = 'cpu'
        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None:
            cuda_available = torch.cuda.is_available()
            if requested_device == 'cuda' and not cuda_available:
                raise RuntimeError(
                    '--device cuda was requested but CUDA is unavailable'
                )
            if requested_device == 'cuda' or (
                requested_device == 'auto' and cuda_available
            ):
                self.torch = torch
                self.device = 'cuda'

    def nearest(self, target_x, train_x, target_chunk=512):
        target_x = np.asarray(target_x, dtype=np.float32)
        train_x = np.asarray(train_x, dtype=np.float32)
        if len(train_x) == 0:
            raise ValueError('Nearest-neighbor candidate set is empty')
        if len(target_x) == 0:
            return np.empty(0, dtype=np.int64)

        if self.device == 'cpu':
            output = []
            train_squared = np.sum(
                train_x * train_x, axis=1, dtype=np.float64
            )
            train_64 = train_x.astype(np.float64, copy=False)
            for start in range(0, len(target_x), target_chunk):
                target = target_x[
                    start:start + target_chunk
                ].astype(np.float64, copy=False)
                # Some Accelerate-backed NumPy builds emit spurious floating
                # warnings for finite BLAS matmuls; the resulting values remain
                # finite and are checked below.
                with np.errstate(
                    over='ignore', divide='ignore', invalid='ignore'
                ):
                    distances = (
                        np.sum(target * target, axis=1, keepdims=True)
                        + train_squared[None, :]
                        - 2.0 * np.matmul(target, train_64.T)
                    )
                if not np.all(np.isfinite(distances)):
                    raise ValueError(
                        'Non-finite CPU distance values were produced'
                    )
                output.append(np.argmin(distances, axis=1))
            return np.concatenate(output).astype(np.int64)

        torch = self.torch
        with torch.no_grad():
            train = torch.as_tensor(
                train_x, dtype=torch.float32, device=self.device
            )
            train_squared = torch.sum(train * train, dim=1)
            output = []
            for start in range(0, len(target_x), target_chunk):
                target = torch.as_tensor(
                    target_x[start:start + target_chunk],
                    dtype=torch.float32,
                    device=self.device,
                )
                distances = (
                    torch.sum(target * target, dim=1, keepdim=True)
                    + train_squared.unsqueeze(0)
                    - 2.0 * torch.matmul(target, train.T)
                )
                output.append(
                    torch.argmin(distances, dim=1).cpu().numpy()
                )
        return np.concatenate(output).astype(np.int64)


def sentence_labels_from_records(records):
    labels = {}
    for record in records:
        sentence_id = record['sentence_id']
        label = int(record['label'])
        if sentence_id in labels and labels[sentence_id] != label:
            raise ValueError(
                f'Conflicting labels for sentence {sentence_id}'
            )
        labels[sentence_id] = label
    return labels


def split_record_indices(arrays, assignments):
    return {
        split: np.asarray([
            index
            for index, sentence_id in enumerate(arrays['sentence_ids'])
            if assignments[sentence_id] == split
        ], dtype=np.int64)
        for split in ('train', 'validation', 'test')
    }


def within_record_per_band_zscore(raw_features):
    bands = raw_features.reshape(
        len(raw_features), len(BANDS), ELECTRODES_PER_BAND
    )
    means = bands.mean(axis=2, keepdims=True)
    scales = safe_scale(bands, axis=2, keepdims=True)
    return ((bands - means) / scales).reshape(
        len(raw_features), FEATURE_WIDTH
    )


def subject_train_per_band_zscore(
    raw_features, subjects, train_indices
):
    bands = raw_features.reshape(
        len(raw_features), len(BANDS), ELECTRODES_PER_BAND
    )
    normalized = np.empty_like(bands)
    parameters = {}
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        subject_train_indices = train_indices[
            subjects[train_indices] == subject
        ]
        if len(subject_train_indices) == 0:
            raise ValueError(
                f'Subject {subject} has no training records'
            )
        training_bands = bands[subject_train_indices]
        mean = training_bands.mean(axis=(0, 2), keepdims=True)
        scale = safe_scale(
            training_bands, axis=(0, 2), keepdims=True
        )
        normalized[subject_mask] = (
            bands[subject_mask] - mean
        ) / scale
        parameters[str(subject)] = {
            'training_record_count': int(len(subject_train_indices)),
            'band_means': mean.reshape(-1).tolist(),
            'band_stds': scale.reshape(-1).tolist(),
        }
    return normalized.reshape(len(raw_features), FEATURE_WIDTH), parameters


def restricted_neighbor_indices(
    backend,
    features,
    subjects,
    train_indices,
    target_indices,
    relation,
):
    output = np.empty(len(target_indices), dtype=np.int64)
    target_subjects = subjects[target_indices]
    for subject in np.unique(target_subjects):
        target_positions = np.flatnonzero(target_subjects == subject)
        if relation == 'same':
            candidates = train_indices[subjects[train_indices] == subject]
        elif relation == 'different':
            candidates = train_indices[subjects[train_indices] != subject]
        else:
            raise ValueError(f'Unknown subject relation: {relation}')
        local_neighbors = backend.nearest(
            features[target_indices[target_positions]],
            features[candidates],
        )
        output[target_positions] = candidates[local_neighbors]
    return output


def neighbor_artifact(
    target_sentence_ids, neighbor_sentence_ids, sentence_vote
):
    return {
        'target_sentence_ids': np.asarray(
            target_sentence_ids, dtype=object
        ),
        'neighbor_sentence_ids': np.asarray(
            neighbor_sentence_ids, dtype=object
        ),
        'sentence_vote': bool(sentence_vote),
    }


def metrics_from_neighbor_artifact(
    artifact, train_label_map, target_label_map
):
    target_ids = artifact['target_sentence_ids']
    neighbor_ids = artifact['neighbor_sentence_ids']
    record_labels = np.asarray([
        target_label_map[sentence_id] for sentence_id in target_ids
    ], dtype=np.int64)
    record_predictions = np.asarray([
        train_label_map[sentence_id] for sentence_id in neighbor_ids
    ], dtype=np.int64)

    if not artifact['sentence_vote']:
        return metric_summary(record_labels, record_predictions)

    grouped_predictions = defaultdict(list)
    grouped_labels = {}
    for sentence_id, label, prediction in zip(
        target_ids, record_labels, record_predictions
    ):
        grouped_predictions[sentence_id].append(int(prediction))
        grouped_labels[sentence_id] = int(label)
    ordered_ids = sorted(grouped_predictions, key=stable_id_key)
    sentence_labels = np.asarray([
        grouped_labels[sentence_id] for sentence_id in ordered_ids
    ], dtype=np.int64)
    sentence_predictions = np.asarray([
        np.bincount(
            grouped_predictions[sentence_id], minlength=CLASS_NUM
        ).argmax()
        for sentence_id in ordered_ids
    ], dtype=np.int64)
    result = metric_summary(sentence_labels, sentence_predictions)
    result['sentence_count'] = len(ordered_ids)
    result['vote_tie_break'] = 'lowest class index'
    return result


def build_seed_artifacts(
    backend,
    arrays,
    aggregate_vectors,
    split_manifest,
    fixed_record_features,
):
    split_ids = split_manifest['split_ids']
    split_indices = split_record_indices(
        arrays, split_manifest['assignments']
    )
    train_indices = split_indices['train']

    aggregate_train_ids = np.asarray(
        split_ids['train'], dtype=object
    )
    aggregate_train_x = np.stack([
        aggregate_vectors[sentence_id]
        for sentence_id in aggregate_train_ids
    ])
    subject_features, subject_parameters = (
        subject_train_per_band_zscore(
            arrays['features'], arrays['subjects'], train_indices
        )
    )

    artifacts = {
        setup: {} for setup in SETUP_NAMES
    }
    for split in TARGET_SPLITS:
        target_ids = np.asarray(split_ids[split], dtype=object)
        target_x = np.stack([
            aggregate_vectors[sentence_id]
            for sentence_id in target_ids
        ])
        neighbors = backend.nearest(target_x, aggregate_train_x)
        artifacts[SETUP_A][split] = neighbor_artifact(
            target_ids, aggregate_train_ids[neighbors], False
        )

        target_indices = split_indices[split]
        same_neighbors = restricted_neighbor_indices(
            backend,
            fixed_record_features,
            arrays['subjects'],
            train_indices,
            target_indices,
            relation='same',
        )
        artifacts[SETUP_B][split] = neighbor_artifact(
            arrays['sentence_ids'][target_indices],
            arrays['sentence_ids'][same_neighbors],
            True,
        )

        different_neighbors = restricted_neighbor_indices(
            backend,
            subject_features,
            arrays['subjects'],
            train_indices,
            target_indices,
            relation='different',
        )
        artifacts[SETUP_C][split] = neighbor_artifact(
            arrays['sentence_ids'][target_indices],
            arrays['sentence_ids'][different_neighbors],
            True,
        )

    return artifacts, split_indices, subject_parameters


def permuted_label_maps(
    sentence_labels, split_ids, rng
):
    result = {}
    for split in ('train', 'validation', 'test'):
        ids = list(split_ids[split])
        values = [sentence_labels[sentence_id] for sentence_id in ids]
        rng.shuffle(values)
        result[split] = dict(zip(ids, values))
    return result


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'sample_std': (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        'median': float(np.median(values)),
        'minimum': float(np.min(values)),
        'maximum': float(np.max(values)),
        'quantile_2_5_percent': float(np.quantile(values, 0.025)),
        'quantile_97_5_percent': float(np.quantile(values, 0.975)),
    }


def cache_vector(item):
    return np.concatenate([
        np.asarray(item[band], dtype=np.float64) for band in BANDS
    ])


def load_or_build_released_cache(args):
    if os.path.exists(args.released_cache):
        with open(args.released_cache, 'rb') as cache_file:
            return pickle.load(cache_file), 'existing cache'

    from audit_eeg_features import load_or_build_eeg_dict

    cache_args = SimpleNamespace(
        eeg_cache=args.released_cache,
        eeg_dir=args.eeg_dir,
        labels_csv=args.labels_csv,
    )
    return load_or_build_eeg_dict(cache_args)


def compare_reconstruction(cached, reconstructed, sentence_labels):
    cache_ids = set(cached)
    reconstructed_ids = set(reconstructed)
    common_ids = cache_ids & reconstructed_ids
    differences = []
    exact_sentence_count = 0
    allclose_counts = {
        'atol_1e-7_rtol_0': 0,
        'atol_1e-6_rtol_0': 0,
        'atol_1e-5_rtol_0': 0,
    }
    label_mismatches = []

    for sentence_id in common_ids:
        cached_vector = cache_vector(cached[sentence_id])
        reconstructed_vector = np.asarray(
            reconstructed[sentence_id], dtype=np.float64
        )
        difference = np.abs(cached_vector - reconstructed_vector)
        differences.append(difference)
        exact_sentence_count += int(
            np.array_equal(cached_vector, reconstructed_vector)
        )
        for tolerance, key in (
            (1e-7, 'atol_1e-7_rtol_0'),
            (1e-6, 'atol_1e-6_rtol_0'),
            (1e-5, 'atol_1e-5_rtol_0'),
        ):
            allclose_counts[key] += int(np.allclose(
                cached_vector, reconstructed_vector,
                atol=tolerance, rtol=0.0,
            ))
        if int(cached[sentence_id]['label']) != int(
            sentence_labels[sentence_id]
        ):
            label_mismatches.append(json_scalar(sentence_id))

    if differences:
        all_differences = np.concatenate(differences)
        maximum = float(np.max(all_differences))
        mean = float(np.mean(all_differences))
        rms = float(np.sqrt(np.mean(all_differences ** 2)))
    else:
        maximum = mean = rms = None

    result = {
        'cache_sentence_count': len(cached),
        'reconstructed_sentence_count': len(reconstructed),
        'common_sentence_count': len(common_ids),
        'missing_in_reconstruction': sorted(
            [json_scalar(value) for value in cache_ids - common_ids],
            key=stable_id_key,
        ),
        'missing_in_cache': sorted(
            [json_scalar(value) for value in reconstructed_ids - common_ids],
            key=stable_id_key,
        ),
        'label_mismatches': label_mismatches,
        'exact_float_equality_sentence_count': exact_sentence_count,
        'allclose_sentence_counts': allclose_counts,
        'maximum_absolute_feature_difference': maximum,
        'mean_absolute_feature_difference': mean,
        'root_mean_square_feature_difference': rms,
        'strict_interpretation': (
            'Exact float equality may differ solely because float32 means '
            'depend on subject-file accumulation order. ID/label equality and '
            'allclose at 1e-5 are required for this audit.'
        ),
    }
    result['passes_required_reconstruction_check'] = (
        not result['missing_in_reconstruction']
        and not result['missing_in_cache']
        and not label_mismatches
        and allclose_counts['atol_1e-5_rtol_0'] == len(common_ids)
    )
    return result


def summarize_results(
    observed_by_seed, null_values, majority_by_seed
):
    summaries = {}
    for setup in SETUP_NAMES:
        summaries[setup] = {}
        for split in TARGET_SPLITS:
            setup_split = {}
            for metric in METRIC_NAMES:
                observed = np.asarray([
                    seed_result[setup][split][metric]
                    for seed_result in observed_by_seed
                ])
                majority = np.asarray([
                    seed_result[split][metric]
                    for seed_result in majority_by_seed
                ])
                null_matrix = np.asarray(
                    null_values[setup][split][metric]
                )
                null_across_seed_means = null_matrix.mean(axis=0)
                observed_mean = float(observed.mean())
                setup_split[metric] = {
                    'observed_seed_distribution':
                        distribution_summary(observed),
                    'majority_seed_distribution':
                        distribution_summary(majority),
                    'paired_difference_from_majority':
                        distribution_summary(observed - majority),
                    'seeds_above_majority': int(
                        np.sum(observed > majority)
                    ),
                    'seeds_equal_majority': int(
                        np.sum(observed == majority)
                    ),
                    'permutation_null_across_seed_mean_distribution':
                        distribution_summary(null_across_seed_means),
                    'empirical_p_null_mean_ge_observed_mean': float(
                        (
                            1
                            + np.sum(
                                null_across_seed_means >= observed_mean
                            )
                        )
                        / (len(null_across_seed_means) + 1)
                    ),
                }
            summaries[setup][split] = setup_split

        validation_accuracy = np.asarray([
            seed_result[setup]['validation']['accuracy']
            for seed_result in observed_by_seed
        ])
        test_accuracy = np.asarray([
            seed_result[setup]['test']['accuracy']
            for seed_result in observed_by_seed
        ])
        summaries[setup]['validation_test_accuracy_correlation'] = (
            float(np.corrcoef(
                validation_accuracy, test_accuracy
            )[0, 1])
            if np.std(validation_accuracy) > 0
            and np.std(test_accuracy) > 0
            else None
        )
    return summaries


def main():
    args = get_args()
    if args.num_seeds < 1:
        raise ValueError('--num_seeds must be at least 1')
    if args.permutations < 1:
        raise ValueError('--permutations must be at least 1')

    print('Loading raw subject-sentence recordings...')
    records, raw_file_metadata = load_subject_records(
        args.eeg_dir, args.labels_csv
    )
    arrays = records_to_arrays(records)
    if not np.all(np.isfinite(arrays['features'])):
        raise ValueError('Raw feature matrix contains non-finite values')
    sentence_labels = sentence_labels_from_records(records)

    first_manifest = canonical_stratified_sentence_split(
        sentence_labels, args.seed_start
    )
    _, aggregate_vectors = aggregate_subjects_by_sentence(
        records, first_manifest['assignments']
    )
    print('Loading or building the released post-aggregation cache...')
    released_cache, cache_source = load_or_build_released_cache(args)
    cache_comparison = compare_reconstruction(
        released_cache, aggregate_vectors, sentence_labels
    )
    if not cache_comparison['passes_required_reconstruction_check']:
        failure = {
            'audit_type': (
                'predeclared three-setup multi-seed ZuCo EEG 1-NN audit'
            ),
            'scientific_model_result': False,
            'code_commit': git_commit(),
            'status': 'aborted before scoring',
            'reason': 'released-cache reconstruction check failed',
            'released_cache_verification': {
                'cache_source': cache_source,
                **cache_comparison,
            },
        }
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w') as output_file:
            json.dump(failure, output_file, indent=2)
        print(json.dumps(failure, indent=2))
        raise ValueError(
            'Released-cache reconstruction check failed; failure details '
            f'were saved to {args.output_json}'
        )
    print(
        'Released-cache reconstruction passed; max absolute difference: '
        f'{cache_comparison["maximum_absolute_feature_difference"]:.9g}'
    )
    # Setup A must use the exact feature vectors consumed by the released
    # training path. The independently reconstructed copy above is only the
    # provenance/checking path.
    released_vectors = {
        sentence_id: cache_vector(item)
        for sentence_id, item in released_cache.items()
    }

    fixed_record_features = within_record_per_band_zscore(
        arrays['features']
    )
    backend = NearestNeighborBackend(args.device)
    seeds = list(range(
        args.seed_start, args.seed_start + args.num_seeds
    ))
    print(
        f'Evaluating {len(seeds)} split seeds on {backend.device}; '
        f'{args.permutations} label permutations per seed.'
    )

    observed_by_seed = []
    majority_by_seed = []
    split_manifests = []
    subject_normalization_parameters = {}
    null_values = {
        setup: {
            split: {
                metric: [
                    [] for _ in range(len(seeds))
                ]
                for metric in METRIC_NAMES
            }
            for split in TARGET_SPLITS
        }
        for setup in SETUP_NAMES
    }

    for seed_position, seed in enumerate(seeds):
        print(
            f'[{seed_position + 1:02d}/{len(seeds):02d}] split seed {seed}',
            flush=True,
        )
        manifest = canonical_stratified_sentence_split(
            sentence_labels, seed
        )
        artifacts, split_indices, subject_parameters = (
            build_seed_artifacts(
                backend,
                arrays,
                released_vectors,
                manifest,
                fixed_record_features,
            )
        )
        subject_normalization_parameters[str(seed)] = subject_parameters

        observed = {setup: {} for setup in SETUP_NAMES}
        majority = {}
        for split in TARGET_SPLITS:
            true_map = {
                sentence_id: sentence_labels[sentence_id]
                for sentence_id in manifest['split_ids'][split]
            }
            train_map = {
                sentence_id: sentence_labels[sentence_id]
                for sentence_id in manifest['split_ids']['train']
            }
            target_labels = np.asarray(list(true_map.values()))
            majority_class = int(np.bincount([
                train_map[sentence_id]
                for sentence_id in manifest['split_ids']['train']
            ], minlength=CLASS_NUM).argmax())
            majority[split] = metric_summary(
                target_labels,
                np.full_like(target_labels, majority_class),
            )
            for setup in SETUP_NAMES:
                observed[setup][split] = (
                    metrics_from_neighbor_artifact(
                        artifacts[setup][split],
                        train_map,
                        true_map,
                    )
                )

        for permutation in range(args.permutations):
            rng = random.Random(
                args.permutation_seed
                + seed * 1000003
                + permutation
            )
            permuted = permuted_label_maps(
                sentence_labels, manifest['split_ids'], rng
            )
            for setup in SETUP_NAMES:
                for split in TARGET_SPLITS:
                    null_metrics = metrics_from_neighbor_artifact(
                        artifacts[setup][split],
                        permuted['train'],
                        permuted[split],
                    )
                    for metric in METRIC_NAMES:
                        null_values[setup][split][metric][
                            seed_position
                        ].append(null_metrics[metric])

        observed_by_seed.append(observed)
        majority_by_seed.append(majority)
        split_manifests.append({
            'seed': seed,
            'algorithm': manifest['algorithm'],
            'split_ids': {
                split: [
                    json_scalar(value)
                    for value in manifest['split_ids'][split]
                ]
                for split in ('train', 'validation', 'test')
            },
            'record_counts': {
                split: int(len(split_indices[split]))
                for split in ('train', 'validation', 'test')
            },
        })

    result = {
        'audit_type': (
            'predeclared three-setup multi-seed ZuCo EEG 1-NN audit'
        ),
        'scientific_model_result': False,
        'paper_comparable_result': False,
        'code_commit': git_commit(),
        'protocol': {
            'seeds': seeds,
            'seed_count': len(seeds),
            'split_algorithm': first_manifest['algorithm'],
            'permutations_per_seed': args.permutations,
            'permutation_master_seed': args.permutation_seed,
            'permutation_unit': 'sentence ID, never individual recording',
            'permutation_scheme': (
                'labels are shuffled separately within train, validation, '
                'and test for each split seed; this preserves the exact '
                'sentence-level class counts in every split'
            ),
            'selection_warning': (
                'The three setups were chosen after v14 exploration. This '
                'audit measures split stability and label-association evidence; '
                'it does not erase the earlier selection or make test-guided '
                'preprocessing a confirmatory scientific result.'
            ),
        },
        'frozen_setups': {
            SETUP_A: {
                'unit': 'one vector per sentence',
                'features': (
                    'exact released-cache vector: float32 mean of all valid '
                    'participant recordings for each sentence, followed by a '
                    'separate Z-score across the 104 electrodes of each of 8 '
                    'bands'
                ),
                'neighbor_pool': 'all training sentences',
                'evaluation': 'one nearest-neighbor prediction per sentence',
            },
            SETUP_B: {
                'unit': 'one record per participant-sentence',
                'features': (
                    'for each record and band independently, Z-score the '
                    '104 electrodes'
                ),
                'neighbor_pool': (
                    'training records from the target participant only'
                ),
                'evaluation': (
                    'one prediction per available participant recording, '
                    'then majority vote per sentence; ties choose class 0'
                ),
            },
            SETUP_C: {
                'unit': 'one record per participant-sentence',
                'features': (
                    'one mean/std per participant and band, fitted from that '
                    'participant training records pooled across all 104 '
                    'electrodes; apply to all records of that participant'
                ),
                'neighbor_pool': (
                    'training records excluding the target participant'
                ),
                'evaluation': (
                    'one prediction per available participant recording, '
                    'then majority vote per sentence; ties choose class 0'
                ),
                'methodology_limit': (
                    'not LOSO: the target participant has other sentences in '
                    'training and supplies unlabeled training-split statistics '
                    'to its fitted normalization'
                ),
            },
        },
        'dataset': {
            'subject_count': int(len(np.unique(arrays['subjects']))),
            'sentence_count': len(sentence_labels),
            'valid_subject_sentence_record_count': len(records),
            'sentence_class_counts': np.bincount(
                list(sentence_labels.values()), minlength=CLASS_NUM
            ).astype(int).tolist(),
            'class_names': list(CLASS_NAMES),
            'raw_file_metadata': raw_file_metadata,
        },
        'released_cache_verification': {
            'cache_source': cache_source,
            **cache_comparison,
        },
        'runtime': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'distance_backend': backend.device,
            'cuda_device': (
                backend.torch.cuda.get_device_name(0)
                if backend.device == 'cuda' else None
            ),
        },
        'summary_across_seeds': summarize_results(
            observed_by_seed, null_values, majority_by_seed
        ),
        'per_seed_observed_metrics': [
            {
                'seed': seed,
                'majority': majority_by_seed[position],
                'setups': observed_by_seed[position],
            }
            for position, seed in enumerate(seeds)
        ],
        'split_manifests': split_manifests,
        'subject_normalization_parameters_by_seed':
            subject_normalization_parameters,
        'permutation_null_raw': {
            setup: {
                split: {
                    metric: values
                    for metric, values in metrics.items()
                }
                for split, metrics in splits.items()
            }
            for setup, splits in null_values.items()
        },
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w') as output_file:
        json.dump(result, output_file, indent=2)

    concise = {
        'output_json': args.output_json,
        'code_commit': result['code_commit'],
        'distance_backend': backend.device,
        'released_cache_verification':
            result['released_cache_verification'],
        'summary_across_seeds': result['summary_across_seeds'],
    }
    print(json.dumps(concise, indent=2))


if __name__ == '__main__':
    main()
