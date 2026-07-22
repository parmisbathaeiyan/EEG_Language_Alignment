"""Run the single bounded v16b ZuCo feature-construction follow-up.

This audit does not train or select a model. It tests only the concrete
questions raised by v16: whether ``mean_*_sec`` reconstructs ``mean_*``,
whether FFD/TRT duration weighting helps, whether differences survive the
released first-104/per-band Z-score, and how the stored 105th value behaves.
"""

import argparse
import gc
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict

import numpy as np
import scipy
import scipy.io as io

from audit_reproduction_features import (
    BANDS,
    ComparisonAccumulator,
    RELEASED_EXCLUDED_FILE,
    field_names,
    sentence_words,
)


STORED_CHANNELS = 105
RELEASED_CHANNELS = 104
REPRESENTATIONS = (
    'sentence_mean_sec_unweighted',
    'word_FFD_unweighted',
    'word_FFD_duration_weighted',
    'word_TRT_unweighted',
    'word_TRT_duration_weighted',
)
LABEL_HINTS = ('chan', 'electrode', 'label', 'sensor', 'montage', 'cap')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_dir', default='data/SR')
    parser.add_argument('--output_json', required=True)
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def subject_id(filename):
    stem = os.path.splitext(filename)[0]
    if stem.startswith('results'):
        stem = stem[len('results'):]
    if stem.endswith('_SR'):
        stem = stem[:-len('_SR')]
    return stem


def numeric_vector(record, field, width=STORED_CHANNELS):
    if not hasattr(record, field):
        return None
    try:
        values = np.asarray(
            getattr(record, field), dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < width:
        return None
    return values[:width].copy()


def released_vector(values):
    if values is None or len(values) < RELEASED_CHANNELS:
        return None
    result = np.asarray(
        values[:RELEASED_CHANNELS], dtype=np.float64
    ).copy()
    nan_mask = np.isnan(result)
    if np.all(nan_mask):
        return None
    result[nan_mask] = 0.0
    return result if np.all(np.isfinite(result)) else None


def released_expanded_105_vector(values):
    """Apply the released partial-NaN policy, hypothetically, to all 105."""
    if values is None or len(values) < STORED_CHANNELS:
        return None
    result = np.asarray(values[:STORED_CHANNELS], dtype=np.float64).copy()
    if np.all(np.isnan(result[:RELEASED_CHANNELS])):
        return None
    result[np.isnan(result)] = 0.0
    return result if np.all(np.isfinite(result)) else None


def finite_column_mean(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, matrix, 0.0).sum(axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=counts != 0,
    )


def mean_sec_vector(sentence, band, status):
    field = f'mean_{band}_sec'
    status['expected_records'] += 1
    if not hasattr(sentence, field):
        status['field_missing'] += 1
        return None
    value = getattr(sentence, field)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        status['not_numeric'] += 1
        return None
    status[f'shape:{"x".join(map(str, array.shape)) or "scalar"}'] += 1
    if array.ndim == 1 and array.size == STORED_CHANNELS:
        rows = array[:STORED_CHANNELS].reshape(1, STORED_CHANNELS)
        status['channel_axis:last'] += 1
    elif array.ndim >= 2 and array.shape[-1] == STORED_CHANNELS:
        rows = array.reshape(-1, array.shape[-1])[:, :STORED_CHANNELS]
        status['channel_axis:last'] += 1
    elif array.ndim == 2 and array.shape[0] == STORED_CHANNELS:
        rows = array[:STORED_CHANNELS, :].T
        status['channel_axis:first'] += 1
    else:
        status['unusable_shape'] += 1
        return None
    candidate = finite_column_mean(rows)
    if not np.any(np.isfinite(candidate[:RELEASED_CHANNELS])):
        status['all_nonfinite'] += 1
        return None
    status['usable_records'] += 1
    status['usable_rows'] += int(len(rows))
    return candidate


def scalar_duration(word, field, status):
    status['expected_words'] += 1
    if not hasattr(word, field):
        status['field_missing'] += 1
        return None
    status['field_present'] += 1
    try:
        values = np.asarray(
            getattr(word, field), dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError):
        status['not_numeric'] += 1
        return None
    if values.size != 1:
        status['not_scalar'] += 1
        return None
    duration = float(values[0])
    if not np.isfinite(duration):
        status['nonfinite'] += 1
        return None
    if duration <= 0:
        status['nonpositive'] += 1
        return None
    status['usable_positive_scalar'] += 1
    return duration


def word_window_candidates(words, window, duration_status):
    rows_by_band = defaultdict(list)
    weighted_by_band = defaultdict(list)
    for word in words:
        duration = scalar_duration(
            word, window, duration_status[window]
        )
        for band in BANDS:
            vector = numeric_vector(word, f'{window}_{band}')
            if vector is None:
                continue
            if not np.any(np.isfinite(vector[:RELEASED_CHANNELS])):
                continue
            rows_by_band[band].append(vector)
            if duration is not None:
                weighted_by_band[band].append((duration, vector))

    unweighted = {}
    weighted = {}
    for band in BANDS:
        rows = rows_by_band[band]
        if rows:
            unweighted[band] = finite_column_mean(np.stack(rows))
        pairs = weighted_by_band[band]
        if pairs:
            weights = np.asarray(
                [weight for weight, _ in pairs], dtype=np.float64
            )
            matrix = np.stack([vector for _, vector in pairs])
            finite = np.isfinite(matrix)
            numerator = np.where(
                finite, matrix * weights[:, None], 0.0
            ).sum(axis=0)
            denominator = np.where(
                finite, weights[:, None], 0.0
            ).sum(axis=0)
            weighted[band] = np.divide(
                numerator,
                denominator,
                out=np.full(STORED_CHANNELS, np.nan, dtype=np.float64),
                where=denominator != 0,
            )
    return unweighted, weighted


def all_bands_usable(representation):
    return all(
        band in representation
        and released_vector(representation[band]) is not None
        for band in BANDS
    )


def zscore_across_channels(values):
    if values is None:
        return None
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        return None
    standard_deviation = float(np.std(values, ddof=0))
    if standard_deviation <= 1e-12:
        return None
    return (values - float(np.mean(values))) / standard_deviation


def mean_vectors(vectors):
    # Mirrors dataset_new.py, which converts participant vectors to float32
    # immediately before cross-participant averaging.
    return np.mean(np.stack(vectors).astype(np.float32), axis=0)


def new_comparison_table():
    return {
        representation: {
            band: ComparisonAccumulator() for band in BANDS
        }
        for representation in REPRESENTATIONS
    }


def finalize_comparison_table(table):
    output = {}
    for representation in REPRESENTATIONS:
        output[representation] = {}
        for band in BANDS:
            result = table[representation][band].result()
            if result is not None:
                output[representation][band] = result
    return output


def aggregate_comparisons(results):
    summary = {}
    for representation, band_results in results.items():
        usable = list(band_results.values())
        if not usable:
            continue
        relative = [
            result['relative_rmse'] for result in usable
            if result['relative_rmse'] is not None
        ]
        correlations = [
            result['flattened_pearson_correlation'] for result in usable
            if result['flattened_pearson_correlation'] is not None
        ]
        summary[representation] = {
            'bands': len(usable),
            'mean_relative_rmse': (
                float(np.mean(relative)) if relative else None
            ),
            'minimum_band_relative_rmse': (
                float(np.min(relative)) if relative else None
            ),
            'maximum_band_relative_rmse': (
                float(np.max(relative)) if relative else None
            ),
            'mean_flattened_correlation': (
                float(np.mean(correlations)) if correlations else None
            ),
            'total_vectors_with_max_abs_error_at_most_0_001': int(sum(
                result['whole_vectors_with_max_abs_error_at_most']['0.001']
                for result in usable
            )),
            'total_sentence_band_vector_pairs': int(sum(
                result['sentence_band_vector_pairs'] for result in usable
            )),
        }
    return summary


def counter_result(counter):
    return {key: int(value) for key, value in sorted(counter.items())}


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        'count': int(len(values)),
        'minimum': float(np.min(values)),
        'median': float(np.median(values)),
        'mean': float(np.mean(values)),
        'maximum': float(np.max(values)),
    }


def metadata_snapshot(value):
    try:
        array = np.asarray(value)
        shape = list(array.shape)
        dtype = str(array.dtype)
    except Exception:
        shape = None
        dtype = type(value).__name__
    return {
        'python_type': type(value).__name__,
        'shape': shape,
        'dtype': dtype,
        'mat_struct_fields': list(field_names(value)),
    }


def collect_strings(value, depth=0, limit=300):
    """Collect bounded string metadata without traversing raw numeric data."""
    if depth > 3 or limit <= 0:
        return []
    if isinstance(value, bytes):
        return [value.decode('utf-8', errors='replace')]
    if isinstance(value, str):
        return [value]
    names = field_names(value)
    if names:
        strings = []
        for name in names:
            strings.extend(collect_strings(
                getattr(value, name), depth + 1, limit - len(strings)
            ))
            if len(strings) >= limit:
                break
        return strings[:limit]
    try:
        array = np.asarray(value)
    except Exception:
        return []
    if array.dtype.kind in ('U', 'S'):
        return [str(item) for item in array.reshape(-1)[:limit]]
    if array.dtype != object:
        return []
    strings = []
    for item in array.reshape(-1):
        strings.extend(collect_strings(
            item, depth + 1, limit - len(strings)
        ))
        if len(strings) >= limit:
            break
    return strings[:limit]


def label_like_metadata(loaded, first_sentence):
    candidates = {}

    def inspect(container_name, container):
        if isinstance(container, dict):
            names = [name for name in container if not name.startswith('__')]
            getter = container.get
        else:
            names = field_names(container)
            getter = lambda name: getattr(container, name)
        for name in names:
            if any(hint in name.lower() for hint in LABEL_HINTS):
                value = getter(name)
                candidates[f'{container_name}.{name}'] = {
                    'snapshot': metadata_snapshot(value),
                    'string_values': collect_strings(value),
                }

    inspect('mat', loaded)
    inspect('sentenceData[0]', first_sentence)
    if hasattr(first_sentence, 'rawData'):
        inspect('sentenceData[0].rawData', first_sentence.rawData)
    return candidates


class LastChannelStats:
    def __init__(self):
        self.records = 0
        self.finite_values = []
        self.nan = 0
        self.infinite = 0
        self.exact_zero = 0

    def update(self, vector):
        if vector is None or len(vector) < STORED_CHANNELS:
            return
        self.records += 1
        value = float(vector[STORED_CHANNELS - 1])
        if np.isnan(value):
            self.nan += 1
        elif not np.isfinite(value):
            self.infinite += 1
        else:
            self.finite_values.append(value)
            if value == 0.0:
                self.exact_zero += 1

    def result(self):
        values = np.asarray(self.finite_values, dtype=np.float64)
        return {
            'stored_vectors_seen': self.records,
            'finite_105th_values': int(len(values)),
            'nan_105th_values': self.nan,
            'infinite_105th_values': self.infinite,
            'exact_zero_105th_values': self.exact_zero,
            'finite_value_summary': None if not len(values) else {
                'minimum': float(np.min(values)),
                'mean': float(np.mean(values)),
                'standard_deviation': float(np.std(values, ddof=0)),
                'maximum': float(np.max(values)),
            },
        }


def main():
    args = get_args()
    filenames = sorted(
        filename for filename in os.listdir(args.eeg_dir)
        if filename.endswith('.mat')
    )
    if not filenames:
        raise ValueError(f'No .mat files found in {args.eeg_dir}')

    included_files = []
    excluded_files = []
    reference_content = None
    content_mismatches = []
    records = defaultdict(dict)
    raw_104 = new_comparison_table()
    raw_105 = new_comparison_table()
    mean_sec_status = {band: Counter() for band in BANDS}
    duration_status = {'FFD': Counter(), 'TRT': Counter()}
    candidate_record_counts = Counter()
    last_channel_stats = {
        band: LastChannelStats() for band in BANDS
    }
    metadata = {}

    for filename in filenames:
        if filename == RELEASED_EXCLUDED_FILE:
            excluded_files.append({
                'filename': filename,
                'reason': 'excluded by the released sentence loader',
            })
            continue
        path = os.path.join(args.eeg_dir, filename)
        loaded = io.loadmat(path, squeeze_me=True, struct_as_record=False)
        if 'sentenceData' not in loaded:
            raise KeyError(f'{filename} has no sentenceData variable')
        sentences = np.atleast_1d(loaded['sentenceData']).reshape(-1)
        current_content = [str(sentence.content) for sentence in sentences]
        if reference_content is None:
            reference_content = current_content
        else:
            if len(current_content) != len(reference_content):
                content_mismatches.append({
                    'filename': filename,
                    'expected_sentence_count': len(reference_content),
                    'observed_sentence_count': len(current_content),
                })
            for index, (expected, observed) in enumerate(zip(
                reference_content, current_content
            )):
                if expected != observed:
                    content_mismatches.append({
                        'filename': filename,
                        'sentence_index': index,
                        'expected': expected,
                        'observed': observed,
                    })
        first_sentence = sentences[0]
        metadata[filename] = {
            'top_level_mat_variables': sorted(
                key for key in loaded if not key.startswith('__')
            ),
            'sentence_rawData': (
                metadata_snapshot(first_sentence.rawData)
                if hasattr(first_sentence, 'rawData') else None
            ),
            'label_like_metadata_candidates': label_like_metadata(
                loaded, first_sentence
            ),
        }

        subject = subject_id(filename)
        valid_targets = 0
        for sentence_index, sentence in enumerate(sentences):
            target_105 = {
                band: numeric_vector(sentence, f'mean_{band}')
                for band in BANDS
            }
            for band in BANDS:
                last_channel_stats[band].update(target_105[band])
            target_104 = {
                band: released_vector(target_105[band])
                for band in BANDS
            }
            if not all(target_104[band] is not None for band in BANDS):
                continue
            valid_targets += 1

            representations = {}
            mean_sec = {
                band: mean_sec_vector(
                    sentence, band, mean_sec_status[band]
                )
                for band in BANDS
            }
            if all_bands_usable(mean_sec):
                representations['sentence_mean_sec_unweighted'] = mean_sec

            words = sentence_words(sentence)
            for window in ('FFD', 'TRT'):
                unweighted, weighted = word_window_candidates(
                    words, window, duration_status
                )
                if all_bands_usable(unweighted):
                    representations[f'word_{window}_unweighted'] = unweighted
                if all_bands_usable(weighted):
                    representations[
                        f'word_{window}_duration_weighted'
                    ] = weighted

            stored_record = {
                'target_104': {
                    band: target_104[band]
                    for band in BANDS
                },
                'target_105': {
                    band: released_expanded_105_vector(target_105[band])
                    for band in BANDS
                },
                'representations': {},
            }
            for representation, candidate in representations.items():
                candidate_record_counts[representation] += 1
                stored_record['representations'][representation] = {
                    band: released_vector(candidate[band])
                    for band in BANDS
                }
                for band in BANDS:
                    raw_104[representation][band].update(
                        target_104[band], candidate[band][:RELEASED_CHANNELS]
                    )
                    raw_105[representation][band].update(
                        target_105[band], candidate[band]
                    )
            records[sentence_index][subject] = stored_record

        included_files.append({
            'filename': filename,
            'subject_id': subject,
            'bytes': int(os.path.getsize(path)),
            'sentence_records': int(len(sentences)),
            'valid_released_target_records': valid_targets,
        })
        print(
            f'{filename}: {len(sentences)} sentences, '
            f'{valid_targets} valid released sentence records'
        )
        del sentences, loaded
        gc.collect()

    if content_mismatches:
        raise ValueError(
            f'Sentence order/content differed across participant files; '
            f'examples: {content_mismatches[:3]}'
        )

    final_vs_released_all = new_comparison_table()
    final_vs_same_subjects = new_comparison_table()
    contributor_counts = {
        representation: [] for representation in REPRESENTATIONS
    }
    released_target_contributors = []
    channel_inclusion_effect = {
        band: ComparisonAccumulator() for band in BANDS
    }

    for sentence_index in sorted(records):
        subject_records = records[sentence_index]
        target_records = list(subject_records.values())
        released_target_contributors.append(len(target_records))
        target_all_104 = {
            band: mean_vectors([
                record['target_104'][band] for record in target_records
            ])
            for band in BANDS
        }
        target_all_105 = {}
        for band in BANDS:
            expanded_vectors = [
                record['target_105'][band] for record in target_records
                if record['target_105'][band] is not None
            ]
            target_all_105[band] = (
                mean_vectors(expanded_vectors) if expanded_vectors else None
            )

        for band in BANDS:
            z_104 = zscore_across_channels(target_all_104[band])
            z_105 = zscore_across_channels(target_all_105[band])
            if z_104 is not None and z_105 is not None:
                channel_inclusion_effect[band].update(
                    z_104, z_105[:RELEASED_CHANNELS]
                )

        for representation in REPRESENTATIONS:
            eligible = [
                record for record in target_records
                if representation in record['representations']
            ]
            contributor_counts[representation].append(len(eligible))
            if not eligible:
                continue
            for band in BANDS:
                candidate_mean = mean_vectors([
                    record['representations'][representation][band]
                    for record in eligible
                ])
                same_subject_target = mean_vectors([
                    record['target_104'][band] for record in eligible
                ])
                candidate_z = zscore_across_channels(candidate_mean)
                same_subject_target_z = zscore_across_channels(
                    same_subject_target
                )
                released_all_target_z = zscore_across_channels(
                    target_all_104[band]
                )
                if candidate_z is None:
                    continue
                if released_all_target_z is not None:
                    final_vs_released_all[representation][band].update(
                        released_all_target_z, candidate_z
                    )
                if same_subject_target_z is not None:
                    final_vs_same_subjects[representation][band].update(
                        same_subject_target_z, candidate_z
                    )

    raw_104_results = finalize_comparison_table(raw_104)
    raw_105_results = finalize_comparison_table(raw_105)
    released_all_results = finalize_comparison_table(
        final_vs_released_all
    )
    same_subject_results = finalize_comparison_table(
        final_vs_same_subjects
    )
    channel_effect_results = {}
    for band, accumulator in channel_inclusion_effect.items():
        result = accumulator.result()
        if result is not None:
            channel_effect_results[band] = result

    output = {
        'audit': 'v16b bounded ZuCo feature-construction follow-up',
        'scope': {
            'included': [
                'mean_<band>_sec row averaging',
                'unweighted and duration-weighted FFD/TRT pooling',
                'comparison after released first-104 per-band Z-scoring',
                '105th stored value summary and metadata presence check',
            ],
            'excluded': [
                'model training',
                'raw-voltage analysis',
                'feature selection by label or test performance',
                'additional pooling searches after this run',
            ],
        },
        'provenance': {
            'git_commit': git_commit(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'eeg_dir': os.path.abspath(args.eeg_dir),
            'stored_channels_per_band': STORED_CHANNELS,
            'released_channels_per_band': RELEASED_CHANNELS,
            'band_order': list(BANDS),
        },
        'source': {
            'included_files': included_files,
            'excluded_files': excluded_files,
            'sentence_indices_grouped': len(records),
            'valid_released_participant_sentence_records': int(sum(
                item['valid_released_target_records']
                for item in included_files
            )),
            'content_order_verified_identical_across_files': True,
        },
        'candidate_definitions': {
            'sentence_mean_sec_unweighted': (
                'NaN-aware unweighted row mean of stored mean_<band>_sec'
            ),
            'word_FFD_unweighted': (
                'NaN-aware unweighted mean of available stored FFD_<band> '
                'word arrays'
            ),
            'word_FFD_duration_weighted': (
                'Mean of available FFD_<band> arrays weighted by each '
                'word\'s positive finite scalar FFD duration'
            ),
            'word_TRT_unweighted': (
                'NaN-aware unweighted mean of available stored TRT_<band> '
                'word arrays'
            ),
            'word_TRT_duration_weighted': (
                'Mean of available TRT_<band> arrays weighted by each '
                'word\'s positive finite scalar TRT duration'
            ),
        },
        'coverage': {
            'candidate_complete_participant_sentence_records': {
                representation: int(candidate_record_counts[representation])
                for representation in REPRESENTATIONS
            },
            'mean_sec_status_by_band': {
                band: counter_result(mean_sec_status[band])
                for band in BANDS
            },
            'duration_status': {
                window: counter_result(duration_status[window])
                for window in ('FFD', 'TRT')
            },
            'released_target_contributors_per_sentence': distribution(
                released_target_contributors
            ),
            'candidate_contributors_per_sentence': {
                representation: distribution(counts)
                for representation, counts in contributor_counts.items()
            },
        },
        'participant_level_raw_comparison_first_104': {
            'by_representation_and_band': raw_104_results,
            'aggregate_across_bands': aggregate_comparisons(
                raw_104_results
            ),
        },
        'participant_level_raw_comparison_all_105': {
            'by_representation_and_band': raw_105_results,
            'aggregate_across_bands': aggregate_comparisons(
                raw_105_results
            ),
        },
        'final_sentence_representation_after_released_zscore': {
            'explanation': (
                'For each sentence, participant vectors were averaged, then '
                'each 104-electrode band was independently Z-scored exactly '
                'as in the released loader. Candidate representations use '
                'their available participant records.'
            ),
            'candidate_vs_released_all_subject_target': {
                'by_representation_and_band': released_all_results,
                'aggregate_across_bands': aggregate_comparisons(
                    released_all_results
                ),
            },
            'candidate_vs_target_using_same_contributing_subjects': {
                'by_representation_and_band': same_subject_results,
                'aggregate_across_bands': aggregate_comparisons(
                    same_subject_results
                ),
            },
        },
        'stored_105th_value': {
            'per_band': {
                band: last_channel_stats[band].result()
                for band in BANDS
            },
            'effect_of_including_105th_value_on_first_104_zscores': {
                'hypothetical_preprocessing': (
                    'Extend the released partial-NaN-to-zero rule to the '
                    '105th entry, use the released float32 participant '
                    'averaging, Z-score all 105 values, then compare the '
                    'first 104 coordinates with the released 104-value '
                    'Z-score.'
                ),
                'by_band': channel_effect_results,
                'aggregate': aggregate_comparisons({
                    'include_105_then_compare_first_104':
                        channel_effect_results
                }),
            },
            'metadata_snapshots': metadata,
            'metadata_search_method': (
                'Searched top-level MAT, first sentence, and first sentence '
                'rawData fields whose names contain channel/electrode/label/'
                'sensor/montage/cap; recursively saved string values only.'
            ),
            'interpretation_limit': (
                'Finite/variable behavior cannot by itself identify the '
                'channel. If no direct labels are present, E129/Cz remains '
                'an inference from official ZuCo and repository channel lists.'
            ),
        },
        'hard_stop': (
            'This run consumes the one targeted post-v16 construction '
            'follow-up. Record unresolved ambiguities and proceed to the '
            'frozen released 832-feature MLP; do not add further pooling '
            'searches based on these results.'
        ),
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w') as output_file:
        json.dump(output, output_file, indent=2, allow_nan=False)

    print('\nParticipant-level raw reconstruction (first 104)')
    for name, result in output[
        'participant_level_raw_comparison_first_104'
    ]['aggregate_across_bands'].items():
        print(
            f"  {name}: relative_RMSE={result['mean_relative_rmse']}, "
            f"corr={result['mean_flattened_correlation']}, "
            f"vectors<=0.001={result['total_vectors_with_max_abs_error_at_most_0_001']}"
        )
    print('\nFinal representation after released first-104 Z-score')
    final_summary = output[
        'final_sentence_representation_after_released_zscore'
    ]['candidate_vs_released_all_subject_target'][
        'aggregate_across_bands'
    ]
    for name, result in final_summary.items():
        print(
            f"  {name}: relative_RMSE={result['mean_relative_rmse']}, "
            f"corr={result['mean_flattened_correlation']}, "
            f"vectors<=0.001={result['total_vectors_with_max_abs_error_at_most_0_001']}"
        )
    print('\n105th stored value')
    for band in BANDS:
        result = output['stored_105th_value']['per_band'][band]
        finite = result['finite_value_summary']
        print(
            f"  {band}: finite={result['finite_105th_values']}, "
            f"NaN={result['nan_105th_values']}, "
            f"std={None if finite is None else finite['standard_deviation']}"
        )
    print('\nSaved:', args.output_json)


if __name__ == '__main__':
    main()
