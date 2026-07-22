"""Audit the released ZuCo sentence/word EEG feature construction.

This is the bounded v16 reproduction audit. It does not split data, fit a
model, inspect raw voltage, or choose a representation by test performance.
It inventories the precomputed frequency-band fields used or advertised by
the paper/repository and tests whether common word-level pooling rules
reconstruct each participant file's supplied sentence-level ``mean_*`` fields.
"""

import argparse
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict

import numpy as np
import scipy
import scipy.io as io


BANDS = ('t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2')
WINDOWS = ('FFD', 'SFD', 'GD', 'GPT', 'TRT')
ELECTRODES = 104
RELEASED_EXCLUDED_FILE = 'resultsZDN_SR.mat'
TOLERANCES = (1e-7, 1e-5, 1e-3)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_dir', default='data/SR')
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--electrodes', type=int, default=ELECTRODES)
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def field_names(value):
    return tuple(getattr(value, '_fieldnames', ()) or ())


def shape_text(value):
    try:
        return 'x'.join(map(str, np.asarray(value).shape)) or 'scalar'
    except Exception:
        return 'unavailable'


def dtype_text(value):
    try:
        return str(np.asarray(value).dtype)
    except Exception:
        return type(value).__name__


def update_field_inventory(inventory, record):
    for name in field_names(record):
        value = getattr(record, name)
        entry = inventory[name]
        entry['occurrences'] += 1
        entry['shapes'][shape_text(value)] += 1
        entry['dtypes'][dtype_text(value)] += 1


def finalize_inventory(inventory):
    result = {}
    for name in sorted(inventory):
        entry = inventory[name]
        result[name] = {
            'occurrences': int(entry['occurrences']),
            'shapes': dict(sorted(entry['shapes'].items())),
            'dtypes': dict(sorted(entry['dtypes'].items())),
        }
    return result


def empty_feature_status():
    return Counter({
        'expected_records': 0,
        'field_present': 0,
        'numeric_array': 0,
        'usable_104_electrodes': 0,
        'too_short': 0,
        'all_nan': 0,
        'partial_nan': 0,
        'fully_finite': 0,
    })


def feature_vector(record, field, electrodes, status):
    status['expected_records'] += 1
    if not hasattr(record, field):
        return None
    status['field_present'] += 1
    try:
        values = np.asarray(getattr(record, field), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    status['numeric_array'] += 1
    if values.size < electrodes:
        status['too_short'] += 1
        return None
    values = values[:electrodes].copy()
    nan_mask = np.isnan(values)
    if np.all(nan_mask):
        status['all_nan'] += 1
        return None
    status['usable_104_electrodes'] += 1
    if np.any(nan_mask):
        status['partial_nan'] += 1
    elif np.all(np.isfinite(values)):
        status['fully_finite'] += 1
    return values


def sentence_words(sentence):
    if not hasattr(sentence, 'word'):
        return []
    value = getattr(sentence, 'word')
    try:
        words = np.asarray(value, dtype=object).reshape(-1).tolist()
    except Exception:
        words = [value]
    return [word for word in words if field_names(word)]


def nanmean_rows(rows):
    matrix = np.stack(rows)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, matrix, 0.0).sum(axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=counts != 0,
    )


def pooling_candidates(rows, total_words):
    if not rows:
        return {}
    matrix = np.stack(rows)
    zero_filled = np.where(np.isfinite(matrix), matrix, 0.0)
    candidates = {
        'nanmean_available_arrays': nanmean_rows(rows),
        'zero_fill_then_mean_available_arrays': zero_filled.mean(axis=0),
    }
    if total_words:
        candidates['zero_fill_missing_then_mean_all_words'] = (
            zero_filled.sum(axis=0) / total_words
        )
    complete = matrix[np.all(np.isfinite(matrix), axis=1)]
    if len(complete):
        candidates['mean_fully_finite_arrays'] = complete.mean(axis=0)
    return candidates


class ComparisonAccumulator:
    def __init__(self):
        self.vector_pairs = 0
        self.value_pairs = 0
        self.sum_abs = 0.0
        self.sum_sq_error = 0.0
        self.sum_target_sq = 0.0
        self.max_abs = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.within = Counter({str(tolerance): 0 for tolerance in TOLERANCES})

    def update(self, target, candidate):
        mask = np.isfinite(target) & np.isfinite(candidate)
        if not np.any(mask):
            return
        x = target[mask]
        y = candidate[mask]
        difference = y - x
        absolute = np.abs(difference)
        self.vector_pairs += 1
        self.value_pairs += int(mask.sum())
        self.sum_abs += float(absolute.sum())
        self.sum_sq_error += float(np.square(difference).sum())
        self.sum_target_sq += float(np.square(x).sum())
        self.max_abs = max(self.max_abs, float(absolute.max()))
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.square(x).sum())
        self.sum_y2 += float(np.square(y).sum())
        self.sum_xy += float((x * y).sum())
        for tolerance in TOLERANCES:
            if mask.all() and np.all(absolute <= tolerance):
                self.within[str(tolerance)] += 1

    def result(self):
        if not self.value_pairs:
            return None
        count = self.value_pairs
        rmse = float(np.sqrt(self.sum_sq_error / count))
        target_rms = float(np.sqrt(self.sum_target_sq / count))
        covariance = self.sum_xy - self.sum_x * self.sum_y / count
        variance_x = self.sum_x2 - self.sum_x * self.sum_x / count
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / count
        denominator = np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
        correlation = float(covariance / denominator) if denominator > 0 else None
        return {
            'sentence_band_vector_pairs': self.vector_pairs,
            'finite_electrode_value_pairs': self.value_pairs,
            'mean_absolute_error': self.sum_abs / count,
            'root_mean_square_error': rmse,
            'target_root_mean_square': target_rms,
            'relative_rmse': rmse / target_rms if target_rms > 0 else None,
            'maximum_absolute_error': self.max_abs,
            'flattened_pearson_correlation': correlation,
            'whole_vectors_with_max_abs_error_at_most': {
                tolerance: int(value)
                for tolerance, value in self.within.items()
            },
        }


def nested_status_result(statuses):
    return {
        field: {key: int(value) for key, value in status.items()}
        for field, status in sorted(statuses.items())
    }


def main():
    args = get_args()
    if not os.path.isdir(args.eeg_dir):
        raise FileNotFoundError(args.eeg_dir)

    filenames = sorted(
        filename for filename in os.listdir(args.eeg_dir)
        if filename.endswith('.mat')
    )
    if not filenames:
        raise ValueError(f'No .mat files found in {args.eeg_dir}')

    sentence_inventory = defaultdict(
        lambda: {'occurrences': 0, 'shapes': Counter(), 'dtypes': Counter()}
    )
    word_inventory = defaultdict(
        lambda: {'occurrences': 0, 'shapes': Counter(), 'dtypes': Counter()}
    )
    sentence_status = defaultdict(empty_feature_status)
    word_status = defaultdict(empty_feature_status)
    comparisons = defaultdict(ComparisonAccumulator)
    per_file = {}
    excluded = []
    total_sentences = 0
    total_words = 0

    for filename in filenames:
        if filename == RELEASED_EXCLUDED_FILE:
            excluded.append({
                'filename': filename,
                'reason': 'excluded by the released sentence loader',
            })
            continue
        path = os.path.join(args.eeg_dir, filename)
        loaded = io.loadmat(path, squeeze_me=True, struct_as_record=False)
        if 'sentenceData' not in loaded:
            raise KeyError(f'{filename} has no sentenceData variable')
        sentences = np.atleast_1d(loaded['sentenceData']).reshape(-1)
        file_words = 0
        sentences_with_words = 0

        for sentence in sentences:
            total_sentences += 1
            update_field_inventory(sentence_inventory, sentence)
            words = sentence_words(sentence)
            if words:
                sentences_with_words += 1
            file_words += len(words)
            total_words += len(words)
            for word in words:
                update_field_inventory(word_inventory, word)

            targets = {}
            for band in BANDS:
                field = f'mean_{band}'
                targets[band] = feature_vector(
                    sentence, field, args.electrodes, sentence_status[field]
                )

            rows_by_window_band = defaultdict(list)
            for word in words:
                for window in WINDOWS:
                    for band in BANDS:
                        field = f'{window}_{band}'
                        vector = feature_vector(
                            word, field, args.electrodes, word_status[field]
                        )
                        if vector is not None:
                            rows_by_window_band[(window, band)].append(vector)

            for window in WINDOWS:
                for band in BANDS:
                    target = targets[band]
                    if target is None:
                        continue
                    rows = rows_by_window_band[(window, band)]
                    for rule, candidate in pooling_candidates(
                        rows, len(words)
                    ).items():
                        comparisons[(window, rule, band)].update(
                            target, candidate
                        )

            for band in BANDS:
                target = targets[band]
                if target is None:
                    continue
                pooled_rows = []
                for window in WINDOWS:
                    pooled_rows.extend(rows_by_window_band[(window, band)])
                for rule, candidate in pooling_candidates(
                    pooled_rows, len(words) * len(WINDOWS)
                ).items():
                    comparisons[('ALL_WINDOWS', rule, band)].update(
                        target, candidate
                    )

        per_file[filename] = {
            'bytes': int(os.path.getsize(path)),
            'sentence_records': int(len(sentences)),
            'sentences_with_word_structs': int(sentences_with_words),
            'word_records': int(file_words),
        }
        print(
            f'{filename}: {len(sentences)} sentences, '
            f'{file_words} word records'
        )

    reconstruction = {}
    ranking = []
    for (window, rule, band), accumulator in sorted(comparisons.items()):
        result = accumulator.result()
        if result is None:
            continue
        reconstruction.setdefault(window, {}).setdefault(rule, {})[band] = result
        ranking.append({
            'window': window,
            'pooling_rule': rule,
            'band': band,
            **result,
        })
    ranking.sort(key=lambda item: (
        item['relative_rmse'] is None,
        item['relative_rmse'] if item['relative_rmse'] is not None else np.inf,
    ))

    output = {
        'audit': 'v16 reproduction-scoped ZuCo EEG feature audit',
        'scope': {
            'included': (
                'precomputed sentence mean-band fields and repository-visible '
                'word/fixation-window band fields'
            ),
            'excluded': (
                'raw voltage, model training, split selection, and novel EEG '
                'feature engineering'
            ),
        },
        'provenance': {
            'git_commit': git_commit(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'eeg_dir': os.path.abspath(args.eeg_dir),
            'electrodes_per_band': args.electrodes,
            'band_order': list(BANDS),
            'fixation_windows_checked': list(WINDOWS),
        },
        'source': {
            'included_files': per_file,
            'excluded_files': excluded,
            'total_sentence_records': total_sentences,
            'total_word_records': total_words,
        },
        'sentence_field_inventory': finalize_inventory(sentence_inventory),
        'word_field_inventory': finalize_inventory(word_inventory),
        'sentence_band_feature_status': nested_status_result(sentence_status),
        'word_band_feature_status': nested_status_result(word_status),
        'reconstruction': {
            'comparison_target': (
                'within-participant sentenceData.mean_<band>, first 104 '
                'electrodes, before released cross-participant averaging and '
                'per-sentence Z-scoring'
            ),
            'important_limit': (
                'A low-error pooling rule is evidence about stored-feature '
                'construction; failure only means these declared simple rules '
                'did not reconstruct mean_<band>.'
            ),
            'by_window_rule_band': reconstruction,
            'ranked_best_first': ranking,
        },
        'ambiguities_preserved_for_report': [
            (
                'The paper says Z-score normalization over each frequency '
                'band but does not specify axes or train-fitted statistics.'
            ),
            (
                'The released sentence loader averages available participant '
                'recordings for a sentence before splitting; the paper does '
                'not clearly state that subject aggregation.'
            ),
            (
                'The released word and concatword training branches are '
                'placeholders, so stored word fields do not by themselves '
                'define the intended model input construction.'
            ),
        ],
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w') as output_file:
        json.dump(output, output_file, indent=2, allow_nan=False)

    print('\nAudit totals')
    print('  included files:', len(per_file))
    print('  sentence records:', total_sentences)
    print('  word records:', total_words)
    print('  sentence fields:', ', '.join(sorted(sentence_inventory)))
    print('  word fields:', ', '.join(sorted(word_inventory)))
    print('\nBest reconstruction comparisons by relative RMSE')
    for item in ranking[:12]:
        print(
            f"  {item['window']}/{item['pooling_rule']}/{item['band']}: "
            f"pairs={item['sentence_band_vector_pairs']}, "
            f"relative_RMSE={item['relative_rmse']:.6g}, "
            f"corr={item['flattened_pearson_correlation']}"
        )
    if not ranking:
        print('  No usable word-level band arrays were found.')
    print('\nSaved:', args.output_json)


if __name__ == '__main__':
    main()
