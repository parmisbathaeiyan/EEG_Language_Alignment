"""Canonical sentence-level train/validation/test splitting utilities.

The released code shuffled dictionary insertion order. That makes a nominal
seed depend on filesystem traversal and cache construction order. These helpers
sort sentence IDs before using a local RNG, so the same IDs and labels produce
the same split on every runtime.
"""

import json
import numbers
import random
from collections import defaultdict


SPLIT_NAMES = ('train', 'validation', 'test')


def json_scalar(value):
    """Convert NumPy-like scalar values without importing NumPy."""
    return value.item() if hasattr(value, 'item') else value


def stable_id_key(value):
    """Return an order key that is stable across mixed scalar ID types."""
    value = json_scalar(value)
    if isinstance(value, bool):
        return ('boolean', int(value))
    if isinstance(value, numbers.Real):
        return ('number', float(value))
    if isinstance(value, str):
        return ('string', value)
    return (
        'other',
        type(value).__name__,
        json.dumps(value, sort_keys=True, ensure_ascii=True, default=str),
    )


def canonical_stratified_sentence_split(
    sentence_labels,
    seed,
    train_fraction=0.8,
    validation_fraction=0.1,
):
    """Split sentence IDs after canonical sorting and class-wise shuffling.

    A single local RNG is used while labels are traversed in canonical order.
    Train and validation counts use the released code's integer truncation;
    test receives the remainder.
    """
    if train_fraction < 0 or validation_fraction < 0:
        raise ValueError('Split fractions must be non-negative')
    if train_fraction + validation_fraction > 1:
        raise ValueError('Train + validation fractions cannot exceed 1')

    grouped = defaultdict(list)
    for sentence_id, label in sentence_labels.items():
        grouped[json_scalar(label)].append(sentence_id)

    rng = random.Random(int(seed))
    split_ids = {name: [] for name in SPLIT_NAMES}
    assignments = {}
    class_counts = {}

    for label in sorted(grouped, key=stable_id_key):
        ids = sorted(grouped[label], key=stable_id_key)
        rng.shuffle(ids)
        train_count = int(train_fraction * len(ids))
        validation_count = int(validation_fraction * len(ids))
        class_counts[json_scalar(label)] = {
            'total': len(ids),
            'train': train_count,
            'validation': validation_count,
            'test': len(ids) - train_count - validation_count,
        }

        for index, sentence_id in enumerate(ids):
            if index < train_count:
                split_name = 'train'
            elif index < train_count + validation_count:
                split_name = 'validation'
            else:
                split_name = 'test'
            assignments[sentence_id] = split_name
            split_ids[split_name].append(sentence_id)

    for split_name in SPLIT_NAMES:
        split_ids[split_name].sort(key=stable_id_key)

    return {
        'seed': int(seed),
        'algorithm': (
            'sort sentence IDs and labels canonically, then class-wise '
            'shuffle with one local Python random.Random(seed); integer-truncate '
            '80% train and 10% validation, test gets the remainder'
        ),
        'assignments': assignments,
        'split_ids': split_ids,
        'class_counts': class_counts,
    }


def split_eeg_dict(eeg_dict, seed):
    """Apply the canonical split to a released-style EEG dictionary."""
    sentence_labels = {
        sentence_id: item['label']
        for sentence_id, item in eeg_dict.items()
    }
    manifest = canonical_stratified_sentence_split(
        sentence_labels, seed
    )
    split_dicts = {name: {} for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for sentence_id in manifest['split_ids'][split_name]:
            split_dicts[split_name][sentence_id] = eeg_dict[sentence_id]
    return (
        split_dicts['train'],
        split_dicts['validation'],
        split_dicts['test'],
        manifest,
    )
