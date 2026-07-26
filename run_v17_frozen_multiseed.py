"""Run and summarize the frozen v17 sentence-level MLP-EEG baseline.

The scientific grid is intentionally fixed: five canonical split seeds
(0--4) crossed with three model/training seeds (0--2). The sampler seed is
explicit and equals the model seed. Completed run JSONs are validated and
reused, so an interrupted Colab session can safely resume without selecting
or repeating runs based on their results.
"""

import argparse
import datetime as dt
import json
import math
import os
import platform
import statistics
import subprocess


SPLIT_SEEDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (0, 1, 2)
PAPER_TARGET = {'accuracy': 0.499, 'f1_macro': 0.480}
PRIMARY = 'primary_minimum_validation_loss'
SECONDARY = 'secondary_maximum_validation_accuracy'
CHECKPOINTS = (PRIMARY, SECONDARY)
CLASS_NAMES = ('neutral', 'positive', 'negative')
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

FROZEN_CONFIG = {
    'dataset': 'ZuCo',
    'task': 'SA',
    'level': 'sentence',
    'modality': 'eeg',
    'model': 'MLP',
    'loss': 'CE',
    'input_width': 832,
    'mlp_implementation': 'released',
    'hidden_sizes': [256, 128, 64],
    'linear_bias': False,
    'dropout': 0.3,
    'batch_size': 32,
    'epochs_maximum': 200,
    'early_stopping_patience': 20,
    'early_stopping_delta': 0.01,
    'optimizer_type': 'scheduled_adam',
    'adam_constructor_lr': 1e-5,
    'adam_betas': [0.9, 0.98],
    'adam_epsilon': 1e-4,
    'weight_decay': 1e-2,
    'warmup_steps': 2000,
    'oversampling': (
        'inverse-training-class-frequency WeightedRandomSampler with '
        'replacement; balanced in expectation'
    ),
    'primary_checkpoint': 'minimum validation loss',
    'secondary_checkpoint': (
        'maximum validation accuracy, ties broken by lower validation loss'
    ),
    'paper_target': PAPER_TARGET,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--eeg_cache', default='/content/eeg_dict_cache.pkl')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO_DIR
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json_write(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w') as output_file:
        json.dump(value, output_file, indent=2, allow_nan=False)
    os.replace(temporary, path)


def load_json(path):
    with open(path) as input_file:
        return json.load(input_file)


def run_name(split_seed, model_seed):
    return (
        f'v17_MLP_EEG_frozen_split{split_seed}_'
        f'model{model_seed}_sampler{model_seed}'
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close_number(observed, expected, tolerance=1e-12):
    return math.isclose(
        float(observed), float(expected),
        rel_tol=tolerance, abs_tol=tolerance,
    )


def validate_run(result, split_seed, model_seed, commit):
    prefix = f'split={split_seed}, model={model_seed}: '
    require(result.get('code_commit') == commit, prefix + 'commit mismatch')

    hp = result.get('hyperparameters', {})
    expected_exact = {
        'dataset': 'ZuCo', 'task': 'SA', 'level': 'sentence',
        'modality': 'eeg', 'model': 'MLP', 'loss': 'CE',
        'batch_size': 32, 'epochs': 200, 'warm_steps': 2000,
        'mlp_hidden_sizes': [256, 128, 64],
        'mlp_implementation': 'released', 'mlp_bias': 0,
        'optimizer_type': 'scheduled_adam', 'patience': 20,
        'oversample': 1, 'sanity_overfit_per_class': 0,
        'split_seed': split_seed, 'model_seed': model_seed,
        'sampler_seed': model_seed,
    }
    for key, expected in expected_exact.items():
        require(hp.get(key) == expected, prefix + f'{key} mismatch')
    expected_numbers = {
        'dropout': 0.3, 'lr': 1e-5, 'weight_decay': 1e-2,
        'eps': 1e-4, 'adam_beta1': 0.9, 'adam_beta2': 0.98,
        'es_delta': 0.01,
    }
    for key, expected in expected_numbers.items():
        require(
            key in hp and close_number(hp[key], expected),
            prefix + f'{key} mismatch',
        )

    seeds = result.get('effective_random_seeds', {})
    require(seeds.get('split_seed') == split_seed,
            prefix + 'effective split seed mismatch')
    require(seeds.get('model_seed') == model_seed,
            prefix + 'effective model seed mismatch')
    require(seeds.get('sampler_seed') == model_seed,
            prefix + 'effective sampler seed mismatch')

    model = result.get('effective_model', {})
    require(model.get('class') == 'MLP', prefix + 'model class mismatch')
    require(model.get('hidden_sizes') == [256, 128, 64],
            prefix + 'hidden sizes mismatch')
    require(model.get('linear_bias') is False, prefix + 'bias mismatch')
    require(model.get('total_linear_layers_including_output') == 4,
            prefix + 'linear layer count mismatch')

    optimizer = result.get('effective_optimizer', {})
    require(
        optimizer.get('schedule') ==
        'Vaswani inverse-square-root with linear warmup',
        prefix + 'optimizer schedule mismatch',
    )
    require(optimizer.get('warmup_steps') == 2000,
            prefix + 'warmup mismatch')

    sampling = result.get('training_sampling', {})
    require(sampling.get('method') == 'WeightedRandomSampler',
            prefix + 'sampler mismatch')
    require(sampling.get('sampler_seed') == model_seed,
            prefix + 'saved sampler seed mismatch')

    representation = result.get('effective_input_representation', {})
    require(representation.get('concatenated_feature_width') == 832,
            prefix + 'input width mismatch')
    require(representation.get('used_values_per_band') == 104,
            prefix + 'electrode count mismatch')

    splits = result.get('data_splits', {})
    require(splits.get('diagnostic') is None,
            prefix + 'diagnostic mode must be off')
    require(
        splits.get('original_before_diagnostic', {}).get('seed') ==
        split_seed,
        prefix + 'split manifest seed mismatch',
    )
    split_ids = splits.get('original_before_diagnostic', {}).get(
        'train', {}
    ).get('sentence_ids', [])
    validation_ids = splits.get('original_before_diagnostic', {}).get(
        'validation', {}
    ).get('sentence_ids', [])
    test_ids = splits.get('original_before_diagnostic', {}).get(
        'test', {}
    ).get('sentence_ids', [])
    require(len(split_ids) + len(validation_ids) + len(test_ids) == 400,
            prefix + 'split sizes do not sum to 400')
    require(not (set(split_ids) & set(validation_ids)),
            prefix + 'train/validation overlap')
    require(not (set(split_ids) & set(test_ids)),
            prefix + 'train/test overlap')
    require(not (set(validation_ids) & set(test_ids)),
            prefix + 'validation/test overlap')

    evaluations = result.get('checkpoint_evaluations', {})
    for checkpoint in CHECKPOINTS:
        require(checkpoint in evaluations,
                prefix + f'missing {checkpoint}')
        for split in ('validation', 'test'):
            metrics = evaluations[checkpoint].get(split, {})
            require('accuracy' in metrics and 'f1_macro' in metrics,
                    prefix + f'incomplete {checkpoint} {split} metrics')
            cm = metrics.get('confusion_matrix')
            require(
                isinstance(cm, list) and len(cm) == 3
                and all(isinstance(row, list) and len(row) == 3 for row in cm),
                prefix + f'invalid {checkpoint} {split} confusion matrix',
            )
    return result


def prepare_design(output_dir, commit):
    design = {
        'experiment': 'v17 frozen canonical sentence-level MLP-EEG',
        'code_commit': commit,
        'split_seeds': list(SPLIT_SEEDS),
        'model_seeds': list(MODEL_SEEDS),
        'sampler_seed_rule': 'sampler_seed equals model_seed',
        'crossed_runs': len(SPLIT_SEEDS) * len(MODEL_SEEDS),
        'configuration': FROZEN_CONFIG,
        'selection_policy': (
            'Report the full distribution. Minimum-validation-loss is the '
            'primary checkpoint; maximum-validation-accuracy is a secondary '
            'diagnostic. Never select a run or checkpoint using test results.'
        ),
        'resume_policy': (
            'Reuse only complete run JSONs that pass commit, seed, model, '
            'optimizer, representation, split and checkpoint validation.'
        ),
    }
    path = os.path.join(output_dir, 'v17_design.json')
    if os.path.exists(path):
        observed = load_json(path)
        require(observed == design,
                'Existing v17_design.json differs from this frozen design')
    else:
        atomic_json_write(path, design)
    return design


def command_for_run(
    split_seed, model_seed, eeg_cache, device,
    timestamp, json_path, plot_path,
):
    return [
        'python', '-u', 'main_new.py',
        '--dataset', 'ZuCo', '--task', 'SA',
        '--level', 'sentence', '--modality', 'eeg',
        '--model', 'MLP', '--loss', 'CE',
        '--batch_size', '32', '--epochs', '200',
        '--num_layers', '1', '--num_heads', '1',
        '--dropout', '0.3', '--warm_steps', '2000',
        '--mlp_hidden_sizes', '256', '128', '64',
        '--mlp_implementation', 'released', '--mlp_bias', '0',
        '--optimizer_type', 'scheduled_adam', '--lr', '1e-5',
        '--weight_decay', '0.01', '--eps', '0.0001',
        '--adam_beta1', '0.9', '--adam_beta2', '0.98',
        '--patience', '20', '--es_delta', '0.01',
        '--oversample', '1', '--sanity_overfit_per_class', '0',
        '--seed', '42',
        '--split_seed', str(split_seed),
        '--model_seed', str(model_seed),
        '--sampler_seed', str(model_seed),
        '--eeg_cache', eeg_cache,
        '--inference', '0', '--dev', '0', '--device', device,
        '--timestamp', timestamp,
        '--json_path', json_path, '--plot_dst', plot_path,
    ]


def run_one(
    output_dir, eeg_cache, device, split_seed, model_seed, commit,
):
    name = run_name(split_seed, model_seed)
    json_path = os.path.join(output_dir, name + '.json')
    plot_path = os.path.join(output_dir, 'plots', name + '.png')
    if os.path.exists(json_path):
        try:
            result = validate_run(
                load_json(json_path), split_seed, model_seed, commit
            )
        except json.JSONDecodeError:
            quarantine = json_path + '.incomplete_' + dt.datetime.now().strftime(
                '%Y%m%d%H%M%S'
            )
            os.replace(json_path, quarantine)
            print(f'Preserved incomplete JSON as {quarantine}')
        else:
            print(f'SKIP complete validated run: {name}')
            return result

    timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    log_path = os.path.join(
        output_dir, 'logs', f'{name}_{timestamp}.txt'
    )
    command = command_for_run(
        split_seed, model_seed, eeg_cache, device,
        timestamp, json_path, plot_path,
    )
    print('\n' + '=' * 72)
    print(f'RUN {name}')
    print(f'JSON: {json_path}')
    print(f'LOG:  {log_path}')
    print('=' * 72)
    environment = {
        **os.environ,
        'TQDM_DISABLE': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    }
    with open(log_path, 'w') as log_file:
        log_file.write('Command: ' + ' '.join(command) + '\n')
        log_file.write('-' * 72 + '\n')
        process = subprocess.Popen(
            command, cwd=REPO_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=environment,
        )
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f'{name} failed with exit code {return_code}; see {log_path}. '
            'After fixing the objective failure, rerun the driver to resume.'
        )
    require(os.path.exists(json_path), f'{name} did not write its JSON')
    return validate_run(
        load_json(json_path), split_seed, model_seed, commit
    )


def describe(values):
    values = [float(value) for value in values]
    require(values, 'Cannot summarize an empty metric collection')
    return {
        'count': len(values),
        'mean': statistics.fmean(values),
        'population_standard_deviation': statistics.pstdev(values),
        'minimum': min(values),
        'median': statistics.median(values),
        'maximum': max(values),
    }


def metric_value(result, checkpoint, split, metric):
    return float(
        result['checkpoint_evaluations'][checkpoint][split][metric]
    )


def two_way_variance(grid, checkpoint, split, metric):
    values = {
        key: metric_value(result, checkpoint, split, metric)
        for key, result in grid.items()
    }
    grand = statistics.fmean(values.values())
    split_means = {
        split_seed: statistics.fmean(
            values[(split_seed, model_seed)]
            for model_seed in MODEL_SEEDS
        )
        for split_seed in SPLIT_SEEDS
    }
    model_means = {
        model_seed: statistics.fmean(
            values[(split_seed, model_seed)]
            for split_seed in SPLIT_SEEDS
        )
        for model_seed in MODEL_SEEDS
    }
    split_variance = statistics.fmean(
        (value - grand) ** 2 for value in split_means.values()
    )
    model_variance = statistics.fmean(
        (value - grand) ** 2 for value in model_means.values()
    )
    interaction_variance = statistics.fmean(
        (
            value - split_means[split_seed]
            - model_means[model_seed] + grand
        ) ** 2
        for (split_seed, model_seed), value in values.items()
    )
    total_variance = statistics.fmean(
        (value - grand) ** 2 for value in values.values()
    )
    components = {
        'split_seed_main_effect': split_variance,
        'model_training_seed_main_effect': model_variance,
        'split_by_model_interaction_and_residual': interaction_variance,
    }
    return {
        'method': (
            'balanced two-way population variance decomposition of the '
            '5x3 crossed grid'
        ),
        'total_population_variance': total_variance,
        'components': components,
        'fractions_of_total': {
            key: (value / total_variance if total_variance else None)
            for key, value in components.items()
        },
        'split_seed_means': {
            str(key): value for key, value in split_means.items()
        },
        'model_training_seed_means': {
            str(key): value for key, value in model_means.items()
        },
    }


def sum_confusion_matrices(grid, checkpoint, split):
    total = [[0, 0, 0] for _ in range(3)]
    for result in grid.values():
        matrix = result['checkpoint_evaluations'][checkpoint][split][
            'confusion_matrix'
        ]
        for row in range(3):
            for column in range(3):
                total[row][column] += int(matrix[row][column])
    return {
        'class_names': list(CLASS_NAMES),
        'layout': 'rows=true, cols=predicted',
        'summed_over_15_runs': total,
        'true_class_counts': [sum(row) for row in total],
        'predicted_class_counts': [
            sum(total[row][column] for row in range(3))
            for column in range(3)
        ],
    }


def checkpoint_summary(grid, checkpoint):
    output = {
        'selection_role': (
            'primary predeclared result'
            if checkpoint == PRIMARY else 'secondary diagnostic only'
        ),
        'overall': {},
        'by_split_seed': {},
        'by_model_training_seed': {},
        'variance_decomposition': {},
        'summed_confusion_matrices': {},
    }
    split_metrics = {
        'validation': ('accuracy', 'f1_macro', 'loss'),
        'test': (
            'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'
        ),
    }
    for split, metrics in split_metrics.items():
        output['overall'][split] = {}
        output['by_split_seed'][split] = {}
        output['by_model_training_seed'][split] = {}
        for metric in metrics:
            output['overall'][split][metric] = describe([
                metric_value(result, checkpoint, split, metric)
                for result in grid.values()
            ])
            output['by_split_seed'][split][metric] = {
                str(split_seed): describe([
                    metric_value(
                        grid[(split_seed, model_seed)],
                        checkpoint, split, metric,
                    )
                    for model_seed in MODEL_SEEDS
                ])
                for split_seed in SPLIT_SEEDS
            }
            output['by_model_training_seed'][split][metric] = {
                str(model_seed): describe([
                    metric_value(
                        grid[(split_seed, model_seed)],
                        checkpoint, split, metric,
                    )
                    for split_seed in SPLIT_SEEDS
                ])
                for model_seed in MODEL_SEEDS
            }
            if metric in ('accuracy', 'f1_macro'):
                output['variance_decomposition'][f'{split}_{metric}'] = (
                    two_way_variance(
                        grid, checkpoint, split, metric
                    )
                )
        output['summed_confusion_matrices'][split] = (
            sum_confusion_matrices(grid, checkpoint, split)
        )

    test = output['overall']['test']
    output['paper_target_comparison'] = {
        metric: {
            'paper': PAPER_TARGET[metric],
            'mean_minus_paper': test[metric]['mean'] - PAPER_TARGET[metric],
            'runs_meeting_or_exceeding_paper': sum(
                metric_value(result, checkpoint, 'test', metric)
                >= PAPER_TARGET[metric]
                for result in grid.values()
            ),
            'total_runs': len(grid),
        }
        for metric in ('accuracy', 'f1_macro')
    }
    output['runs_meeting_both_paper_targets'] = sum(
        metric_value(result, checkpoint, 'test', 'accuracy')
        >= PAPER_TARGET['accuracy']
        and metric_value(result, checkpoint, 'test', 'f1_macro')
        >= PAPER_TARGET['f1_macro']
        for result in grid.values()
    )
    return output


def majority_baselines(grid):
    output = {}
    for split_seed in SPLIT_SEEDS:
        result = grid[(split_seed, MODEL_SEEDS[0])]
        counts = result['data_splits']['test']['class_counts']
        total = sum(counts)
        majority = max(counts)
        precision = majority / total
        majority_f1 = 2 * precision / (precision + 1)
        output[str(split_seed)] = {
            'test_class_counts': counts,
            'always_majority_accuracy': precision,
            'always_majority_macro_f1': majority_f1 / len(counts),
        }
    return {
        'per_split': output,
        'accuracy_across_splits': describe([
            item['always_majority_accuracy'] for item in output.values()
        ]),
        'macro_f1_across_splits': describe([
            item['always_majority_macro_f1'] for item in output.values()
        ]),
        'nominal_three_class_chance_accuracy': 1 / 3,
    }


def build_summary(grid, design, commit):
    require(len(grid) == 15, 'v17 grid is incomplete')
    for split_seed in SPLIT_SEEDS:
        manifests = []
        for model_seed in MODEL_SEEDS:
            manifest = grid[(split_seed, model_seed)][
                'data_splits'
            ]['original_before_diagnostic']
            manifests.append(manifest)
        require(all(manifest == manifests[0] for manifest in manifests[1:]),
                f'split {split_seed} IDs differ across model seeds')

    per_run = []
    for split_seed in SPLIT_SEEDS:
        for model_seed in MODEL_SEEDS:
            result = grid[(split_seed, model_seed)]
            entry = {
                'split_seed': split_seed,
                'model_seed': model_seed,
                'sampler_seed': model_seed,
                'run_json': run_name(split_seed, model_seed) + '.json',
                'epochs_run': result['total_epochs_run'],
                'checkpoints': {},
            }
            for checkpoint in CHECKPOINTS:
                evaluation = result['checkpoint_evaluations'][checkpoint]
                entry['checkpoints'][checkpoint] = {
                    'checkpoint_epoch': evaluation['validation'][
                        'checkpoint_epoch'
                    ],
                    'validation_accuracy': evaluation['validation'][
                        'accuracy'
                    ],
                    'validation_f1_macro': evaluation['validation'][
                        'f1_macro'
                    ],
                    'validation_loss': evaluation['validation']['loss'],
                    'test_accuracy': evaluation['test']['accuracy'],
                    'test_f1_macro': evaluation['test']['f1_macro'],
                    'test_confusion_matrix': evaluation['test'][
                        'confusion_matrix'
                    ],
                }
            per_run.append(entry)

    paired = {}
    for metric in ('accuracy', 'f1_macro'):
        paired[metric] = describe([
            metric_value(result, SECONDARY, 'test', metric)
            - metric_value(result, PRIMARY, 'test', metric)
            for result in grid.values()
        ])

    return {
        'experiment': design['experiment'],
        'status': 'complete',
        'created_utc': utc_now(),
        'code_commit': commit,
        'runtime': {
            'summary_python': platform.python_version(),
        },
        'design': design,
        'completed_runs': len(grid),
        'paper_target': PAPER_TARGET,
        'baselines': majority_baselines(grid),
        'checkpoint_summaries': {
            checkpoint: checkpoint_summary(grid, checkpoint)
            for checkpoint in CHECKPOINTS
        },
        'secondary_minus_primary_paired_test_difference': paired,
        'per_run': per_run,
        'interpretation_guardrails': [
            'The full 15-run distribution is the result; do not report only '
            'the best seed.',
            'Minimum-validation-loss is primary. Maximum-validation-accuracy '
            'is secondary and must not replace it based on test performance.',
            'The split and model/training variance decomposition describes '
            'this fixed 5x3 seed grid, not a population of all possible seeds.',
        ],
    }


def print_summary(summary, summary_path):
    print('\n' + '=' * 72)
    print('V17 FROZEN 5x3 MLP GRID COMPLETE')
    print('=' * 72)
    for checkpoint in CHECKPOINTS:
        role = 'PRIMARY min-val-loss' if checkpoint == PRIMARY else (
            'SECONDARY max-val-accuracy'
        )
        result = summary['checkpoint_summaries'][checkpoint]['overall'][
            'test'
        ]
        accuracy = result['accuracy']
        f1 = result['f1_macro']
        print(
            f'{role}:\n'
            f'  test accuracy mean={accuracy["mean"]:.4f} '
            f'+/- {accuracy["population_standard_deviation"]:.4f} '
            f'range=[{accuracy["minimum"]:.4f}, '
            f'{accuracy["maximum"]:.4f}]\n'
            f'  test macro F1 mean={f1["mean"]:.4f} '
            f'+/- {f1["population_standard_deviation"]:.4f} '
            f'range=[{f1["minimum"]:.4f}, {f1["maximum"]:.4f}]'
        )
    print(
        f'Paper target: accuracy={PAPER_TARGET["accuracy"]:.3f}, '
        f'macro F1={PAPER_TARGET["f1_macro"]:.3f}'
    )
    print('Summary JSON:', summary_path)


def main():
    args = get_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    commit = git_commit()
    design = prepare_design(output_dir, commit)
    print('Frozen design:', json.dumps(design, indent=2))

    grid = {}
    total = len(SPLIT_SEEDS) * len(MODEL_SEEDS)
    completed = 0
    for split_seed in SPLIT_SEEDS:
        for model_seed in MODEL_SEEDS:
            completed += 1
            print(
                f'\nV17 grid position {completed}/{total}: '
                f'split={split_seed}, model/sampler={model_seed}'
            )
            grid[(split_seed, model_seed)] = run_one(
                output_dir, args.eeg_cache, args.device,
                split_seed, model_seed, commit,
            )

    summary = build_summary(grid, design, commit)
    summary_path = os.path.join(output_dir, 'v17_summary.json')
    atomic_json_write(summary_path, summary)
    print_summary(summary, summary_path)


if __name__ == '__main__':
    main()
