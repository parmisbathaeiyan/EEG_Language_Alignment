import os
import argparse
import platform
import random
import pickle
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.optim import Adam
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
from transformers import BertModel, BertTokenizer

from config import EEG_LEN, TEXT_LEN, d_model, d_inner, d_k, d_v, class_num, dropout
from optim_new import ScheduledOptim, early_stopping
from trainer import train
from evaluator import eval, inference
from model_new import MLP, Transformer
from utils import open_file
from new_plot import plot_learning_curve
from dataset_new import prepare_sr_eeg_data, EEGDataset, clean_dic, shuffle_split_data
torch.set_num_threads(2)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, help="Please choose a model from the following list: ['transformer', 'biLSTM', 'MLP', 'resnet']")
    parser.add_argument('--modality', type = str, default = None, help="Please choose a modality from the following list: ['eeg', 'text', fusion]")
    parser.add_argument('--dataset', type=str, help="Please choose a dataset from the following list: ['KEmoCon', 'ZuCo']")
    parser.add_argument('--task', default ='SA', type=str, help="If dataset == Zuco, please choose a task from the following list: ['SA', 'RD']")
    parser.add_argument('--level', type=str, default = 'sentence', help="If ZuCo, please choose the level of EEG feature you want to work with from this list: ['word', 'concatword', 'sentence']")
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--text_feature_len', type = int, default = 768)
    parser.add_argument('--eeg_feature_len', type = int, default = 832)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--optimizer_type', choices=['scheduled_adam', 'adam'],
                        default='scheduled_adam',
                        help='Use the released Transformer schedule or constant-LR Adam')
    parser.add_argument('--eps', type = float, default = 1e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.98)
    parser.add_argument('--weight_decay', type = float, default = 1e-2)
    parser.add_argument('--warm_steps', type = int, default = 2000)
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--device', type = str, default = 'cpu')
    parser.add_argument('--inference', type = int, default = 0)
    parser.add_argument('--checkpoint', type = str, default = None)
    parser.add_argument('--dev', type = int, default = 0)
    parser.add_argument('--loss', type = str, default = 'CE', help = "Please choose one of the following loss functions [CE, CCA, WD, CCAWD]")
    parser.add_argument('--num_layers', type = int, default = 1, help = 'Please choose how many layers the encoder should have')
    parser.add_argument('--num_heads', type = int, default = 1, help = 'Please choose how many heads the encoder should have')
    parser.add_argument('--dropout', type= float, default = 0.3, help = 'Please indicate the dropout proportion')
    parser.add_argument('--mlp_hidden_sizes', type=int, nargs=3, default=[256, 128, 64],
                        metavar=('H1', 'H2', 'H3'),
                        help='Hidden widths for the released three-hidden-layer MLP')
    parser.add_argument('--text_llm', type=str, default = 'bert', help = 'Please choose which LLM to encode text')
    parser.add_argument('--ce_weight', type = float, default = 1, help = 'Please choose the ce loss weight')
    parser.add_argument('--cca_weight', type = float, default = 1, help = 'Please choose the cca loss weight')
    parser.add_argument('--wd_weight', type = float, default = 1, help = 'Please choose the wd loss weight')
    parser.add_argument('--seed', type = int, default = 42, help = 'Global RNG seed for reproducible split/init')
    parser.add_argument('--patience', type = int, default = 20, help = 'Early-stopping patience (epochs)')
    parser.add_argument('--es_delta', type = float, default = 0.01, help = 'Early-stopping min-improvement delta')
    parser.add_argument('--oversample', type = int, default = 0, help = 'Balance classes per batch via WeightedRandomSampler (paper App C.3); 1 to enable (off by default = faithful upstream)')
    parser.add_argument('--eeg_cache', type = str, default = None, help = 'Path to cache the processed eeg_dict (skips slow .mat parsing on reruns)')
    # Logging infra (no effect on the learning procedure).
    parser.add_argument('--timestamp', type = str, default = None)
    parser.add_argument('--json_path', type = str, default = None)
    parser.add_argument('--plot_dst',  type = str, default = None)
    return parser.parse_args()


if __name__ == '__main__':
    
    
    args = get_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device(args.device)
    print(device)

    # Seed all RNGs before any randomness (the train/val/test split in
    # shuffle_split_data uses Python's random; model init/shuffling use torch).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = 'unknown'
    print(f'code commit: {git_commit}')
    
    if args.dataset == 'KEmoCon':
        ###### COMING SOON #####
        pass
    
    elif args.dataset == 'ZuCo':
        
        if args.task == 'RD':
            ###### COMING SOON #####
            pass
        
        elif args.task == 'SA':
            
            assert (args.level == 'sentence' or args.level == 'word' or args.level == 'concatword'), 'Please choose a correct eeg feature type'
            
            if args.level == 'word':
                ###### COMING SOON #####
                pass
            elif args.level == 'concatword':
                ###### COMING SOON #####
                pass      
            
            else:
                # Load csv
                sentiment_labels = pd.read_csv('data/sentiment_labels_clean.csv')
                
                sr_eeg_data_path = 'data/SR'
                
                sentence_list = sentiment_labels.sentence.tolist()
                labels_list = sentiment_labels.sentiment_label.tolist()
                sentence_ids_list = sentiment_labels.sentence_id.tolist()

                # Optional disk cache: parsing the 12 .mat files is slow (~minutes).
                # Cache the processed eeg_dict so reruns skip it. Disabled in dev mode
                # (partial load) and when --eeg_cache is not set. The cached dict is
                # built before the (seeded) split, so it does not affect the split.
                eeg_dict = None
                if args.eeg_cache and args.dev == 0 and os.path.exists(args.eeg_cache):
                    print(f'Loading cached eeg_dict from {args.eeg_cache}')
                    with open(args.eeg_cache, 'rb') as _cf:
                        eeg_dict = pickle.load(_cf)
                    print(f'  -> {len(eeg_dict)} sentences from cache')
                if eeg_dict is None:
                    eeg_dict = prepare_sr_eeg_data(sr_eeg_data_path, sentence_list, labels_list, sentence_ids_list, args)
                    if args.eeg_cache and args.dev == 0:
                        with open(args.eeg_cache, 'wb') as _cf:
                            pickle.dump(eeg_dict, _cf)
                        print(f'Cached eeg_dict -> {args.eeg_cache}')
                
                eeg_train_split, eeg_val_split, eeg_test_split = shuffle_split_data(eeg_dict)
                
                train_set, train_id_mapping = clean_dic(eeg_train_split)
                val_set, val_id_mapping = clean_dic(eeg_val_split)
                test_set, test_id_mapping = clean_dic(eeg_test_split)

                def _split_summary(items):
                    # clean_dic returns {integer_index: sample_dict}; iterating a
                    # dict directly yields the integer keys, not the samples.
                    records = items.values() if isinstance(items, dict) else items
                    labels = [int(item['label']) for item in records]
                    return {
                        'size': len(items),
                        'class_counts': np.bincount(labels, minlength=class_num).tolist(),
                    }

                split_metadata = {
                    'train': _split_summary(train_set),
                    'validation': _split_summary(val_set),
                    'test': _split_summary(test_set),
                }
                print(f'data splits: {split_metadata}')
                
                
                train_dataset = EEGDataset(train_set, args)
                val_dataset = EEGDataset(val_set, args)
                test_dataset = EEGDataset(test_set,args)

                # Drop the train loader's last batch only if it would be too small
                # for the CCA eigendecomposition / BatchNorm (needs > 16). Otherwise
                # keep every sample. e.g. 234 train: batch 32 -> 10 leftover (drop);
                # batch 64 -> 42 leftover (keep).
                MIN_LAST_BATCH = 17
                _rem = len(train_dataset) % args.batch_size
                _drop_last_train = (_rem != 0) and (_rem < MIN_LAST_BATCH)
                print(f'train={len(train_dataset)} batch={args.batch_size} '
                      f'last_batch={_rem if _rem else args.batch_size} '
                      f'drop_last_train={_drop_last_train}')

                if args.oversample:
                    # Paper App C.3: oversample so each batch is class-balanced.
                    # Weight each sample by 1/freq(its class); draw with replacement.
                    _labels = [int(train_set[i]['label']) for i in range(len(train_dataset))]
                    _class_count = np.bincount(_labels, minlength=class_num)
                    _class_w = 1.0 / np.maximum(_class_count, 1)
                    _sample_w = [float(_class_w[l]) for l in _labels]
                    _sampler = WeightedRandomSampler(_sample_w, num_samples=len(train_dataset), replacement=True)
                    print(f'oversampling ON: train class counts {_class_count.tolist()} -> balanced batches')
                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=args.batch_size,
                        sampler=_sampler,
                        drop_last = _drop_last_train
                    )
                else:
                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last = _drop_last_train
                    )
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last = False
                )
                
                if args.modality == 'eeg':
                    effective_input_features = {'eeg': EEG_LEN}
                elif args.modality == 'text':
                    effective_input_features = {'text': TEXT_LEN}
                else:
                    effective_input_features = {'eeg': EEG_LEN, 'text': TEXT_LEN}

                if args.model == 'transformer':
                    model = Transformer(device = device, d_feature_text = TEXT_LEN, d_feature_eeg = EEG_LEN,\
                                            d_model = d_model, d_inner = d_inner, n_layers = args.num_layers, \
                                            n_head=args.num_heads, d_k = d_k, d_v = d_v, dropout= dropout, \
                                            class_num = class_num, args = args)
                    effective_model_config = {
                        'class': 'Transformer',
                        'input_features': effective_input_features,
                        'd_model': d_model,
                        'd_inner': d_inner,
                        'num_layers': args.num_layers,
                        'num_heads': args.num_heads,
                        'dropout': dropout,
                        'num_classes': class_num,
                    }
                elif args.model == 'MLP':
                    layer2, layer3, layer4 = args.mlp_hidden_sizes
                    print(f'MLP hidden sizes: {layer2} -> {layer3} -> {layer4}; '
                          f'dropout={args.dropout}')
                    model = MLP(d_feature_text=TEXT_LEN, d_feature_eeg=EEG_LEN,
                                layer2=layer2, layer3=layer3, layer4=layer4,
                                class_num=class_num, dropout=args.dropout, args=args)
                    effective_model_config = {
                        'class': 'MLP',
                        'input_features': effective_input_features,
                        'hidden_sizes': [layer2, layer3, layer4],
                        'hidden_linear_layers': 3,
                        'dropout': args.dropout,
                        'num_classes': class_num,
                    }
                elif args.model == 'bert':
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    model = BertModel.from_pretrained("bert-base-uncased")
                    effective_model_config = {
                        'class': 'BertModel',
                        'pretrained_name': 'bert-base-uncased',
                    }
                else:
                    raise ValueError(f'Model {args.model!r} is not wired for this training path')
                    
                model = model.to(device)

                adam = Adam(filter(lambda x: x.requires_grad, model.parameters()),
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.eps, lr=args.lr,
                            weight_decay=args.weight_decay)
                if args.optimizer_type == 'scheduled_adam':
                    optimizer_metadata = {
                        'optimizer': 'Adam',
                        'betas': [args.adam_beta1, args.adam_beta2],
                        'eps': args.eps,
                        'weight_decay': args.weight_decay,
                        'schedule': 'Vaswani inverse-square-root with linear warmup',
                        'schedule_d_model': d_model,
                        'warmup_steps': args.warm_steps,
                        'note': ('ScheduledOptim overwrites the Adam constructor '
                                 'learning rate each step'),
                    }
                    optimizer = ScheduledOptim(
                        adam, d_model=d_model, n_warmup_steps=args.warm_steps
                    )
                else:
                    optimizer_metadata = {
                        'optimizer': 'Adam',
                        'betas': [args.adam_beta1, args.adam_beta2],
                        'eps': args.eps,
                        'weight_decay': args.weight_decay,
                        'schedule': None,
                        'constant_lr': args.lr,
                    }
                    optimizer = adam
                print(f'effective optimizer: {optimizer_metadata}')
                
                all_train_loss, all_train_acc, all_val_loss, all_val_acc = [], [], [], []
                all_pred_val, all_label_val = [], []
                eva_indices = []
                all_epochs  = []
                if args.inference == 1:
                    chkpt_path = os.path.join('baselines', args.checkpoint)
                    print(chkpt_path)
                    checkpoint = torch.load(chkpt_path, map_location = args.device)
                    model.load_state_dict(checkpoint['model'])
                    model = model.to(device)
                    inference(test_loader, device, model, test_dataset.__len__(), args)
                
                else:
                    for epoch in range(args.epochs):
                        
                        print('[ Epoch', epoch, ']')
                        start = time.time()
                        
                        train_loss, train_acc, train_cm, train_preds, train_labels = train(train_loader, device, model, optimizer, train_dataset.__len__(), args)
                        val_loss, val_acc, val_cm, val_preds, val_labels = eval(val_loader, device, model, val_dataset.__len__(), args)
                        
                        model_state_dict = model.state_dict()
                        
                        checkpoint = {
                            'model' : model_state_dict,
                            'config_file' : 'config',
                            'epoch' : epoch
                        }
                    
                        all_pred_val.extend(val_preds)
                        all_label_val.extend(val_labels)
                        all_train_loss.append(train_loss)
                        all_train_acc.append(train_acc)
                        all_val_loss.append(val_loss)
                        all_val_acc.append(val_acc)
                        all_epochs.append(epoch)
                        
                        
                        if val_loss <= min(all_val_loss):
                                torch.save(checkpoint, f'baselines/{args.model}_{args.modality}_{args.level}_{args.num_layers}_{args.num_heads}_{args.batch_size}_{args.loss}_{args.ce_weight}_{args.cca_weight}_{args.wd_weight}.chkpt')
                                print('    - [Info] The checkpoint file has been updated.')
                            
                        early_stop = early_stopping(all_val_loss, patience = args.patience, delta = args.es_delta)
                        
                        if early_stop:
                            print('Validation loss has stopped decreasing. Early stopping...')
                            break   
                    
                    plot_learning_curve(all_train_acc, all_train_loss, all_val_acc, all_val_loss, all_epochs, args)
                    if args.plot_dst:
                        import shutil as _shutil
                        _plot_src = f'lr_curves/learning_curve_{args.model}_{args.modality}_{args.level}_{args.num_layers}_{args.num_heads}_{args.batch_size}_{args.loss}.png'
                        if os.path.exists(_plot_src):
                            _shutil.copy(_plot_src, args.plot_dst)
                    checkpoint = torch.load(f'baselines/{args.model}_{args.modality}_{args.level}_{args.num_layers}_{args.num_heads}_{args.batch_size}_{args.loss}_{args.ce_weight}_{args.cca_weight}_{args.wd_weight}.chkpt', map_location = args.device)
                    model.load_state_dict(checkpoint['model'])
                    model = model.to(device)
                    _test_metrics = inference(test_loader, device, model, test_dataset.__len__(), args)

                    if args.json_path:
                        _results = {
                            'run_name'        : os.path.basename(args.json_path),
                            'timestamp'       : args.timestamp,
                            'code_commit'     : git_commit,
                            'hyperparameters' : vars(args),
                            'effective_model' : effective_model_config,
                            'effective_optimizer': optimizer_metadata,
                            'data_splits'     : split_metadata,
                            'runtime'         : {
                                'python': platform.python_version(),
                                'torch': torch.__version__,
                                'transformers': transformers.__version__,
                                'cuda_device': (torch.cuda.get_device_name(0)
                                                if torch.cuda.is_available() else None),
                            },
                            'total_epochs_run': len(all_epochs),
                            'best_val_loss'   : float(min(all_val_loss)),
                            'best_val_acc'    : float(max(all_val_acc)),
                            'best_val_epoch'  : int(all_epochs[all_val_loss.index(min(all_val_loss))]),
                            'test'            : _test_metrics,
                            'per_epoch'       : [
                                {'epoch': int(all_epochs[i]),
                                 'train_loss': float(all_train_loss[i]),
                                 'train_acc' : float(all_train_acc[i]),
                                 'val_loss'  : float(all_val_loss[i]),
                                 'val_acc'   : float(all_val_acc[i])}
                                for i in range(len(all_epochs))
                            ],
                        }
                        with open(args.json_path, 'w') as _jf:
                            json.dump(_results, _jf, indent=2)
                        print(f'JSON saved -> {args.json_path}')
                                        
                    
                    
                    
            
                
