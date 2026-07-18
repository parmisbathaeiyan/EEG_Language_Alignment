import torch
from loss import cal_loss
from metrics import cal_statistic
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np

# ZuCo SA label index -> name. Loader keeps 0 and 1, remaps original -1 -> 2:
#   0 <- 0 (neutral), 1 <- 1 (positive), 2 <- -1 (negative)
SA_CLASS_NAMES = ['neutral', 'positive', 'negative']


def classification_metrics(cm, class_names=None):
    """Return serializable metrics for a confusion matrix."""
    acc, precision, recall, f1 = cal_statistic(cm)
    names = (
        class_names
        if class_names and len(class_names) == len(precision)
        else [str(i) for i in range(len(precision))]
    )
    return {
        'accuracy': float(acc),
        'precision_macro': float(np.mean(precision)),
        'recall_macro': float(np.mean(recall)),
        'f1_macro': float(np.mean(f1)),
        'precision_per_class': [float(x) for x in precision],
        'recall_per_class': [float(x) for x in recall],
        'f1_per_class': [float(x) for x in f1],
        'class_names': names,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_layout': 'rows=true, cols=pred',
        'predicted_class_counts': cm.sum(axis=0).astype(int).tolist(),
        'true_class_counts': cm.sum(axis=1).astype(int).tolist(),
    }


def format_confusion_matrix(cm, class_names=None):
    """Pretty-print a confusion matrix with rows = TRUE, cols = PREDICTED."""
    n = cm.shape[0]
    if not class_names or len(class_names) != n:
        class_names = [str(i) for i in range(n)]
    cols = ['pred:' + c for c in class_names]
    w = max(13, max(len(c) for c in cols) + 2)
    lines = ['Confusion matrix (rows = TRUE, cols = PREDICTED):',
             ' ' * w + ''.join(c.rjust(w) for c in cols)]
    for i in range(n):
        lines.append(('true:' + class_names[i]).rjust(w)
                     + ''.join(str(int(cm[i][j])).rjust(w) for j in range(n)))
    return '\n'.join(lines)


def eval(valid_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=100, desc='- (Validation)  ', leave=False):
            
            if args.modality == 'text':
                text, label = batch['sentence'].to(device), batch['label'].to(device)
            elif args.modality == 'eeg':
                eeg, label = batch['seq'].to(device), batch['label'].to(device)
            else:
                text, eeg, label = batch['sentence'].to(device), batch['seq'].to(device), batch['label'].to(device)
            
            if args.modality == 'text':
                pred_text = model(text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_text.max(1)[1].cpu().numpy())
                all_pred.extend(pred_text.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_text)

                total_loss += loss.item()
                total_correct += n_correct

            elif args.modality == 'eeg':
                pred_eeg = model(eeg_src_seq = eeg)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_eeg.max(1)[1].cpu().numpy())
                all_pred.extend(pred_eeg.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_eeg)

                total_loss += loss.item()
                total_correct += n_correct
            elif args.modality == 'fusion' and args.model == 'transformer':
                pred, eeg_embed, text_embed = model(eeg_src_seq = eeg, text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred, text_embed = text_embed, eeg_embed = eeg_embed)
                all_pred.extend(pred.cpu().detach().numpy())

                total_loss += loss.item()
                total_correct += n_correct
                
            elif args.modality == 'fusion' and args.model == 'MLP':
                pred = model(eeg_src_seq = eeg, text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred)
                all_pred.extend(pred.cpu().detach().numpy())

                total_loss += loss.item()
                total_correct += n_correct      
                
    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=calculate_average(pre_i)))
    print('rec_i is : {rec_i}'.format(rec_i=calculate_average(rec_i)))
    print('F1_i is : {F1_i}'.format(F1_i=calculate_average(F1_i)))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    print(f'Validation Loss: {valid_loss}')
    print(f'Validation Accuracy: {valid_acc}')
    return valid_loss, valid_acc, cm, all_pred, all_labels


def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Return 0 if the list is empty to avoid division by zero error
    
    total = sum(numbers)
    average = total / len(numbers)
    return average

def inference(test_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

            if args.modality == 'text':
                text, label = batch['sentence'].to(device), batch['label'].to(device)
            elif args.modality == 'eeg':
                eeg, label = batch['seq'].to(device), batch['label'].to(device)
            else:
                text, eeg, label = batch['sentence'].to(device), batch['seq'].to(device), batch['label'].to(device)
            
            if args.modality == 'text':
                pred_text = model(text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_text.max(1)[1].cpu().numpy())
                all_pred.extend(pred_text.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_text)

                total_loss += loss.item()
                total_correct += n_correct

            elif args.modality == 'eeg':
                pred_eeg = model(eeg_src_seq = eeg)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_eeg.max(1)[1].cpu().numpy())
                all_pred.extend(pred_eeg.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_eeg)

                total_loss += loss.item()
                total_correct += n_correct
            elif args.modality == 'fusion' and args.model == 'transformer':
                pred, eeg_embed, text_embed = model(eeg_src_seq = eeg, text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred, eeg_embed = eeg_embed, text_embed = text_embed)
                all_pred.extend(pred.cpu().detach().numpy())

                total_loss += loss.item()
                total_correct += n_correct
                
            elif args.modality == 'fusion' and args.model == 'MLP':
                pred = model(eeg_src_seq = eeg, text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred)
                all_pred.extend(pred.cpu().detach().numpy())

                total_loss += loss.item()
                total_correct += n_correct      


    np.savetxt(f'pred_labels/{args.model}_{args.modality}_{args.level}_{args.num_layers}_{args.num_heads}_{args.batch_size}_all_pred.txt',all_pred)
    np.savetxt(f'pred_labels/{args.model}_{args.modality}_{args.level}_{args.num_layers}_{args.num_heads}_{args.batch_size}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    metrics = classification_metrics(cm, SA_CLASS_NAMES)
    test_acc = total_correct / total_num

    print('\n===== TEST RESULTS =====')
    print(format_confusion_matrix(cm, SA_CLASS_NAMES))
    names = metrics['class_names']
    print('\nPer-class metrics:')
    print(f'  {"class":<10}{"precision":>11}{"recall":>11}{"f1":>11}')
    for i, nm in enumerate(names):
        print(
            f'  {nm:<10}'
            f'{metrics["precision_per_class"][i]:>11.4f}'
            f'{metrics["recall_per_class"][i]:>11.4f}'
            f'{metrics["f1_per_class"][i]:>11.4f}'
        )
    print(
        f'\nMacro-avg   precision: {metrics["precision_macro"]:.4f}   '
        f'recall: {metrics["recall_macro"]:.4f}   '
        f'F1: {metrics["f1_macro"]:.4f}'
    )
    print(f'Overall accuracy     : {test_acc:.4f}')
    print('========================\n')

    # Retain the historical ``test_acc`` key for existing result consumers.
    return {'test_acc': float(test_acc), **metrics}
