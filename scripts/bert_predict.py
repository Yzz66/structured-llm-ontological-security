import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
TEXT_COL = 'Text'
LABEL_COL = 'Label'
BATCH_SIZE = 8
MAX_LEN = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--text-col', default=TEXT_COL)
    parser.add_argument('--label-col', default=LABEL_COL)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-len', type=int, default=MAX_LEN)
    parser.add_argument('--run-dirs', nargs='*')
    return parser.parse_args()

class TextDataset(Dataset):

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0)}

def check_columns(df, text_col, label_col, file_name):
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {file_name}: {sorted(missing)}')

def load_label_info(model_path, model_dir):
    candidate_paths = [Path(model_path) / 'label_map.json', Path(model_dir) / 'label_map.json']
    for file_path in candidate_paths:
        if file_path.exists():
            with file_path.open('r', encoding='utf-8') as f:
                info = json.load(f)
            labels = info['labels']
            id2label = {int(k): v for k, v in info['id2label'].items()}
            return (labels, id2label)
    raise FileNotFoundError('label_map.json was not found')

def find_model_paths(model_dir, run_dirs):
    model_dir = Path(model_dir)
    if run_dirs:
        paths = [model_dir / run_dir for run_dir in run_dirs]
    else:
        paths = [p for p in sorted(model_dir.iterdir()) if p.is_dir() and p.name.startswith('run_')]
    if not paths:
        raise FileNotFoundError(f'No run directories were found in {model_dir}')
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing model directories: {missing}')
    return paths

def format_confusion_matrix(cm, labels):
    col_w = max(10, max((len(str(label)) for label in labels)) + 2)
    header = ' ' * col_w + ''.join((f'{label:>{col_w}}' for label in labels))
    lines = [header]
    for label, row in zip(labels, cm):
        lines.append(f'{label:<{col_w}}' + ''.join((f'{int(value):>{col_w}}' for value in row)))
    return '\n'.join(lines)

def mean_std(values):
    values = np.asarray(list(values), dtype=float)
    if len(values) <= 1:
        return (float(values.mean()), 0.0)
    return (float(values.mean()), float(values.std(ddof=1)))

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    test_df = pd.read_excel(args.test_file)
    check_columns(test_df, args.text_col, args.label_col, args.test_file)
    texts = test_df[args.text_col].astype(str).tolist()
    y_true = test_df[args.label_col].astype(str).tolist()
    model_paths = find_model_paths(args.model_dir, args.run_dirs)
    print('Test size:', len(test_df))
    print('Model runs:', [path.name for path in model_paths])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_rows = []
    txt_blocks = []
    for model_path in model_paths:
        run_name = model_path.name
        print(f'Predicting {run_name}')
        labels, id2label = load_label_info(model_path, args.model_dir)
        unknown_labels = sorted(set(y_true) - set(labels))
        if unknown_labels:
            raise ValueError(f'Labels not found in label_map.json: {unknown_labels}')
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        dataset = TextDataset(texts, tokenizer, args.max_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        pred_ids = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                pred_ids.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        y_pred = [id2label[int(label_id)] for label_id in pred_ids]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        kappa = cohen_kappa_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
        metrics_rows.append({'run': run_name, 'accuracy': acc, 'macro_precision': p_macro, 'macro_recall': r_macro, 'macro_f1': f1_macro, 'kappa': kappa})
        pred_df = test_df.copy()
        pred_df['Run'] = run_name
        pred_df['Pred_Label'] = y_pred
        pred_df['Correct'] = (pred_df[args.label_col].astype(str) == pred_df['Pred_Label'].astype(str)).astype(int)
        pred_df['Error_Type'] = np.where(pred_df['Correct'] == 1, '', pred_df[args.label_col].astype(str) + ' -> ' + pred_df['Pred_Label'].astype(str))
        pred_path = output_dir / f'BERT_{run_name}_test_predictions_{timestamp}.xlsx'
        pred_df.to_excel(pred_path, index=False)
        block = []
        block.append(f'[{run_name}] Confusion Matrix')
        block.append('-' * 70)
        block.append('Rows = true labels, columns = predicted labels')
        block.append('Labels: ' + ', '.join(labels))
        block.append('')
        block.append(format_confusion_matrix(cm, labels))
        block.append('')
        block.append(f"Cohen's kappa: {kappa:.6f}")
        block.append(f'Accuracy: {acc:.6f}')
        block.append(f'Macro precision: {p_macro:.6f}')
        block.append(f'Macro recall: {r_macro:.6f}')
        block.append(f'Macro F1: {f1_macro:.6f}')
        block.append('')
        block.append('Classification report:')
        block.append('')
        block.append(report)
        block.append(f'Prediction Excel: {pred_path}')
        block.append('')
        txt_blocks.append('\n'.join(block))
        print(f'{run_name} | accuracy={acc:.4f} | macro_f1={f1_macro:.4f} | kappa={kappa:.4f}')
    metrics_df = pd.DataFrame(metrics_rows)
    metric_cols = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'kappa']
    summary_df = pd.DataFrame([{'metric': metric, 'mean': mean_std(metrics_df[metric])[0], 'std': mean_std(metrics_df[metric])[1]} for metric in metric_cols])
    metrics_path = output_dir / f'BERT_train_test_metrics_{timestamp}.xlsx'
    with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='per_run_metrics', index=False)
        summary_df.to_excel(writer, sheet_name='summary_mean_std', index=False)
    txt_path = output_dir / f'BERT_train_test_results_{timestamp}.txt'
    with txt_path.open('w', encoding='utf-8') as f:
        f.write('BERT Train/Test Evaluation\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'Test file: {args.test_file}\n')
        f.write(f'Model directory: {args.model_dir}\n')
        f.write(f'Number of runs: {len(metrics_df)}\n\n')
        f.write('[Summary]\n')
        f.write('-' * 70 + '\n')
        for _, row in summary_df.iterrows():
            f.write(f"{row['metric']}: mean={row['mean']:.6f}, std={row['std']:.6f}\n")
        f.write('\n')
        f.write('\n'.join(txt_blocks))
    print(f'TXT saved to: {txt_path}')
    print(f'Metrics Excel saved to: {metrics_path}')
if __name__ == '__main__':
    main()
