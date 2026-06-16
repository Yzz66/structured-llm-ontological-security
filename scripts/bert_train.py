import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
TEXT_COL = 'Text'
LABEL_COL = 'Label'
MODEL_NAME = 'bert-base-uncased'
N_RUNS = 5
RANDOM_STATE = 42
EPOCHS = 4
BATCH_SIZE = 8
MAX_LEN = 128
LR = 1e-05
WARMUP_RATIO = 0.0
LABELS = ['Denial', 'Insult', 'Pride', 'Shame']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--model-output-dir', required=True)
    parser.add_argument('--text-col', default=TEXT_COL)
    parser.add_argument('--label-col', default=LABEL_COL)
    parser.add_argument('--model-name', default=MODEL_NAME)
    parser.add_argument('--n-runs', type=int, default=N_RUNS)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-len', type=int, default=MAX_LEN)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO)
    parser.add_argument('--labels', nargs='+', default=LABELS)
    return parser.parse_args()

class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_columns(df, text_col, label_col, file_name):
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {file_name}: {sorted(missing)}')

def save_json(obj, file_path):
    with Path(file_path).open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)

def main():
    args = parse_args()
    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    train_df = pd.read_excel(args.train_file)
    check_columns(train_df, args.text_col, args.label_col, args.train_file)
    texts = train_df[args.text_col].astype(str).tolist()
    labels_raw = train_df[args.label_col].astype(str).tolist()
    labels_order = list(args.labels)
    unknown_labels = sorted(set(labels_raw) - set(labels_order))
    if unknown_labels:
        raise ValueError(f'Labels not defined in --labels: {unknown_labels}')
    label2id = {label: i for i, label in enumerate(labels_order)}
    id2label = {i: label for label, i in label2id.items()}
    labels = [label2id[label] for label in labels_raw]
    label_info = {'labels': labels_order, 'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}
    save_json(label_info, model_output_dir / 'label_map.json')
    print('Labels:', labels_order)
    print('Train distribution:', Counter(labels_raw))
    print('Train size:', len(train_df))
    train_logs = []
    for run_id in range(1, args.n_runs + 1):
        run_seed = args.random_state + run_id - 1
        set_seed(run_seed)
        print(f'Run {run_id:02d}/{args.n_runs} | seed={run_seed}')
        run_dir = model_output_dir / f'run_{run_id:02d}'
        run_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        train_dataset = TextDataset(texts, labels, tokenizer, args.max_len)
        generator = torch.Generator()
        generator.manual_seed(run_seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_order), id2label={i: label for i, label in enumerate(labels_order)}, label2id=label2id).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        model.train()
        for epoch in range(1, args.epochs + 1):
            epoch_losses = []
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                y = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_losses.append(loss.item())
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            print(f'Run {run_id:02d} | Epoch {epoch}/{args.epochs} | loss={avg_loss:.6f}')
            train_logs.append({'run': run_id, 'seed': run_seed, 'epoch': epoch, 'loss': avg_loss})
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)
        run_config = {'run': run_id, 'seed': run_seed, 'train_file': args.train_file, 'model_name': args.model_name, 'epochs': args.epochs, 'batch_size': args.batch_size, 'max_len': args.max_len, 'lr': args.lr, 'warmup_ratio': args.warmup_ratio, 'labels': labels_order, 'train_distribution': dict(Counter(labels_raw)), 'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        save_json(run_config, run_dir / 'training_config.json')
        save_json(label_info, run_dir / 'label_map.json')
        print(f'Model saved to: {run_dir}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = model_output_dir / f'BERT_train_log_{timestamp}.xlsx'
    pd.DataFrame(train_logs).to_excel(log_path, index=False)
    print(f'Training log saved to: {log_path}')
if __name__ == '__main__':
    main()
