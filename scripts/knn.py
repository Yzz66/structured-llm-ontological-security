import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
TEXT_COL = 'Text'
LABEL_COL = 'Label'
N_RUNS = 1
RANDOM_STATE = 42
SHUFFLE_TRAIN_EACH_RUN = False
N_NEIGHBORS = 5
DISTANCE_WEIGHTED = True
PREFERRED_LABEL_ORDER = ['Denial', 'Insult', 'Pride', 'Shame']
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
METRIC = 'cosine'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--text-col', default=TEXT_COL)
    parser.add_argument('--label-col', default=LABEL_COL)
    parser.add_argument('--n-runs', type=int, default=N_RUNS)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    parser.add_argument('--shuffle-train-each-run', action='store_true')
    return parser.parse_args()

def check_columns(df, text_col, label_col, file_name):
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {file_name}: {sorted(missing)}')

def get_label_order(y_train, y_test):
    observed = set(y_train) | set(y_test)
    labels = [label for label in PREFERRED_LABEL_ORDER if label in observed]
    labels.extend(sorted(observed - set(labels)))
    return labels

def build_pipeline():
    return Pipeline([('tfidf', TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF)), ('clf', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance' if DISTANCE_WEIGHTED else 'uniform', metric=METRIC))])

def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    return {'accuracy': acc, 'macro_precision': p_macro, 'macro_recall': r_macro, 'macro_f1': f1_macro, 'kappa': kappa}

def format_confusion_matrix(cm, labels):
    col_w = max(10, max((len(str(label)) for label in labels)) + 2)
    header = ' ' * col_w + ''.join((f'{label:>{col_w}}' for label in labels))
    lines = [header]
    for label, row in zip(labels, cm):
        lines.append(f'{label:<{col_w}}' + ''.join((f'{int(value):>{col_w}}' for value in row)))
    return '\n'.join(lines)

def mean_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return (float(values.mean()), 0.0)
    return (float(values.mean()), float(values.std(ddof=1)))

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_excel(args.train_file)
    test_df = pd.read_excel(args.test_file)
    check_columns(train_df, args.text_col, args.label_col, args.train_file)
    check_columns(test_df, args.text_col, args.label_col, args.test_file)
    train_df = train_df.dropna(subset=[args.text_col, args.label_col]).copy()
    test_df = test_df.dropna(subset=[args.text_col, args.label_col]).copy()
    X_train_base = train_df[args.text_col].astype(str).tolist()
    y_train_base = train_df[args.label_col].astype(str).tolist()
    X_test = test_df[args.text_col].astype(str).tolist()
    y_test = test_df[args.label_col].astype(str).tolist()
    labels = get_label_order(y_train_base, y_test)
    print('Detected labels:', labels)
    print('Train size:', len(train_df))
    print('Test size:', len(test_df))
    print('Train distribution:', Counter(y_train_base))
    print('Test distribution:', Counter(y_test))
    metrics_records = []
    prediction_frames = []
    confusion_matrices = []
    reports = []
    for run_id in range(1, args.n_runs + 1):
        seed = args.random_state + run_id - 1
        if args.shuffle_train_each_run:
            current_train_df = shuffle(train_df, random_state=seed).reset_index(drop=True)
            X_train = current_train_df[args.text_col].astype(str).tolist()
            y_train = current_train_df[args.label_col].astype(str).tolist()
        else:
            X_train = X_train_base
            y_train = y_train_base
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, labels)
        metrics['run'] = run_id
        metrics['random_state'] = seed
        metrics_records.append(metrics)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        confusion_matrices.append(cm)
        reports.append(classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0))
        pred_df = test_df.copy()
        pred_df.insert(0, 'Run', run_id)
        pred_df['Pred_Label'] = y_pred
        pred_df['Correct'] = (pred_df[args.label_col].astype(str) == pred_df['Pred_Label'].astype(str)).astype(int)
        pred_df['Error_Type'] = np.where(pred_df['Correct'] == 1, '', pred_df[args.label_col].astype(str) + ' -> ' + pred_df['Pred_Label'].astype(str))
        prediction_frames.append(pred_df)
        print(f"Run {run_id:02d} | random_state={seed} | accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | kappa={metrics['kappa']:.4f}")
    metrics_df = pd.DataFrame(metrics_records)
    pred_all_df = pd.concat(prediction_frames, ignore_index=True)
    metric_cols = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'kappa']
    summary_df = pd.DataFrame([{'metric': metric, 'mean': mean_std(metrics_df[metric])[0], 'std': mean_std(metrics_df[metric])[1]} for metric in metric_cols])
    last_cm = confusion_matrices[-1]
    last_report = reports[-1]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = output_dir / f'KNN_train_test_results_{timestamp}.txt'
    with txt_path.open('w', encoding='utf-8') as f:
        f.write('KNN Train/Test Evaluation\n')
        f.write('=' * 70 + '\n\n')
        f.write('[Data and Parameters]\n')
        f.write('-' * 70 + '\n')
        f.write(f'Train file: {args.train_file}\n')
        f.write(f'Test file: {args.test_file}\n')
        f.write(f'Train size: {len(train_df)}\n')
        f.write(f'Test size: {len(test_df)}\n')
        f.write(f"Labels: {', '.join(labels)}\n")
        f.write(f'Train distribution: {dict(Counter(y_train_base))}\n')
        f.write(f'Test distribution: {dict(Counter(y_test))}\n')
        f.write(f'N_RUNS: {args.n_runs}\n')
        f.write(f'Base random_state: {args.random_state}\n')
        f.write(f'Shuffle train each run: {args.shuffle_train_each_run}\n')
        f.write(f'TF-IDF: ngram_range={TFIDF_NGRAM_RANGE}, min_df={TFIDF_MIN_DF}, max_df={TFIDF_MAX_DF}\n')
        f.write(f"KNN: n_neighbors={N_NEIGHBORS}, weights={('distance' if DISTANCE_WEIGHTED else 'uniform')}, metric={METRIC}\n\n")
        f.write('[Summary]\n')
        f.write('-' * 70 + '\n')
        for _, row in summary_df.iterrows():
            f.write(f"{row['metric']}: mean={row['mean']:.6f}, std={row['std']:.6f}\n")
        f.write('\n[Per-run Metrics]\n')
        f.write('-' * 70 + '\n')
        f.write(metrics_df[['run', 'random_state'] + metric_cols].to_string(index=False))
        f.write('\n\n[Confusion Matrix: Last Run]\n')
        f.write('-' * 70 + '\n')
        f.write('Rows = true labels, columns = predicted labels\n')
        f.write(format_confusion_matrix(last_cm, labels))
        f.write('\n\n[Classification Report: Last Run]\n')
        f.write('-' * 70 + '\n')
        f.write(last_report)
    pred_path = output_dir / f'KNN_train_test_predictions_{timestamp}.xlsx'
    with pd.ExcelWriter(pred_path, engine='openpyxl') as writer:
        pred_all_df.to_excel(writer, index=False, sheet_name='predictions_all_runs')
        metrics_df[['run', 'random_state'] + metric_cols].to_excel(writer, index=False, sheet_name='per_run_metrics')
        summary_df.to_excel(writer, index=False, sheet_name='summary_mean_std')
        pd.DataFrame(last_cm, index=labels, columns=labels).to_excel(writer, sheet_name='confusion_matrix_last')
    print(f'TXT saved to: {txt_path}')
    print(f'Excel saved to: {pred_path}')
if __name__ == '__main__':
    main()
