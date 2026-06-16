import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
TEXT_COL = 'Text'
LABEL_COL = 'Label'
N_RUNS = 5
RANDOM_STATE = 42
PREFERRED_LABEL_ORDER = ['Denial', 'Insult', 'Pride', 'Shame']
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
HIDDEN_LAYER_SIZES = (100, 200)
ACTIVATION = 'relu'
SOLVER = 'adam'
LEARNING_RATE_INIT = 0.001
ALPHA = 1e-05
BATCH_SIZE = 'auto'
LEARNING_RATE = 'adaptive'
MAX_ITER = 200
EARLY_STOPPING = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--text-col', default=TEXT_COL)
    parser.add_argument('--label-col', default=LABEL_COL)
    parser.add_argument('--n-runs', type=int, default=N_RUNS)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
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

def build_pipeline(seed):
    return Pipeline([('tfidf', TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF)), ('clf', MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=ACTIVATION, solver=SOLVER, learning_rate_init=LEARNING_RATE_INIT, alpha=ALPHA, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, max_iter=MAX_ITER, early_stopping=EARLY_STOPPING, random_state=seed))])

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
    X_train = train_df[args.text_col].astype(str).tolist()
    y_train = train_df[args.label_col].astype(str).tolist()
    X_test = test_df[args.text_col].astype(str).tolist()
    y_test = test_df[args.label_col].astype(str).tolist()
    labels = get_label_order(y_train, y_test)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_train_enc = label_encoder.transform(y_train)
    print('Detected labels:', labels)
    print('Train size:', len(train_df))
    print('Test size:', len(test_df))
    print('Train distribution:', Counter(y_train))
    print('Test distribution:', Counter(y_test))
    metrics_records = []
    confusion_matrices = []
    reports = {}
    pred_df = test_df.copy()
    for run_id in range(1, args.n_runs + 1):
        seed = args.random_state + run_id - 1
        pipeline = build_pipeline(seed)
        pipeline.fit(X_train, y_train_enc)
        y_pred_enc = pipeline.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        metrics = compute_metrics(y_test, y_pred, labels)
        metrics['run'] = run_id
        metrics['random_state'] = seed
        metrics_records.append(metrics)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        confusion_matrices.append(cm)
        reports[run_id] = classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0)
        pred_df[f'Pred_Label_Run_{run_id}'] = y_pred
        pred_df[f'Correct_Run_{run_id}'] = (pred_df[args.label_col].astype(str) == pred_df[f'Pred_Label_Run_{run_id}'].astype(str)).astype(int)
        pred_df[f'Error_Type_Run_{run_id}'] = np.where(pred_df[f'Correct_Run_{run_id}'] == 1, '', pred_df[args.label_col].astype(str) + ' -> ' + pred_df[f'Pred_Label_Run_{run_id}'].astype(str))
        print(f"Run {run_id:02d} | random_state={seed} | accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | kappa={metrics['kappa']:.4f}")
    metrics_df = pd.DataFrame(metrics_records)
    metric_cols = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'kappa']
    summary_df = pd.DataFrame([{'metric': metric, 'mean': mean_std(metrics_df[metric])[0], 'std': mean_std(metrics_df[metric])[1]} for metric in metric_cols])
    cm_sum = np.sum(confusion_matrices, axis=0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = output_dir / f'MLP_train_test_results_{timestamp}.txt'
    with txt_path.open('w', encoding='utf-8') as f:
        f.write('MLP Train/Test Evaluation\n')
        f.write('=' * 70 + '\n\n')
        f.write('[Data and Parameters]\n')
        f.write('-' * 70 + '\n')
        f.write(f'Train file: {args.train_file}\n')
        f.write(f'Test file: {args.test_file}\n')
        f.write(f'Train size: {len(train_df)}\n')
        f.write(f'Test size: {len(test_df)}\n')
        f.write(f"Labels: {', '.join(labels)}\n")
        f.write(f'Train distribution: {dict(Counter(y_train))}\n')
        f.write(f'Test distribution: {dict(Counter(y_test))}\n')
        f.write(f'N_RUNS: {args.n_runs}\n')
        f.write(f'Base random_state: {args.random_state}\n')
        f.write(f'TF-IDF: ngram_range={TFIDF_NGRAM_RANGE}, min_df={TFIDF_MIN_DF}, max_df={TFIDF_MAX_DF}\n')
        f.write(f'MLP: hidden_layer_sizes={HIDDEN_LAYER_SIZES}, activation={ACTIVATION}, solver={SOLVER}, learning_rate_init={LEARNING_RATE_INIT}, alpha={ALPHA}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}, max_iter={MAX_ITER}, early_stopping={EARLY_STOPPING}\n\n')
        f.write('[Summary]\n')
        f.write('-' * 70 + '\n')
        for _, row in summary_df.iterrows():
            f.write(f"{row['metric']}: mean={row['mean']:.6f}, std={row['std']:.6f}\n")
        f.write('\n[Per-run Metrics]\n')
        f.write('-' * 70 + '\n')
        f.write(metrics_df[['run', 'random_state'] + metric_cols].to_string(index=False))
        f.write('\n\n[Confusion Matrix]\n')
        f.write('-' * 70 + '\n')
        f.write('Rows = true labels, columns = predicted labels\n')
        f.write('Matrix = sum over runs\n')
        f.write(format_confusion_matrix(cm_sum, labels))
        f.write('\n\n[Classification Reports]\n')
        f.write('-' * 70 + '\n')
        for run_id, report in reports.items():
            f.write(f'\nRun {run_id}\n')
            f.write('~' * 40 + '\n')
            f.write(report)
            f.write('\n')
    pred_path = output_dir / f'MLP_train_test_predictions_{timestamp}.xlsx'
    metrics_path = output_dir / f'MLP_train_test_metrics_{timestamp}.xlsx'
    pred_df.to_excel(pred_path, index=False)
    with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:
        metrics_df[['run', 'random_state'] + metric_cols].to_excel(writer, sheet_name='per_run_metrics', index=False)
        summary_df.to_excel(writer, sheet_name='summary_mean_std', index=False)
        pd.DataFrame(cm_sum, index=labels, columns=labels).to_excel(writer, sheet_name='confusion_matrix_sum')
    print(f'TXT saved to: {txt_path}')
    print(f'Prediction Excel saved to: {pred_path}')
    print(f'Metrics Excel saved to: {metrics_path}')
if __name__ == '__main__':
    main()
