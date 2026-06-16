# Replication Materials

This repository contains data, prompts, figures, and Python scripts for reproducing the experiments reported in the paper.

## Repository layout

```text
data/         Input data files used by the replication scripts
figures/      Generated figures used in the manuscript
prompts/      Prompt templates and prompt-related materials
scripts/      Python scripts for model evaluation and OSKR/RAG experiments
README.md     Replication instructions
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Input data

The supervised evaluation scripts expect Excel files with at least these columns:

- `Text`: text instance to classify
- `Label`: gold label, one of `Pride`, `Shame`, `Denial`, or `Insult`

Some LLM scripts optionally use a `Rank` column for row filtering. Rank filtering can be disabled from the command line.

## Traditional, lexicon-based, and BERT baselines

```bash
python scripts/svm.py --train-file data/Data_Train_320.xlsx --test-file data/Data_Test_80.xlsx --output-dir outputs/svm
python scripts/naive_bayes.py --train-file data/Data_Train_320.xlsx --test-file data/Data_Test_80.xlsx --output-dir outputs/naive_bayes
python scripts/knn.py --train-file data/Data_Train_320.xlsx --test-file data/Data_Test_80.xlsx --output-dir outputs/knn
python scripts/random_forest.py --train-file data/Data_Train_320.xlsx --test-file data/Data_Test_80.xlsx --output-dir outputs/random_forest
python scripts/mlp.py --train-file data/Data_Train_320.xlsx --test-file data/Data_Test_80.xlsx --output-dir outputs/mlp
python scripts/nrc_emotion_lexicon.py --data-path data/Data_Test_80.xlsx --lexicon-path data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt --output-dir outputs/nrc_emotion_lexicon
python scripts/bert_train.py --train-file data/Data_Train_320.xlsx --model-output-dir outputs/bert/models
python scripts/bert_predict.py --test-file data/Data_Test_80.xlsx --model-dir outputs/bert/models --output-dir outputs/bert/results
```

## Prompt-based LLM experiments

Set the API key for the selected provider before running LLM-based scripts.

```bash
export NVIDIA_API_KEY=your_key_here
export DASHSCOPE_API_KEY=your_key_here
```

Examples:

```bash
python scripts/baseline_prompting.py --input-file data/Data_Test_80.xlsx --output-dir outputs/baseline_prompting --model-preset llama-3.1-70b --no-rank-filter
python scripts/conceptual_decomposition.py --input-file data/Data_Test_80.xlsx --output-dir outputs/conceptual_decomposition --model-preset llama-3.1-70b --no-rank-filter
python scripts/self_reflective_validation.py --input-file data/Data_Test_80.xlsx --output-dir outputs/self_reflective_validation --model-preset llama-3.1-70b --no-rank-filter
python scripts/few_shot_prompting.py --input-file data/Data_Test_80.xlsx --output-dir outputs/few_shot_prompting --model-preset llama-3.1-70b
```

## OSKR index construction and retrieval-augmented grounding

Build the FAISS index from PDF files:

```bash
python scripts/build_oskr_index.py --input-dir data/oskr_papers --output-dir data/oskr/sentence_index --chunking sentence
```

Run retrieval-augmented grounding:

```bash
python scripts/retrieval_augmented_grounding.py --input-file data/Data_Test_80.xlsx --output-dir outputs/retrieval_augmented_grounding --index-path data/oskr/sentence_index/sentence_index.faiss --meta-path data/oskr/sentence_index/sentence_meta.json --model-preset llama-3.1-70b
```

## Notes

The scripts use relative command-line paths and do not contain local machine paths. Outputs are written to the directory specified by `--output-dir`.
