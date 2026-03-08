# Modelling Second Language Acquisition

This repo is a Cambridge Part II Project for the Computer Science Tripos. 
This project aims to explore and expand upon  the [Duolingo SLAM 2018](https://sharedtask.duolingo.com/2018.html) shared task. Further information can be found in the proposal included.

## Overview

This project implements and compares different approaches to **predicting learner errors** (knowledge tracing/GBDT/LR) and **generating exercises** of controllable difficulty across three language tracks: `en_es`, `fr_en`, `es_en`.

## Models

| Model | Type | Description |
|-------|------|-------------|
| **LR** | Baseline | Logistic regression system |
| **GBDT** | Ensemble | LightGBM gradient-boosted trees with feature engineering |
| **DKT/SeqDKT** | Sequential | Deep Knowledge Tracing |
| **BertDKT** | Sequential | BERT token embeddings fed into an LSTM knowledge tracer |
| **LMKT** | Language Model | GPT-2 fine-tuned for knowledge tracing using special tokens (`<Q>`, `<A>`, `<Y>`, `<N>`) |
| **LMKTQG** | Generation | GPT-2 using difficulty embeddings: generates exercises given target difficulty |

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `model` | One of: `lr`, `gbdt`, `dkt`, `bert_dkt`, `lmkt`, `qg`, `sdkt` |  |
| `-t, --track` | Language track: `en_es`, `fr_en`, `es_en`, or `all` | `en_es` |
| `-s, --subset` | Use a subset of training data (number of users) | `None` (full) |
| `-e, --epochs` | Number of training epochs | model-dependent |
| `-t, --train-with-dev` | Include the dev set in training; evaluate on the final test set

### Examples

```bash
# Train GBDT on en_es (full dataset) on the train set, and evaluate on the dev set
python -m main gbdt -t en_es

# Train LMKT on fr_en for 15 epochs
python -m main lmkt -t fr_en -e 15

# Run question generation (trains LMKTQG + evaluate)
python -m main qg -t en_es -e 10

# Quick test with a small subset
python -m main qg -s 1 -e 1
```

## Dataset

This project uses the [Duolingo SLAM Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO) dataset. Each instance is a token within an exercise attempted by a learner, labelled with whether the learner produced it correctly.

> B. Settles, C. Brust, E. Gustafson, M. Hagiwara, and N. Madnani. "Second Language Acquisition Modeling." *Proceedings of the NAACL-HLT Workshop on Innovative Use of NLP for Building Educational Applications (BEA)*. 2018.

> Settles, Burr. 2018. “Data for the 2018 Duolingo Shared Task on Second Language Acquisition Modeling (SLAM).” Harvard Dataverse. https://doi.org/10.7910/DVN/8SWHNO. 
