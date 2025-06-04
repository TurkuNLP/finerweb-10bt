
# FinerWeb-10BT

Code for paper "FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering". This repository contains tools for cleaning web data line by line using LLMs and training language models on the filtered data.

## Pipeline Steps

### 1. Generate LLM Annotations
Generate initial line quality annotations using GPT-4o mini:
```bash
python annotate_line_quality.py
```

### 2. Train and Use Classifier
Navigate to classifier directory:
```bash
cd classifier
```

#### Prepare Data
Stratify the data for training:
```bash
python stratify.py
```

#### Train Model
Fine-tune DeBERTa-v3 classifier:
```bash
python run.py --train --base_model "microsoft/deberta-v3-base"
```

#### Create Platt Scaler
Calibrate model probabilities:
```bash
python run.py --finetuned_model_path FINETUNED_MODEL_PATH \
              --base_model "microsoft/deberta-v3-base" --platt
```

#### Predict Quality Scores
Apply classifier to FineWeb-10BT:
```bash
python run.py --finetuned_model_path FINETUNED_MODEL_PATH \
              --base_model "microsoft/deberta-v3-base" --predict_fineweb
```

### 3. Tokenize Data
Prepare data for GPT-2 training:
```bash
cd GPT2/scripts
python tokenize_documents_parallel_exquisiteweb.py \
    --dataset_path PATH_TO_FINERWEB-10BT \
    --local_dir LOCAL_DIR \
    --quality_threshold QUALITY_THRESHOLD
```

### 4. Train GPT-2
Train GPT-2 on filtered data:
```bash
cd GPT2
./slurm.sh torchrun --standalone --nproc_per_node=4 scripts/train_gpt.py \
    --data_root TOKENIZED_DATASET \
    --n_batches 16 \
    --n_tokens 1024 \
    --vocab_size 50304 \
    --emb_dim 768 \
    --context_length 1024 \
    --n_layers 12 \
    --n_heads 12 \
    --log_dir LOG_DIRECTORY
```

## Data

The enhanced FineWeb-10BT dataset with quality scores is available at:
https://huggingface.co/datasets/TurkuNLP/finerweb-10bt

## Citation

More details in the [paper](https://dspace.ut.ee/items/71fd1cd0-28c4-423c-8fe4-f66707b8ad62).

If you use our work, please cite:

@inproceedings{finerweb10bt, \
    author = {Henriksson, Erik and Tarkka, Otto and Ginter, Filip}, \
    title = {FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering}, \
    booktitle = {Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025)}, \
    year = {2025},\
    url = {https://hdl.handle.net/10062/107218} \
}

## License

MIT
