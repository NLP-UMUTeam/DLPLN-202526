"""
DL-PLN 2025/26 – ATLAS v2

Fine-tuning a Spanish Transformer model for Named Entity Recognition (NER).

This script performs token classification (NER) using the WikiAnn dataset
for Spanish ("unimelb-nlp/wikiann", configuration "es"). The task consists
of assigning a label to each token in a sentence following the BIO scheme
(e.g., B-PER, I-LOC, O).

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Ronghao Pan <ronghao.pan@um.es>
@author Rafael Valencia-García <valencia@um.es>
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from atlas_utils import (
    ensure_dir,
    setup_hf_caches,
    set_seed,
    get_device,
)


# ------------------------------------------------------------
# NER labels (BIO scheme)
# ------------------------------------------------------------
label_names = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
]


# Seqeval is the standard evaluation library for NER.
metric = evaluate.load ("seqeval")


def tokenize_and_align_labels (examples, tokenizer):
    """
    Tokenize the input tokens and align word-level NER labels to subword tokens.

    Why is this needed?
    - Modern Transformer tokenizers split some words into subwords.
    - NER labels are provided at the *word* level, but the model predicts at the *token* level.
    - We must align each original word label to the corresponding subword tokens.
    - Special tokens (CLS/SEP) receive label -100 so they are ignored in the loss.

    Args:
        examples (dict): Batch containing:
            - examples["tokens"]: list of tokenized sentences (word-level tokens)
            - examples["ner_tags"]: list of label ids (word-level)
        tokenizer: Hugging Face tokenizer.

    Returns:
        dict: Tokenized inputs plus aligned "labels".
    """
    label_all_tokens = True

    tokenized = tokenizer (
        examples["tokens"],
        truncation = True,
        is_split_into_words = True
    )

    labels = []

    for i, word_labels in enumerate (examples["ner_tags"]):
        word_ids = tokenized.word_ids (batch_index = i)
        current_word = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens (CLS/SEP/PAD) -> ignore in loss
            if word_idx is None:
                label_ids.append (-100)

            # First subword of a new word -> use the word label
            elif word_idx != current_word:
                label_ids.append (word_labels[word_idx])

            # Same word (subword token)
            else:
                label_ids.append (word_labels[word_idx] if label_all_tokens else -100)

            current_word = word_idx

        labels.append (label_ids)

    tokenized["labels"] = labels
    return tokenized



def compute_metrics (eval_pred):
    """
    Compute standard NER metrics using seqeval.

    Seqeval expects lists of label strings per sentence, ignoring -100 tokens.

    Args:
        eval_pred (tuple): (predictions, labels)
            - predictions: logits [batch, seq_len, num_labels]
            - labels: gold ids [batch, seq_len]

    Returns:
        dict: overall_precision, overall_recall, overall_f1, overall_accuracy
    """
    predictions, labels = eval_pred
    predictions = np.argmax (predictions, axis = -1)

    
    # Calculate true predictions
    true_predictions = [
        [label_names[p] for (p, l) in zip (pred, lab) if l != -100]
        for pred, lab in zip (predictions, labels)
    ]
    
    
    # Calculate true labels
    true_labels = [
        [label_names[l] for l in lab if l != -100]
        for lab in labels
    ]

    results = metric.compute (predictions = true_predictions, references = true_labels)

    return {
        "precision": results.get ("overall_precision", 0.0),
        "recall": results.get ("overall_recall", 0.0),
        "f1": results.get ("overall_f1", 0.0),
        "accuracy": results.get ("overall_accuracy", 0.0),
    }
    


def main ():
    """
    Student-friendly pipeline:

    1) Setup scratch + caches + seed + device
    2) Load WikiAnn (Spanish)
    3) Tokenize + align labels to subwords
    4) Load model for token classification
    5) Configure TrainingArguments (documented)
    6) Train + save model/tokenizer to /scratch
    7) Evaluate + save a CSV report to /scratch and copy it to $HOME
    """

    # 1) Setup (scratch and reproducibility)
    hf_home, scratch_base = setup_hf_caches ()
    set_seed (42)
    device = get_device ()

    print ("[INFO] hostname:", os.uname ().nodename)
    print ("[INFO] scratch_base:", scratch_base)
    print ("[INFO] HF_HOME:", os.environ.get ("HF_HOME"))
    
    
    # Experiment paths (SCRATCH)
    exp_dir = ensure_dir (scratch_base / "out" / "ner_wikiann_es")
    model_dir = ensure_dir (exp_dir / "model")
    report_dir = ensure_dir (exp_dir / "reports")
    
    
    # 2) Load dataset (cached to scratch)
    dataset_path = "unimelb-nlp/wikiann"
    dataset = load_dataset (
        dataset_path,
        "es",
        trust_remote_code = True,
        cache_dir = str (hf_home / "datasets")
    )
    
    train_ds = dataset["train"]
    eval_ds = dataset["validation"]
    
    
    # 3) Tokenizer + preprocessing
    model_name = "dccuchile/bert-base-spanish-wwm-cased"

    tokenizer = AutoTokenizer.from_pretrained (
        model_name,
        cache_dir = str (hf_home / "transformers")
    )

    # Remove raw columns after tokenization to keep datasets lighter in memory.
    # We keep only model inputs + labels.
    train_ds = train_ds.map (
        tokenize_and_align_labels,
        batched = True,
        fn_kwargs = {"tokenizer": tokenizer},
        remove_columns = train_ds.column_names
    )

    eval_ds = eval_ds.map (
        tokenize_and_align_labels,
        batched = True,
        fn_kwargs = {"tokenizer": tokenizer},
        remove_columns = eval_ds.column_names
    )

    # 4) Model
    num_labels = len (label_names)
    label2id = {label: i for i, label in enumerate (label_names)}
    id2label = {i: label for i, label in enumerate (label_names)}

    model = AutoModelForTokenClassification.from_pretrained (
        model_name,
        num_labels = num_labels,
        id2label = id2label,
        label2id = label2id,
        cache_dir = str (hf_home / "transformers")
    ).to (device)

    data_collator = DataCollatorForTokenClassification (tokenizer = tokenizer)
    
    
    # 5) Training arguments (look sentence_classification for more details)
    training_args = TrainingArguments (
        output_dir = str (exp_dir / "trainer_runs"),
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit = 1,
        load_best_model_at_end = True,
        metric_for_best_model = "f1",
        greater_is_better = True,

        num_train_epochs = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,

        learning_rate = 2e-5,
        weight_decay = 0.01,

        seed = 42,
        fp16 = torch.cuda.is_available (),

        logging_steps = 50,
        report_to = [],
    )
    
    
    # Save a README for reproducibility (very useful in HPC environments)
    readme_path = exp_dir / "README.txt"
    readme_path.write_text (
        "Experiment: NER (Token Classification)\n"
        f"Dataset: {dataset_path}\n"
        f"Model: {model_name}\n"
        f"Epochs: {training_args.num_train_epochs}\n"
        f"Batch train: {training_args.per_device_train_batch_size}\n"
        f"Batch eval: {training_args.per_device_eval_batch_size}\n"
        f"Learning rate: {training_args.learning_rate}\n"
        f"Weight decay: {training_args.weight_decay}\n"
        f"FP16: {training_args.fp16}\n",
        encoding = "utf-8"
    )
    
    
    # 6) Trainer
    trainer = Trainer (
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )

    trainer.train ()
    
    
    # Save model + tokenizer to SCRATCH
    trainer.save_model (str (model_dir))
    tokenizer.save_pretrained (str (model_dir))

    print (f"[OK] Model and tokenizer saved to: {model_dir}")
    
    
    # 7) Evaluation + report
    pred = trainer.predict (eval_ds)
    
    
    # Trainer already logs metrics, but we also save them to CSV
    metrics = pred.metrics
    report_df = pd.DataFrame ([{
        "eval_loss": metrics.get ("test_loss", metrics.get ("eval_loss", None)),
        "precision": metrics.get ("test_precision", metrics.get ("eval_precision", None)),
        "recall": metrics.get ("test_recall", metrics.get ("eval_recall", None)),
        "f1": metrics.get ("test_f1", metrics.get ("eval_f1", None)),
        "accuracy": metrics.get ("test_accuracy", metrics.get ("eval_accuracy", None)),
    }])
    
    
    report_csv_scratch = report_dir / "ner_report.csv"
    report_df.to_csv (report_csv_scratch, index = False)
    
    
    # Copy the report to HOME
    home_reports = ensure_dir (Path.home () / "reports")
    report_csv_home = home_reports / "ner_wikiann_es_report.csv"
    report_df.to_csv (report_csv_home, index = False)

    print (f"[OK] Report saved to scratch: {report_csv_scratch}")
    print (f"[OK] Report copied to HOME:  {report_csv_home}")


if __name__ == "__main__":
    main ()