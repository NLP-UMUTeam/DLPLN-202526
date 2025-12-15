"""
DL-PLN 2025/26 – ATLAS v2

Fine-tuning a Spanish BERT model for sentiment classification.

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
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from atlas_utils import (
    ensure_dir,
    setup_hf_caches,
    set_seed,
    get_device,
)


# Metrics (loaded once)
metric_precision = evaluate.load ("precision")
metric_recall = evaluate.load ("recall")
metric_f1 = evaluate.load ("f1")


def compute_metrics (eval_pred):
    """
    Compute classification metrics from model outputs.

    The Trainer returns:
    - logits: raw model scores per class
    - labels: gold labels (integers)

    We convert logits -> predicted class with argmax, and compute:
    - weighted precision/recall/F1 (accounts for class imbalance)
    - macro F1 (treats all classes equally)

    Args:
        eval_pred (tuple): (logits, labels)

    Returns:
        dict: Metrics to be logged by Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax (logits, axis = -1)
    
    return {
        "precision": metric_precision.compute (
            predictions = preds,
            references = labels,
            average = "weighted"
        )["precision"],

        "recall": metric_recall.compute (
            predictions = preds,
            references = labels,
            average = "weighted"
        )["recall"],

        "f1_weighted": metric_f1.compute (
            predictions = preds,
            references = labels,
            average = "weighted"
        )["f1"],

        "f1_macro": metric_f1.compute (
            predictions = preds,
            references = labels,
            average = "macro"
        )["f1"],
    }


def main ():

    """
    Main training pipeline:

    1) Setup scratch + caches + seed + device
    2) Load dataset
    3) Tokenize
    4) Load model
    5) Configure training arguments
    6) Train + save model/tokenizer
    7) Evaluate + save report
    """

    # 1) Setup
    hf_home, scratch_base = setup_hf_caches ()
    set_seed (42)
    device = get_device ()

    print ("[INFO] hostname:", os.uname ().nodename)
    print ("[INFO] scratch_base:", scratch_base)
    print ("[INFO] HF_HOME:", os.environ.get ("HF_HOME"))
    
    
    # Experiment directories in scratch
    exp_dir = ensure_dir (scratch_base / "out" / "sentiment_cls")
    model_dir = ensure_dir (exp_dir / "model")
    report_dir = ensure_dir (exp_dir / "reports")
    
    
    # 2) Load dataset (cached to scratch)
    dataset_path = "cardiffnlp/tweet_sentiment_multilingual"
    ds = load_dataset (
        dataset_path,
        "spanish",
        cache_dir = str (hf_home / "datasets")
    )
    
    
    # 3) Model + tokenizer
    model_name = "dccuchile/bert-base-spanish-wwm-cased"
    
    tokenizer = AutoTokenizer.from_pretrained (
        model_name,
        cache_dir = str (hf_home / "transformers")
    )
    
    
    # Labels from dataset: 0=neg, 1=neu, 2=pos
    label_names = ["negative", "neutral", "positive"]
    id2label = {i: n for i, n in enumerate (label_names)}
    label2id = {n: i for i, n in enumerate (label_names)}
    
    
    def tok_fn (batch):
        """
        Tokenize a batch of texts.

        We use:
        - truncation: cut long texts
        - max_length: fixed maximum sequence length
        - padding="max_length": produces fixed-size tensors

        Args:
            batch (dict): Batch with a "text" field.

        Returns:
            dict: Tokenized fields (input_ids, attention_mask, etc.)
        """
        return tokenizer (
            batch["text"],
            padding = "max_length",
            truncation = True,
            max_length = 128
        )


    # Tokenize the dataset
    ds_token = ds.map (
        tok_fn,
        batched = True,
        remove_columns = [c for c in ds["train"].column_names if c not in ("label",)]
    )
    
    ds_token = ds_token.rename_column ("label", "labels")
    ds_token.set_format (type = "torch")
    
    
    # 4) Load model
    model = AutoModelForSequenceClassification.from_pretrained (
        model_name,
        num_labels = len (label_names),
        id2label = id2label,
        label2id = label2id,
        cache_dir = str (hf_home / "transformers"),
    ).to (device)
    
    
    # Collator
    collator = DataCollatorWithPadding (tokenizer = tokenizer)
    
    
# 5) Training arguments (student-friendly cheat-sheet)
    #
    # - output_dir: where checkpoints/logs are saved. We use /scratch to avoid quota.
    # - evaluation_strategy="epoch": evaluate once per epoch.
    # - save_strategy="epoch": save one checkpoint per epoch
    # - save_total_limit=1: keep only the latest checkpoint
    # - load_best_model_at_end: reload best checkpoint according to metric_for_best_model
    # - metric_for_best_model="f1_macro". This is good for imbalanced classes (treats classes equally)
    # - per_device_train_batch_size: batch size per GPU (bigger is faster, but more VRAM)
    # - learning_rate: step size for optimizer (too high => unstable; typical 2e-5 for BERT)
    # - fp16: mixed precision to reduce GPU memory usage and often speed up training on GPU
    # - report_to=[]: disables external loggers (W&B, etc.) for simplicity
    args = TrainingArguments (
        output_dir = str (exp_dir / "trainer_runs"),
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit = 1,
        load_best_model_at_end = True,
        metric_for_best_model = "f1_macro",
        greater_is_better = True,

        num_train_epochs = 1,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,

        learning_rate = 2e-5,
        weight_decay = 0.01,

        seed = 42,
        fp16 = torch.cuda.is_available (),

        logging_steps = 50,
        report_to = [],
    )
    
    
    # Configure the trainer
    trainer = Trainer (
        model = model,
        args = args,
        train_dataset = ds_token["train"],
        eval_dataset = ds_token["validation"],
        tokenizer = tokenizer,
        data_collator = collator,
        compute_metrics = compute_metrics,
    )
    
    
    # Save a small README (useful for students and reproducibility)
    readme_path = exp_dir / "README.txt"
    readme_path.write_text (
        "Experiment: Sentiment classification (Spanish)\n"
        f"Dataset: {dataset_path}\n"
        f"Model: {model_name}\n"
        f"Labels: " + (' / '.join (label_names)) + "\n"
        f"Epochs: {args.num_train_epochs}\n"
        f"Batch train: {args.per_device_train_batch_size}\n"
        f"Batch eval: {args.per_device_eval_batch_size}\n"
        f"Learning rate: {args.learning_rate}\n"
        f"Weight decay: {args.weight_decay}\n"
        f"FP16: {args.fp16}\n",
        encoding = "utf-8"
    )
    
    
    # Train the model
    trainer.train ()
    
    
    # Save model and tokenizer to SCRATCH. Note that these are large files
    trainer.save_model (str (model_dir))
    tokenizer.save_pretrained (str (model_dir))
    print (f"[OK] Model saved to: {model_dir}")
    
    
    # 7) Evaluation report
    pred = trainer.predict (ds_token["validation"])
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax (axis = -1)

    report_dict = classification_report (
        y_true,
        y_pred,
        target_names = label_names,
        output_dict = True,
        digits = 4
    )
    
    
    # Generate report
    report_df = pd.DataFrame (report_dict).transpose ()


    # Save report in scratch
    report_csv_scratch = report_dir / "classification_report.csv"
    report_df.to_csv (report_csv_scratch, index = True)


    # Copy only the CSV to HOME
    home_reports = ensure_dir (Path.home () / "reports")
    report_csv_home = home_reports / "sentiment_classification_report.csv"
    report_df.to_csv (report_csv_home, index = True)

    print (f"[OK] Report saved to scratch: {report_csv_scratch}")
    print (f"[OK] Report copied to HOME:  {report_csv_home}")


if __name__ == "__main__":
    main ()
