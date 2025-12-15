"""
DL-PLN 2025/26 – ATLAS v2

Fine-tuning a Question Answering (QA) model using the SQuAD dataset.

What this script does:
- Trains a QA model to answer questions given a context paragraph.
- Uses the SQuAD dataset (English) from Hugging Face: "squad".
- Saves heavy artifacts (model/tokenizer/checkpoints) to /scratch/<user>.
- Copies small artifacts (CSV report + a demo example) back to $HOME.

Why SQuAD?
- It is the most classic benchmark for extractive QA.
- The answer is a text span within the provided context.

Authors:
@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Ronghao Pan <ronghao.pan@um.es>
@author Rafael Valencia-García <valencia@um.es>
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

from atlas_utils import (
    ensure_dir,
    setup_hf_caches,
    set_seed,
    get_device,
)



def preprocess_function (examples, tokenizer):
    """
    Tokenize (question, context) and align answer spans to token start/end positions.

    In extractive QA, the model learns to predict:
    - start_position: token index where the answer begins
    - end_position: token index where the answer ends

    This function:
    1) Tokenizes question + context (context may be truncated to max_length)
    2) Uses offset_mapping to map tokens back to character positions
    3) Converts character-based answer spans into token indices

    Args:
        examples (dict): Batch with keys "question", "context", "answers"
        tokenizer: Hugging Face tokenizer

    Returns:
        dict: Tokenized inputs with "start_positions" and "end_positions"
    """
    questions = [q.strip () for q in examples["question"]]

    inputs = tokenizer (
        questions,
        examples["context"],
        max_length = 512,
        truncation = "only_second",
        return_offsets_mapping = True,
        padding = "max_length",
    )

    offset_mapping = inputs.pop ("offset_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate (offset_mapping):
        answer = answers[i]

        start_char = answer["answer_start"][0]
        end_char = start_char + len (answer["text"][0])

        sequence_ids = inputs.sequence_ids (i)

        # Find the context span in the tokenized sequence
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while idx < len (sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If answer is outside the truncated context window, set to (0, 0)
        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append (0)
            end_positions.append (0)
        else:
            # Token start
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append (idx - 1)

            # Token end
            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append (idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


def predict_answer (model, tokenizer, question: str, context: str, device):
    """
    Run a single QA inference and return the extracted answer text.

    This is a *didactic* inference: we just take argmax start/end.
    (Proper QA evaluation requires post-processing across long contexts.)

    Args:
        model: QA model
        tokenizer: corresponding tokenizer
        question (str): question text
        context (str): context paragraph
        device: torch.device

    Returns:
        str: predicted answer span
    """
    inputs = tokenizer (
        question,
        context,
        return_tensors = "pt",
        truncation = "only_second",
        max_length = 512
    ).to (device)

    with torch.no_grad ():
        outputs = model (**inputs)

    start_idx = int (torch.argmax (outputs.start_logits, dim = -1).item ())
    end_idx = int (torch.argmax (outputs.end_logits, dim = -1).item ())

    if end_idx < start_idx:
        end_idx = start_idx

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    return tokenizer.decode (answer_ids, skip_special_tokens = True).strip ()


def main ():
    """
    Pipeline:

    1) Setup scratch + caches + seed + device
    2) Load SQuAD dataset (subset for fast training)
    3) Tokenize + align answer spans
    4) Load model
    5) Configure TrainingArguments (documented)
    6) Train + save model/tokenizer to /scratch
    7) Save a small report + a demo example and copy them to $HOME
    """

    # 1) Setup
    hf_home, scratch_base = setup_hf_caches ()
    set_seed (42)
    device = get_device ()

    print ("[INFO] hostname:", os.uname ().nodename)
    print ("[INFO] scratch_base:", scratch_base)
    print ("[INFO] HF_HOME:", os.environ.get ("HF_HOME"))
    
    
    # Experiment directories in scratch
    exp_dir = ensure_dir (scratch_base / "out" / "qa_squad_bert_uncased")
    model_dir = ensure_dir (exp_dir / "model_finetuned")
    report_dir = ensure_dir (exp_dir / "reports")
    
    
    # 2) Load dataset (cached to scratch)
    ds = load_dataset ("squad", cache_dir = str (hf_home / "datasets"))
    
    
    # Use a subset to keep the practice fast and stable
    train_ds = ds["train"].select (range (3000))
    val_ds = ds["validation"].select (range (1000))
    
    
    # 3) Tokenizer + preprocessing
    model_name = "google-bert/bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained (
        model_name,
        cache_dir = str (hf_home / "transformers"),
    )

    train_ds = train_ds.map (
        preprocess_function,
        batched = True,
        fn_kwargs = {"tokenizer": tokenizer},
        remove_columns = train_ds.column_names,
    )

    val_ds = val_ds.map (
        preprocess_function,
        batched = True,
        fn_kwargs = {"tokenizer": tokenizer},
        remove_columns = val_ds.column_names,
    )

    # 4) Model
    model = AutoModelForQuestionAnswering.from_pretrained (
        model_name,
        cache_dir = str (hf_home / "transformers"),
    ).to (device)

    data_collator = DefaultDataCollator ()
    
    
    # 5) TrainingArguments
    #
    # Note (important):
    # QA official metrics (Exact Match / F1) require additional post-processing.
    training_args = TrainingArguments (
        output_dir = str (exp_dir / "trainer_runs"),
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit = 1,
        load_best_model_at_end = False,

        learning_rate = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 2,

        weight_decay = 0.01,
        seed = 42,
        fp16 = torch.cuda.is_available (),

        logging_steps = 50,
        report_to = [],
    )

    trainer = Trainer (
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )
    
    
    # 6) Train
    trainer.train ()
    trainer.save_model (str (model_dir))
    tokenizer.save_pretrained (str (model_dir))

    print (f"[OK] Model and tokenizer saved to: {model_dir}")
    
    
    # 7) Report (scratch + HOME)
    eval_metrics = trainer.evaluate (eval_dataset = val_ds)

    report_df = pd.DataFrame ([{
        "eval_loss": eval_metrics.get ("eval_loss", None),
        "eval_runtime": eval_metrics.get ("eval_runtime", None),
        "eval_samples_per_second": eval_metrics.get ("eval_samples_per_second", None),
        "eval_steps_per_second": eval_metrics.get ("eval_steps_per_second", None),
        "epochs": training_args.num_train_epochs,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "fp16": training_args.fp16,
        "model": model_name,
        "dataset": "squad",
        "train_subset": 3000,
        "val_subset": 1000,
    }])

    report_csv_scratch = report_dir / "qa_report.csv"
    report_df.to_csv (report_csv_scratch, index = False)

    home_reports = ensure_dir (Path.home () / "reports")
    report_csv_home = home_reports / "qa_squad_report.csv"
    report_df.to_csv (report_csv_home, index = False)
    
    
    # Demo: run QA on a real example
    demo_question = "Who wrote the novel 1984?"
    demo_context = (
        "George Orwell was an English novelist and journalist. "
        "He wrote the novel 1984, which depicts a dystopian future."
    )

    demo_answer = predict_answer (
        model = model,
        tokenizer = tokenizer,
        question = demo_question,
        context = demo_context,
        device = device
    )

    demo_txt_scratch = report_dir / "qa_demo.txt"
    demo_txt_scratch.write_text (
        "QA Demo (Extractive)\n"
        "--------------------\n"
        f"Question: {demo_question}\n"
        f"Context: {demo_context}\n"
        f"Predicted answer: {demo_answer}\n",
        encoding = "utf-8"
    )

    demo_txt_home = home_reports / "qa_demo.txt"
    demo_txt_home.write_text (demo_txt_scratch.read_text (encoding = "utf-8"), encoding = "utf-8")

    print (f"[OK] Report saved to scratch: {report_csv_scratch}")
    print (f"[OK] Report copied to HOME:  {report_csv_home}")
    print (f"[OK] Demo saved to HOME:     {demo_txt_home}")


if __name__ == "__main__":
    main ()
