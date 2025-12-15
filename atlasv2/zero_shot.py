"""
DL-PLN 2025/26 – ATLAS v2

Zero-shot sentiment classification with an instruction-tuned LLM (Gemma).

This script shows how to use a Large Language Model (LLM) as a *zero-shot classifier*:
- We provide an instruction (the prompt) describing the labels (Positive/Negative)
- We ask the model to output ONLY one label

Decoding parameters (important):
- temperature controls randomness:
    * 0.0 => deterministic (recommended for classification)
    * higher => more diverse, but less stable labels
- top_p (nucleus sampling) controls how many tokens are considered:
    * 1.0 => no restriction
    * 0.9 => sample from a smaller set of likely tokens
For classification we usually use:
    do_sample = False, temperature = 0.0, top_p = 1.0

How to get a Hugging Face token (HF_TOKEN):
1) Create / log into your account on Hugging Face.
2) Go to Settings -> Access Tokens.
3) Create a token (read access is usually enough for downloading models).
4) Put the token in your bootstrap.sh file

@author Ronghao Pan <ronghao.pan@um.es>
@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-García <valencia@um.es>
"""

import os
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from atlas_utils import (
    ensure_dir,
    setup_hf_caches,
    set_seed,
    get_device,
)

from prompting_utils import (
    build_chat_prompt,
    extract_generated_text,
    normalize_binary_label,
)

# -------------------------
# Prompt
# -------------------------
MODEL_PROMPT = """You are a classifier that determines the sentiment of a given text.
Return ONLY one label, with no explanation:

Positive
Negative

Definition:
Positive: The text expresses satisfaction, happiness, approval, or a favorable opinion.
Negative: The text expresses dissatisfaction, frustration, criticism, or an unfavorable opinion.

Text:
{text}
"""

def main ():
    """
    Main execution pipeline for zero-shot sentiment classification.

    Even though we do NOT use Hugging Face's `pipeline()` abstraction,
    the logic is conceptually the same:
    
    > from transformers import pipeline
    > clf = pipeline(
    >    "text-generation",
    >    model = model_id,
    >    tokenizer = model_id,
    >    device_map = "auto"
    > )
    > clf(MODEL_PROMPT.format(text=message), max_new_tokens=6)
    
    This is the explicit implementation, so you can see each step clearly.
        1) Load model + tokenizer
        2) Build the prompt
        3) Tokenize inputs
        4) Generate model output
        5) Post-process the output (label extraction)
        6) Save results
    """
    
    # Setup
    hf_home, scratch_base = setup_hf_caches ()
    set_seed (42)
    device = get_device ()
    
    
    # Folders
    exp_dir = ensure_dir (scratch_base / "out" / "zsl_gemma_sentiment")
    report_dir = ensure_dir (exp_dir / "reports")
    
    
    # HF token
    token = os.getenv ("HF_TOKEN", None)
    if token is None or token.strip () == "":
        print ("[WARN] HF_TOKEN is not set. Model download may fail.")
        print ("       Export it before running: export HF_TOKEN='hf_...'")
    
    
     # Optional 4-bit quantization (recommended on Tesla T4 GPUs)
    quant_config = None
    if torch.cuda.is_available ():
        quant_config = BitsAndBytesConfig (
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
        )
        
        
    # Load model and tokenizer
    model_id = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained (
        model_id,
        token = token,
        cache_dir = str (hf_home / "transformers"),
    )

    model = AutoModelForCausalLM.from_pretrained (
        model_id,
        token = token,
        cache_dir = str (hf_home / "transformers"),
        device_map = "auto" if torch.cuda.is_available () else None,
        quantization_config = quant_config,
        torch_dtype = torch.float16 if torch.cuda.is_available () else None,
    )
    
    
    # Messages to evaluate
    messages = [
        "El servicio fue excelente y el personal muy amable.",
        "La experiencia fue horrible y no pienso volver.",
        "El producto cumple su función, sin más.",
        "Estoy muy satisfecho con el resultado final.",
        "Fue una pérdida de tiempo y dinero.",
        "La atención al cliente fue rápida y eficaz.",
        "Nada funcionaba como prometían.",
        "Una experiencia aceptable, aunque mejorable.",
        "Superó mis expectativas.",
        "Me sentí frustrado durante todo el proceso."
    ]
    
    
    rows = []
    for index, message in enumerate (messages):
        
        # 5.1) Build the task prompt (instruction + text)
        user_text = MODEL_PROMPT.format (text = message)
        prompt = build_chat_prompt (tokenizer, user_text)
        
        
        # 5.2) Tokenize prompt
        inputs = tokenizer (prompt, return_tensors = "pt", add_special_tokens = False)
        if torch.cuda.is_available ():
            inputs = {k: v.to (model.device) for k, v in inputs.items ()}

    
        # Tokenize (do NOT force .to('cuda'); let model/device_map handle it)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if torch.cuda.is_available():
            # if model is sharded by device_map, moving inputs to model.device is safer than 'cuda:0'
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 5.3) Generate model output
        # We use deterministic decoding because this is classification.
        # Check params:
        # > max_new_tokens, to get only a small text for extracting the label
        # > do_sample for no no randomness
        # > temperature, for being fully deterministic
        # > top_p (this is irrelevant when do_sample is False)
        with torch.no_grad ():
            outputs = model.generate (
                **inputs,
                max_new_tokens = 6,
                do_sample = False,
                temperature = 0.0,
                top_p = 1.0,
                eos_token_id = tokenizer.eos_token_id,
            )

        # 5.4) Decode and extract only the generated text
        decoded = tokenizer.decode (outputs[0], skip_special_tokens = True)
        generated = extract_generated_text (tokenizer, inputs, decoded)

    
        # 5.5) Normalize model output to a clean label
        label = normalize_binary_label (
            generated,
            positive_label = "Positive",
            negative_label = "Negative"
        )

        # Attach
        rows.append ({
            "index": index,
            "text": message,
            "raw_output": generated,
            "label": label,
        })


    # Generate reports
    report_df = pd.DataFrame (rows)

    report_csv_scratch = report_dir / "zsl_outputs.csv"
    report_df.to_csv (report_csv_scratch, index = False)

    home_reports = ensure_dir (Path.home () / "reports")
    report_csv_home = home_reports / "zsl_outputs.csv"
    report_df.to_csv (report_csv_home, index = False)

    print (f"[OK] Report saved to scratch: {report_csv_scratch}")
    print (f"[OK] Report copied to HOME:  {report_csv_home}")    


    print ("JSON output")
    print ("-----------------------------------")
    print (json.dumps (rows, indent = 2, ensure_ascii = False))
    json_path = report_dir / "zsl_outputs.json"
    json_path.write_text (
        json.dumps (rows, indent = 2, ensure_ascii = False),
        encoding = "utf-8"
    )
        
    

if __name__ == "__main__":
    main()
