"""
DL-PLN 2025/26 – ATLAS v2

few-shot sentiment classification with an instruction-tuned LLM (Gemma).

This script shows how to use a Large Language Model (LLM) applying *chain-of-thought reasoning*:
- We provide an instruction (the prompt) describing the labels (Positive/Negative)
- We ask the model to output ONLY one label

Decoding parameters (important) (check zero_shot example)

How to get a Hugging Face token (HF_TOKEN): (check zero_shot example)

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
# check the prompt contains step-by-step reasoning.
MODEL_PROMPT = """
You are a sentiment classifier. You must analyze the sentiment of the given text.

Task:
Given the text input, think step-by-step about the overall sentiment.
Consider emotional tone, satisfaction/dissatisfaction, praise/complaint, and whether the message is overall favorable or unfavorable.
Then, make a final decision: "Positive" or "Negative".

Labels:
"Positive": The text expresses satisfaction, approval, happiness, gratitude, or a favorable opinion.
"Negative": The text expresses dissatisfaction, frustration, criticism, anger, disappointment, or an unfavorable opinion.

Examples:

Example 1:
Text: "El servicio fue excelente y el personal muy amable."
Step-by-step reasoning:
- The text contains strong praise ("excelente", "muy amable").
- The tone is clearly favorable and satisfied.
- There are no complaints or negative cues.
Classification: Positive

Example 2:
Text: "La experiencia fue horrible, llegó tarde y además estaba roto."
Step-by-step reasoning:
- The text contains explicit negative words ("horrible", "tarde", "roto").
- It describes problems and dissatisfaction.
- The overall tone is clearly unfavorable.
Classification: Negative

Now, classify the following text by thinking step-by-step:

Text: {text}
"""

def main ():
    """
    Main execution pipeline for few-shot sentiment classification with chain-of-thought reasoning.
    """
    
    # Setup
    hf_home, scratch_base = setup_hf_caches ()
    set_seed (42)
    device = get_device ()
    
    
    # Folders
    exp_dir = ensure_dir (scratch_base / "out" / "cot_gemma_sentiment")
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
        # 
        # IMPORTANT: 
        # Check we have increased max_new_tokens as we want to get the reasoning of the 
        # model, but for deterministic short generation in production environments we can add 
        # something like:
        # > Think step-by-step internally, but do NOT reveal your reasoning.
        # > Return ONLY the final label.
        # to the prompt:
        with torch.no_grad ():
            outputs = model.generate (
                **inputs,
                max_new_tokens = 128,
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

    report_csv_scratch = report_dir / "cot_outputs.csv"
    report_df.to_csv (report_csv_scratch, index = False)

    home_reports = ensure_dir (Path.home () / "reports")
    report_csv_home = home_reports / "cot_outputs.csv"
    report_df.to_csv (report_csv_home, index = False)

    print (f"[OK] Report saved to scratch: {report_csv_scratch}")
    print (f"[OK] Report copied to HOME:  {report_csv_home}")    


    print ("JSON output")
    print ("-----------------------------------")
    print (json.dumps (rows, indent = 2, ensure_ascii = False))
    json_path = report_dir / "cot_outputs.json"
    json_path.write_text (
        json.dumps (rows, indent = 2, ensure_ascii = False),
        encoding = "utf-8"
    )
    

if __name__ == "__main__":
    main()
