import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import pandas as pd
import os
import json
import random
from tqdm import tqdm
from morphology import apply_modifier
from tokenizer import tokenize
from generate import FinalLogitsProcessor, get_expanded_vocab

# 1. Setup Data Paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RAW_OUTPUT = os.path.join(DATA_DIR, "raw_definitions.jsonl")
PRIMITIVE_OUTPUT = os.path.join(DATA_DIR, "train_primitives.txt")

# 2. Load Complex Words
if not os.path.exists("complex_words.txt"):
    print("Error: complex_words.txt not found. Run extract_complex_words.py first.")
    exit()

with open("complex_words.txt", "r") as f:
    COMPLEX_WORDS = [line.strip() for line in f if line.strip()]

# 3. Initialize Model and Constraints
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use optimized expansion from generate.py
ALLOWED_SURFACE = get_expanded_vocab()
GLOBAL_PROCESSOR = FinalLogitsProcessor(tokenizer, ALLOWED_SURFACE)

# 4. Generation Logic
def generate_definition(word):
    # UpGoer5 style prompt
    prompt = f"Define '{word.upper()}' in the style of UpGoer5 using only the 1000 most simple words: A {word} is a "
    
    # Use optimized mask from GLOBAL_PROCESSOR
    GLOBAL_PROCESSOR.active_allowed_ids = GLOBAL_PROCESSOR.get_mask_for_excluded_word(word)
    processors = LogitsProcessorList([GLOBAL_PROCESSOR])
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Using Beam Search for higher quality
    output = model.generate(
        input_ids,
        max_length=80,
        logits_processor=processors,
        num_beams=3,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract part after " is a "
    if " is a " in full_text:
        definition = full_text.split(" is a ", 1)[1].strip()
    else:
        definition = full_text
    return definition

# 5. Execution Loop
def run_batch(count=100):
    print(f"Starting generation of {count} definitions...")
    targets = random.sample(COMPLEX_WORDS, min(count, len(COMPLEX_WORDS)))
    
    with open(RAW_OUTPUT, "a", encoding="utf-8") as raw_f, \
         open(PRIMITIVE_OUTPUT, "a", encoding="utf-8") as prim_f:
        
        for i, word in enumerate(tqdm(targets)):
            try:
                raw_def = generate_definition(word)
                raw_f.write(json.dumps({"id": i, "word": word, "definition": raw_def}) + "\n")
                
                # Tokenize to Primitives
                prim_tokens = tokenize(raw_def)
                prim_f.write(prim_tokens + "\n")
                
                if (i + 1) % 5 == 0:
                    raw_f.flush()
                    prim_f.flush()
            except Exception as e:
                print(f"Error on word '{word}': {e}")

if __name__ == "__main__":
    run_batch(50)
    print(f"\nBatch complete. Check {DATA_DIR} for results.")
