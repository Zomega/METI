import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import pandas as pd
import re
from morphology import apply_modifier

# 1. Expand Vocabulary


def get_expanded_vocab():
    df = pd.read_csv("vocab.csv")
    core_words = sorted(list(df["word"].dropna().str.lower().unique()))
    modifiers = ["[PL]", "[ER]", "[ING]", "[ED]", "[LY]", "[CMP]", "[SUP]"]
    expanded = set(core_words)
    # Essential glue words and pronouns
    expanded.update(["i", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                     "a", "the", "this", "that", "all", "some", "and", "but", "or", "because",
                     "am", "is", "are", "was", "were", "be", "of", "to", "in", "on", "at", "for", "with", "by", "not"])
    for word in core_words:
        for mod in modifiers:
            modified = apply_modifier(word, mod)
            if modified and modified != word:
                expanded.add(modified)
    return sorted(list(expanded))


ALLOWED_WORDS = get_expanded_vocab()

# 2. Load Modern Model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Strict Logits Processor


class FinalLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_words):
        self.tokenizer = tokenizer
        self.allowed_ids = set()
        # Modern tokenizers (Llama) handle spaces differently (underscore char)
        # We need to find every token ID that matches our words exactly.
        for w in allowed_words:
            for var in [w.lower(), w.capitalize(), w.upper()]:
                # Try with and without leading space
                for prefix in [" ", ""]:
                    ids = tokenizer.encode(
                        prefix + var, add_special_tokens=False)
                    # ONLY allow if it's a single token (Strict Unit)
                    if len(ids) == 1:
                        self.allowed_ids.add(ids[0])

        # Allow punctuation
        for char in [".", ",", "!", "?", ";", ":", "\n", "(", ")", "-", '"', "'"]:
            ids = tokenizer.encode(char, add_special_tokens=False)
            self.allowed_ids.update(ids)
            ids_sp = tokenizer.encode(" " + char, add_special_tokens=False)
            self.allowed_ids.update(ids_sp)

        print(
            f"Strict Vocabulary: {len(self.allowed_ids)} valid token units allowed.")

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        valid = [idx for idx in self.allowed_ids if idx < scores.shape[-1]]
        mask[:, valid] = 0
        return scores + mask


def generate(prompt_text, processors):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=100,
        logits_processor=processors,
        num_beams=5,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# 4. Test multiple words
test_words = ["satellite", "microscope", "electricity", "democracy"]

for target_word in test_words:
    prompt = f"Define {target_word.upper()} in the style of UpGoer5 using only the most simple words: A {target_word} is "
    print(f"\n" + "="*50)
    print(f"STRICT DEFINITION FOR: '{target_word.upper()}'")
    print("="*50)

    # Re-initialize processor without the target word
    strict_allowed = [
        w for w in ALLOWED_WORDS if target_word.lower() not in w.lower()]
    processor = FinalLogitsProcessor(tokenizer, strict_allowed)
    processors = LogitsProcessorList([processor])

    result = generate(prompt, processors)
    print(f"\nRaw Result: {result}")

    if " is " in result:
        definition_part = result.split(" is ", 1)[1].strip()
        from tokenizer import tokenize, untokenize
        primitives = tokenize(definition_part)
        print("\n--- PRIMITIVE ANALYSIS ---")
        print(f"Primitives: {primitives}")
        print(f"Cleaned:    {untokenize(primitives)}")
