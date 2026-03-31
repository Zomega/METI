import pandas as pd
from morphology import apply_modifier, decompose_word
from tokenizer import tokenize, untokenize
from grammar import generate_random_tokens
import os


def check_loop(word, mod):
    # Apply modifier
    modified = apply_modifier(word, mod)
    if modified == word.lower():
        return "invalid", modified, word

    # Decompose modified word
    decomposed_list = decompose_word(modified)
    target = f"{word.upper()} {mod.upper()}"
    if mod == "[UN]":
        target = f"[UN] {word.upper()}"

    if target in decomposed_list:
        return "success", modified, str(decomposed_list)
    else:
        return "fail", modified, str(decomposed_list)


def run_morphology_tests():
    print("\n--- RUNNING MORPHOLOGY COMPLETENESS TESTS ---")
    common_words_path = "common_words.txt"
    if not os.path.exists(common_words_path):
        print("Error: common_words.txt not found.")
        return

    with open(common_words_path, "r") as f:
        common_words = [line.strip() for line in f if line.strip()]

    modifiers = ["[PL]", "[ER]", "[ING]",
                 "[ED]", "[LY]", "[UN]", "[CMP]", "[SUP]"]
    results = []

    for word in common_words:
        for mod in modifiers:
            status, modified, res = check_loop(word, mod)
            results.append({"word": word, "mod": mod, "status": status,
                           "modified": modified, "result": res})

    df = pd.DataFrame(results)
    total = len(df)
    successes = (df["status"] == "success").sum()
    invalids = (df["status"] == "invalid").sum()
    fails = (df["status"] == "fail").sum()

    print(f"Total Tests:        {total}")
    print(f"Valid Loops:        {successes} ({successes/total:.1%})")
    print(f"Invalid for POS:    {invalids} ({invalids/total:.1%})")
    print(f"Actual Failures:    {fails} ({fails/total:.1%})")
    if (total - invalids) > 0:
        print(f"Functional Accuracy: {successes / (total - invalids):.1%}")


def run_round_trip_tests(count=100):
    print(f"\n--- RUNNING {count} SENTENCE ROUND-TRIP TESTS ---")
    results = []
    for i in range(count):
        orig_tokens = generate_random_tokens()
        human_text = untokenize(orig_tokens)
        new_tokens = tokenize(human_text)
        final_human = untokenize(new_tokens)
        is_success = (human_text.lower() == final_human.lower())
        results.append({"success": is_success})

    df = pd.DataFrame(results)
    print(f"Success Rate: {df['success'].mean():.1%}")


if __name__ == "__main__":
    run_morphology_tests()
    run_round_trip_tests(100)
