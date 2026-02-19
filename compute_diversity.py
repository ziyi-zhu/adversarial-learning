#!/usr/bin/env python
"""
Compute standard diversity metrics on user utterances from experiment results.

Metrics:
  - TTR (Type-Token Ratio): unique words / total words
  - Distinct-1: unique unigrams / total unigrams
  - Distinct-2: unique bigrams / total bigrams
  - Distinct-3: unique trigrams / total trigrams
  - Word Entropy (unigram Shannon entropy)
  - Mean Utterance Length (tokens)
"""

import json, os, math, sys
from collections import Counter
from pathlib import Path

RESULTS_DIR = "experiment_results/test10"

# For rule_us_rule_sys, user text is in "user_nl"; for LLM combos it's in "user" field of conversation.
# For TUS/GenTUS, user text is in "user_nl".

def extract_user_utterances(results_path):
    with open(results_path) as f:
        data = json.load(f)

    utterances = []
    for dialog in data["per_dialogue"]:
        for turn in dialog["conversation"]:
            text = turn.get("user_nl") or turn.get("user") or ""
            text = str(text).strip()
            if text and text not in ("[]", ""):
                text = text.replace("[END]", "").replace("**", "").strip()
                if text:
                    utterances.append(text)
    return utterances


def tokenize(text):
    return text.lower().split()


def compute_distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_entropy(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_metrics(utterances):
    all_tokens = []
    lengths = []
    for utt in utterances:
        toks = tokenize(utt)
        all_tokens.extend(toks)
        lengths.append(len(toks))

    n_total = len(all_tokens)
    n_unique = len(set(all_tokens))

    return {
        "n_utterances": len(utterances),
        "n_tokens": n_total,
        "n_unique_tokens": n_unique,
        "ttr": n_unique / max(n_total, 1),
        "distinct_1": compute_distinct_n(all_tokens, 1),
        "distinct_2": compute_distinct_n(all_tokens, 2),
        "distinct_3": compute_distinct_n(all_tokens, 3),
        "entropy": compute_entropy(all_tokens),
        "mean_utt_length": sum(lengths) / max(len(lengths), 1),
    }


def main():
    combos = [
        "rule_us_rule_sys",
        "tus_rule_sys",
        "gentus_rule_sys",
        "llm_us_llm_rg",
        "llm_us_rule_sys",
    ]

    all_metrics = {}
    for combo in combos:
        results_path = os.path.join(RESULTS_DIR, combo, "results.json")
        if not os.path.exists(results_path):
            print(f"  {combo}: results.json not found, skipping")
            continue

        utterances = extract_user_utterances(results_path)
        metrics = compute_metrics(utterances)
        all_metrics[combo] = metrics

    header = f"{'Combo':<22} {'Utts':>5} {'Tokens':>7} {'TTR':>6} {'D-1':>6} {'D-2':>6} {'D-3':>6} {'Entropy':>8} {'AvgLen':>7}"
    print(header)
    print("-" * len(header))
    for combo, m in all_metrics.items():
        print(f"{combo:<22} {m['n_utterances']:>5} {m['n_tokens']:>7} "
              f"{m['ttr']:>6.3f} {m['distinct_1']:>6.3f} {m['distinct_2']:>6.3f} "
              f"{m['distinct_3']:>6.3f} {m['entropy']:>8.3f} {m['mean_utt_length']:>7.1f}")

    out_path = os.path.join(RESULTS_DIR, "diversity_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
