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
  - Mean Conversation Length (user turns per dialogue)
"""

import json, os, math, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ConvLab-3"))

RESULTS_DIR = "experiment_results"

def extract_user_utterances(results_path):
    with open(results_path) as f:
        data = json.load(f)

    utterances = []
    conversation_lengths = []
    for dialog in data["per_dialogue"]:
        n_user = 0
        for turn in dialog["conversation"]:
            text = turn.get("user_nl") or turn.get("user") or ""
            text = str(text).strip()
            if text and text not in ("[]", ""):
                text = text.replace("[END]", "").replace("**", "").strip()
                if text:
                    utterances.append(text)
                    n_user += 1
        conversation_lengths.append(n_user)
    return utterances, conversation_lengths


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


def compute_metrics(utterances, conversation_lengths=None):
    all_tokens = []
    lengths = []
    for utt in utterances:
        toks = tokenize(utt)
        all_tokens.extend(toks)
        lengths.append(len(toks))

    n_total = len(all_tokens)
    n_unique = len(set(all_tokens))

    out = {
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
    if conversation_lengths:
        out["mean_conversation_length"] = sum(conversation_lengths) / max(len(conversation_lengths), 1)
    else:
        out["mean_conversation_length"] = None
    return out


def extract_ground_truth_utterances(n_dialogues=None):
    """Extract user utterances from the MultiWOZ 2.1 test split."""
    from convlab.util.unified_datasets_util import load_dataset
    dataset = load_dataset("multiwoz21")
    test_dialogues = dataset["test"]
    if n_dialogues is not None:
        test_dialogues = test_dialogues[:n_dialogues]

    utterances = []
    conversation_lengths = []
    for dlg in test_dialogues:
        n_user = 0
        for turn in dlg["turns"]:
            if turn["speaker"] == "user":
                text = turn["utterance"].strip()
                if text:
                    utterances.append(text)
                    n_user += 1
        conversation_lengths.append(n_user)
    return utterances, conversation_lengths


def main():
    all_metrics = {}

    # Detect how many dialogues were used in baselines (from any results file)
    n_dialogues = None
    for entry in sorted(os.listdir(RESULTS_DIR)):
        results_path = os.path.join(RESULTS_DIR, entry, "results.json")
        if os.path.isfile(results_path):
            with open(results_path) as f:
                data = json.load(f)
            n_dialogues = data.get("summary", {}).get("n_dialogues")
            if n_dialogues:
                break

    gt_utterances, gt_conv_lengths = extract_ground_truth_utterances(n_dialogues)
    all_metrics["ground_truth"] = compute_metrics(gt_utterances, gt_conv_lengths)

    for entry in sorted(os.listdir(RESULTS_DIR)):
        results_path = os.path.join(RESULTS_DIR, entry, "results.json")
        if not os.path.isfile(results_path):
            continue
        try:
            utterances, conv_lengths = extract_user_utterances(results_path)
        except (KeyError, json.JSONDecodeError):
            print(f"  {entry}: could not parse results.json, skipping")
            continue
        if not utterances:
            print(f"  {entry}: no user utterances found, skipping")
            continue
        metrics = compute_metrics(utterances, conv_lengths)
        all_metrics[entry] = metrics

    header = f"{'Combo':<22} {'Utts':>5} {'Tokens':>7} {'TTR':>6} {'D-1':>6} {'D-2':>6} {'D-3':>6} {'Entropy':>8} {'AvgLenUttr':>9} {'AvgLenConv':>9}"
    print(header)
    print("-" * len(header))
    for combo, m in all_metrics.items():
        avg_conv = m["mean_conversation_length"]
        avg_conv_str = f"{avg_conv:>9.1f}" if avg_conv is not None else "      n/a"
        print(f"{combo:<22} {m['n_utterances']:>5} {m['n_tokens']:>7} "
              f"{m['ttr']:>6.3f} {m['distinct_1']:>6.3f} {m['distinct_2']:>6.3f} "
              f"{m['distinct_3']:>6.3f} {m['entropy']:>8.3f} {m['mean_utt_length']:>9.1f} {avg_conv_str}")

    out_path = os.path.join(RESULTS_DIR, "diversity_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
