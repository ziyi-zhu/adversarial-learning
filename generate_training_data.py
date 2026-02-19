#!/usr/bin/env python
"""
Generate SFT training data for LLM user simulator from MultiWOZ 2.1 train split.

One row per dialogue.  The `messages` field is a complete conversation in OpenAI
chat format: system message (goal + instructions), then alternating user / assistant
turns.  "user" = dialogue-system utterances, "assistant" = user-simulator utterances
(the training target).  Train with causal LM loss masked to assistant tokens only.

Reuses LLM_US.DEFAULT_SYSTEM_INSTRUCTION and its init_session formatting logic so
the training distribution matches inference exactly.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ConvLab-3"))

from convlab.util.unified_datasets_util import load_dataset as load_convlab_dataset
from convlab.base_models.llm.user_simulator import LLM_US
from datasets import Dataset

INITIAL_SYSTEM_GREETING = "Hello, how may I help you today?"
HF_REPO = "slingshot/multiwoz-2.1-convlab"


def _repair_description(description):
    """Fix descriptions corrupted by preprocess.py joining a string (not a list)
    with '. ', which inserts '. ' between every character.  Detected when >80%
    of the '. '-split parts are single characters."""
    parts = description.split('. ')
    if len(parts) > 10 and sum(len(p) <= 2 for p in parts) / len(parts) > 0.8:
        return ''.join(parts)
    return description


def build_system_prompt(goal):
    """Replicate the system prompt construction from LLM_US.init_session,
    with a repair pass for corrupted descriptions."""
    description = _repair_description(goal.get("description", "") or "")
    goal_description = '.\n'.join(
        ['* ' + item for item in description.split('. ')]
    )
    return f"Goal:\n{goal_description}\n\n{LLM_US.DEFAULT_SYSTEM_INSTRUCTION}"


def process_dialogue(dialogue):
    """Convert one MultiWOZ dialogue into a single row with full message list."""
    goal = dialogue["goal"]
    turns = dialogue["turns"]
    dialogue_id = dialogue["dialogue_id"]
    domains = dialogue.get("domains", [])

    system_prompt = build_system_prompt(goal)

    messages = [{"role": "system", "content": system_prompt}]
    first_user = True

    for turn in turns:
        if turn["speaker"] == "user":
            if first_user:
                messages.append({"role": "user", "content": INITIAL_SYSTEM_GREETING})
                first_user = False
            messages.append({"role": "assistant", "content": turn["utterance"]})

        elif turn["speaker"] == "system":
            messages.append({"role": "user", "content": turn["utterance"]})

    if messages[-1]["role"] == "user" and len(messages) > 1:
        messages.pop()

    n_user_turns = sum(1 for m in messages if m["role"] == "assistant")

    return {
        "dialogue_id": dialogue_id,
        "domains": domains,
        "n_turns": n_user_turns,
        "messages": messages,
    }


def sanity_check(rows):
    """Verify every dialogue follows the role pattern LLM.chat() expects:
    [system, user, assistant, user, assistant, ...].  The last message should
    be assistant (the final user-simulator utterance)."""
    for row in rows[:500]:
        msgs = row["messages"]
        assert msgs[0]["role"] == "system", f"{row['dialogue_id']}: first msg not system"
        assert msgs[-1]["role"] == "assistant", f"{row['dialogue_id']}: last msg not assistant"
        for i, m in enumerate(msgs[1:], 1):
            expected = "user" if i % 2 == 1 else "assistant"
            assert m["role"] == expected, (
                f"{row['dialogue_id']}: index {i} expected {expected}, got {m['role']}"
            )
    print(f"Sanity check passed on {min(500, len(rows))} dialogues.")


def main():
    print("Loading MultiWOZ 2.1 ...")
    dataset = load_convlab_dataset("multiwoz21")
    train_dialogues = dataset["train"]
    print(f"  {len(train_dialogues)} train dialogues")

    all_rows = []
    skipped = 0
    for dlg in train_dialogues:
        row = process_dialogue(dlg)
        if row["n_turns"] == 0:
            skipped += 1
            continue
        all_rows.append(row)

    total_turns = sum(r["n_turns"] for r in all_rows)
    print(f"  {len(all_rows)} dialogues, {total_turns} assistant turns total "
          f"({skipped} skipped)")
    sanity_check(all_rows)

    print("Building HuggingFace dataset ...")
    hf_dataset = Dataset.from_list(all_rows)
    print(hf_dataset)
    print("\nExample row (first dialogue):")
    ex = hf_dataset[0]
    print(f"  dialogue_id: {ex['dialogue_id']}")
    print(f"  domains:     {ex['domains']}")
    print(f"  n_turns:     {ex['n_turns']}")
    print(f"  messages ({len(ex['messages'])} total):")
    for m in ex["messages"][:6]:
        content_preview = m["content"][:80] + ("..." if len(m["content"]) > 80 else "")
        print(f"    [{m['role']:>9}] {content_preview}")
    if len(ex["messages"]) > 6:
        print(f"    ... {len(ex['messages']) - 6} more messages")

    print(f"\nPushing to {HF_REPO} ...")
    hf_dataset.push_to_hub(HF_REPO)
    print("Done.")


if __name__ == "__main__":
    main()
