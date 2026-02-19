#!/usr/bin/env python
"""
Evaluate multiple user simulator + system agent combinations on MultiWOZ 2.1.

Combinations:
  1. semantic_rule_us_rule_sys      -- Rule US (DA) + Rule Sys (DA)
  2. nl_rule_us_llmnlu_rule_sys     -- Rule US (TemplateNLG) + LLM_NLU on both sides
  3. llm_us_llm_rg                  -- LLM_US + LLM_RG end-to-end
  4. llm_us_rule_sys                -- LLM_US user + Rule pipeline system (text level)
"""

import os
import sys
import json
import random
import traceback
import numpy as np
from copy import deepcopy
from datetime import datetime

SEED = 20200202
LLM_MODEL = "openrouter/google/gemini-2.0-flash-001"
RESULTS_DIR = "experiment_results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Compatibility DST that bridges ConvLab-3 unified format to the old
# semi/book state format expected by RuleBasedMultiwozBot.
# Also remaps user_action slot names from new -> old display format.
# ---------------------------------------------------------------------------
from convlab.dst.dst import DST
from convlab.util.multiwoz.multiwoz_slot_trans import REF_USR_DA

SLOT_NEW_TO_OLD = {
    "price range": "pricerange",
    "leave at": "leaveAt",
    "arrive by": "arriveBy",
    "book day": "day",
    "book people": "people",
    "book stay": "stay",
    "book time": "time",
}

BOOK_SLOTS = {"day", "people", "stay", "time"}

# ConvLab-3 slot name -> display name expected by the system bot in user_action
_SLOT_DISPLAY = {}
for _dom, _mapping in REF_USR_DA.items():
    for _old, _disp in _mapping.items():
        _SLOT_DISPLAY[(_dom.lower(), _old)] = _disp

def _to_display_slot(domain, new_slot):
    """Convert a ConvLab-3 slot name to the display slot name the system bot expects."""
    old = SLOT_NEW_TO_OLD.get(new_slot, new_slot)
    return _SLOT_DISPLAY.get((domain.lower(), old), new_slot)


class CompatRuleDST(DST):
    """Thin DST that provides old-format state and remaps user_action slots."""

    def __init__(self):
        DST.__init__(self)
        self._init_state()

    def _init_state(self):
        from convlab.util.multiwoz.state import default_state_old
        self.state = default_state_old()
        self.state.setdefault("booked", {})

    def init_session(self):
        self._init_state()

    def update(self, user_act=None):
        if not user_act:
            return self.state

        remapped = []
        for intent, domain, slot, value in user_act:
            dom = domain.lower()
            old_slot = SLOT_NEW_TO_OLD.get(slot, slot)
            display_slot = _to_display_slot(dom, slot)

            if intent == "inform" and dom in self.state["belief_state"]:
                if old_slot not in ("none", ""):
                    if old_slot in BOOK_SLOTS and "book" in self.state["belief_state"][dom]:
                        if old_slot in self.state["belief_state"][dom]["book"]:
                            self.state["belief_state"][dom]["book"][old_slot] = value
                    elif "semi" in self.state["belief_state"][dom]:
                        if old_slot in self.state["belief_state"][dom]["semi"]:
                            self.state["belief_state"][dom]["semi"][old_slot] = value
            elif intent == "request":
                if dom not in self.state["request_state"]:
                    self.state["request_state"][dom] = {}
                if old_slot not in self.state["request_state"][dom]:
                    self.state["request_state"][dom][old_slot] = 0

            remapped.append([intent, domain, display_slot, value])

        self.state["user_action"] = remapped
        return self.state


# ---------------------------------------------------------------------------
# Combo 1 – pure semantic (DA-level) rule-based baseline
# ---------------------------------------------------------------------------
def run_semantic_rule(n_dialogues=100):
    from convlab.policy.rule.multiwoz import RulePolicy
    from convlab.dialog_agent import PipelineAgent
    from convlab.util.analysis_tool.analyzer import Analyzer

    sys_agent = PipelineAgent(None, CompatRuleDST(), RulePolicy(), None, name="sys")
    user_agent = PipelineAgent(None, None, RulePolicy(character="usr"), None, name="user")

    analyzer = Analyzer(user_agent=user_agent, dataset="multiwoz")
    set_seed(SEED)
    return analyzer.comprehensive_analyze(
        sys_agent=sys_agent,
        model_name="semantic_rule_us_rule_sys",
        total_dialog=n_dialogues,
    )


# ---------------------------------------------------------------------------
# Combo 2 – text-level with LLM NLU on both sides + TemplateNLG
# ---------------------------------------------------------------------------
def run_nl_rule_llmnlu(n_dialogues=20):
    from convlab.policy.rule.multiwoz import RulePolicy
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.base_models.llm.nlu import LLM_NLU
    from convlab.dialog_agent import PipelineAgent
    from convlab.util.analysis_tool.analyzer import Analyzer
    from convlab.util.unified_datasets_util import load_dataset

    dataset = load_dataset("multiwoz21")
    example_dialogs = dataset["train"][:3]

    user_nlu = LLM_NLU("multiwoz21", "litellm", LLM_MODEL, "system", example_dialogs)
    user_agent = PipelineAgent(
        user_nlu, None, RulePolicy(character="usr"), TemplateNLG(is_user=True), name="user"
    )

    sys_nlu = LLM_NLU("multiwoz21", "litellm", LLM_MODEL, "user", example_dialogs)
    sys_agent = PipelineAgent(
        sys_nlu, CompatRuleDST(), RulePolicy(), TemplateNLG(is_user=False), name="sys"
    )

    analyzer = Analyzer(user_agent=user_agent, dataset="multiwoz")
    set_seed(SEED)
    return analyzer.comprehensive_analyze(
        sys_agent=sys_agent,
        model_name="nl_rule_us_llmnlu_rule_sys",
        total_dialog=n_dialogues,
    )


# ---------------------------------------------------------------------------
# Combo 3 – fully LLM: LLM_US + LLM_RG (custom eval loop)
# ---------------------------------------------------------------------------
def run_llm_us_llm_rg(n_dialogues=20):
    from convlab.base_models.llm.user_simulator import LLM_US, LLM_RG
    from convlab.util.unified_datasets_util import load_dataset

    dataset = load_dataset("multiwoz21")
    goals = [d["goal"] for d in dataset["validation"][:n_dialogues]]

    user_model = LLM_US("litellm", LLM_MODEL)
    system_model = LLM_RG("litellm", LLM_MODEL)

    results = []
    for i, goal in enumerate(goals):
        print(f"\n{'='*50}\nDialogue {i+1}/{len(goals)}")
        try:
            user_model.init_session(goal)
            system_model.init_session()

            sys_msg = "Hello, how may I help you today?"
            conversation = [{"role": "system", "content": sys_msg}]
            turns = 0

            for _ in range(20):
                user_msg = user_model.response(sys_msg)
                conversation.append({"role": "user", "content": user_msg})

                if user_model.is_terminated:
                    break

                sys_msg = system_model.response(user_msg)
                conversation.append({"role": "assistant", "content": sys_msg})
                turns += 1

            completed = user_model.is_terminated and any(
                "[END]" in c.get("content", "") for c in conversation if c["role"] == "user"
            )
            reward = user_model.get_reward()
        except Exception as e:
            print(f"  Error in dialogue {i}: {e}")
            completed = False
            reward = None
            conversation = []
            turns = 0

        results.append({
            "dialogue_id": i,
            "goal_description": goal.get("description", ""),
            "turns": turns,
            "completed": completed,
            "reward": reward,
            "conversation": conversation,
        })
        print(f"  Turns: {turns}, Completed: {completed}, Reward: {reward}")

    out_dir = os.path.join(RESULTS_DIR, "llm_us_llm_rg")
    os.makedirs(out_dir, exist_ok=True)

    completed_count = sum(1 for r in results if r["completed"])
    rewards = [r["reward"] for r in results if r["reward"] is not None]
    summary = {
        "model": LLM_MODEL,
        "total_dialogs": len(results),
        "completed_rate": completed_count / max(len(results), 1),
        "avg_turns": float(np.mean([r["turns"] for r in results])),
        "avg_reward": float(np.mean(rewards)) if rewards else None,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "dialogs": results}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"\nLLM_US + LLM_RG summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


# ---------------------------------------------------------------------------
# Combo 4 – LLM_US user + rule-based pipeline system (text level)
# ---------------------------------------------------------------------------
def run_llm_us_rule_sys(n_dialogues=20):
    from convlab.policy.rule.multiwoz import RulePolicy
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.dialog_agent import PipelineAgent
    from convlab.base_models.llm.nlu import LLM_NLU
    from convlab.base_models.llm.user_simulator import LLM_US
    from convlab.util.unified_datasets_util import load_dataset

    dataset = load_dataset("multiwoz21")
    example_dialogs = dataset["train"][:3]
    goals = [d["goal"] for d in dataset["validation"][:n_dialogues]]

    sys_nlu = LLM_NLU("multiwoz21", "litellm", LLM_MODEL, "user", example_dialogs)
    sys_agent = PipelineAgent(
        sys_nlu, CompatRuleDST(), RulePolicy(), TemplateNLG(is_user=False), name="sys"
    )

    user_model = LLM_US("litellm", LLM_MODEL)

    results = []
    for i, goal in enumerate(goals):
        print(f"\n{'='*50}\nDialogue {i+1}/{len(goals)}")
        try:
            user_model.init_session(goal)
            sys_agent.init_session()

            sys_msg = "Hello, how may I help you today?"
            conversation = [{"role": "system", "content": sys_msg}]
            turns = 0

            for _ in range(20):
                user_msg = user_model.response(sys_msg)
                conversation.append({"role": "user", "content": user_msg})

                if user_model.is_terminated:
                    break

                sys_msg = sys_agent.response(user_msg)
                conversation.append({"role": "assistant", "content": sys_msg})
                turns += 1

            completed = user_model.is_terminated and any(
                "[END]" in c.get("content", "") for c in conversation if c["role"] == "user"
            )
            reward = user_model.get_reward()
        except Exception as e:
            print(f"  Error in dialogue {i}: {e}")
            traceback.print_exc()
            completed = False
            reward = None
            conversation = []
            turns = 0

        results.append({
            "dialogue_id": i,
            "goal_description": goal.get("description", ""),
            "turns": turns,
            "completed": completed,
            "reward": reward,
            "conversation": conversation,
        })
        print(f"  Turns: {turns}, Completed: {completed}, Reward: {reward}")

    out_dir = os.path.join(RESULTS_DIR, "llm_us_rule_sys")
    os.makedirs(out_dir, exist_ok=True)

    completed_count = sum(1 for r in results if r["completed"])
    rewards = [r["reward"] for r in results if r["reward"] is not None]
    summary = {
        "model": LLM_MODEL,
        "total_dialogs": len(results),
        "completed_rate": completed_count / max(len(results), 1),
        "avg_turns": float(np.mean([r["turns"] for r in results])),
        "avg_reward": float(np.mean(rewards)) if rewards else None,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "dialogs": results}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"\nLLM_US + Rule Sys summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
COMBOS = [
    ("semantic_rule_us_rule_sys", run_semantic_rule, {"n_dialogues": 100}),
    ("llm_us_llm_rg", run_llm_us_llm_rg, {"n_dialogues": 20}),
    ("llm_us_rule_sys", run_llm_us_rule_sys, {"n_dialogues": 20}),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for name, func, kwargs in COMBOS:
        print(f"\n{'#'*80}")
        print(f"# Running: {name}")
        print(f"{'#'*80}\n")
        try:
            result = func(**kwargs)
            if isinstance(result, tuple):
                labels = [
                    "complete_rate", "success_rate", "precision",
                    "recall", "f1", "book_rate", "avg_turn",
                ]
                all_results[name] = dict(zip(labels, [float(v) for v in result]))
            else:
                all_results[name] = result
            print(f"\nSUCCESS: {name}")
        except Exception:
            traceback.print_exc()
            all_results[name] = {"error": traceback.format_exc()}
            print(f"\nFAILED: {name} -- skipping")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(RESULTS_DIR, f"all_results_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    for name, result in all_results.items():
        print(f"\n  {name}:")
        if isinstance(result, dict):
            for k, v in result.items():
                if k == "error":
                    print(f"    ERROR: {v[:200]}")
                elif isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
