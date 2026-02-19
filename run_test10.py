#!/usr/bin/env python
"""
Run 3 working combinations on 10 MultiWOZ test dialogues each.
Save per-dialogue conversations and aggregate metrics (success rate, completion rate).

Combinations:
  1. rule_us_rule_sys    -- Rule-based user simulator + Rule-based system (semantic)
  2. llm_us_llm_rg       -- LLM user simulator + LLM response generator
  3. llm_us_rule_sys      -- LLM user simulator + Rule-based pipeline system
"""

import os, sys, json, random, traceback, logging
import numpy as np
from copy import deepcopy
from datetime import datetime

# Suppress litellm / httpx noise before any imports that pull them in
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
import litellm
litellm.suppress_debug_info = True

SEED = 42
N_DIALOGUES = 10
LLM_MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct"
OUT_ROOT = "experiment_results/test10"

# -- reuse compat DST from run_baselines ---------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ConvLab-3"))
from run_baselines import CompatRuleDST, set_seed


def _load_test_goals():
    from convlab.util.unified_datasets_util import load_dataset
    dataset = load_dataset("multiwoz21")
    return [d["goal"] for d in dataset["test"][:N_DIALOGUES]]


def _save(out_dir, combo_name, results, extra_summary=None):
    os.makedirs(out_dir, exist_ok=True)

    completed_count = sum(1 for r in results if r["completed"])
    success_count = sum(1 for r in results if r.get("success", False))
    total = len(results)

    summary = {
        "combo": combo_name,
        "n_dialogues": total,
        "completion_rate": completed_count / max(total, 1),
        "success_rate": success_count / max(total, 1),
        "avg_turns": float(np.mean([r["turns"] for r in results])),
    }
    if extra_summary:
        summary.update(extra_summary)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "per_dialogue": results}, f, indent=2, default=str)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")

    print(f"\n  --- {combo_name} summary ---")
    for k, v in summary.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return summary


# =========================================================================
# 1. Rule US + Rule Sys (semantic level, with NL post-processing)
# =========================================================================
def run_rule_us_rule_sys():
    from convlab.policy.rule.multiwoz import RulePolicy
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.dialog_agent import PipelineAgent, BiSession
    from convlab.evaluator.multiwoz_eval import MultiWozEvaluator

    sys_agent = PipelineAgent(None, CompatRuleDST(), RulePolicy(), None, name="sys")
    user_agent = PipelineAgent(None, None, RulePolicy(character="usr"), None, name="user")
    evaluator = MultiWozEvaluator()

    user_nlg = TemplateNLG(is_user=True)
    sys_nlg = TemplateNLG(is_user=False)

    def _da_to_nl(da, nlg):
        try:
            return nlg.generate(da)
        except Exception:
            return str(da)

    set_seed(SEED)
    results = []
    for i in range(N_DIALOGUES):
        seed_i = random.randint(1, 100000)
        random.seed(seed_i)
        np.random.seed(seed_i)

        sess = BiSession(sys_agent=sys_agent, user_agent=user_agent,
                         kb_query=None, evaluator=evaluator)
        sess.init_session()

        sys_response = []
        conversation = []
        for t in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            conversation.append({
                "turn": t,
                "user_da": str(user_response),
                "user_nl": _da_to_nl(user_response, user_nlg),
                "system_da": str(sys_response),
                "system_nl": _da_to_nl(sys_response, sys_nlg),
            })
            if session_over:
                break

        task_success = sess.evaluator.task_success()
        task_complete = sess.evaluator.complete
        stats = sess.evaluator.inform_F1()

        results.append({
            "dialogue_id": i,
            "seed": seed_i,
            "turns": len(conversation),
            "completed": bool(task_complete),
            "success": bool(task_success),
            "inform_f1": stats[2],
            "book_rate": sess.evaluator.book_rate(),
            "conversation": conversation,
        })
        print(f"  Dialog {i+1}/{N_DIALOGUES}: turns={len(conversation)} "
              f"complete={task_complete} success={task_success}")

    return _save(os.path.join(OUT_ROOT, "rule_us_rule_sys"), "rule_us_rule_sys", results)


# =========================================================================
# 2. LLM US + LLM RG (end-to-end)
# =========================================================================
def run_llm_us_llm_rg():
    from convlab.base_models.llm.user_simulator import LLM_US, LLM_RG

    goals = _load_test_goals()
    user_model = LLM_US("litellm", LLM_MODEL)
    system_model = LLM_RG("litellm", LLM_MODEL)

    results = []
    for i, goal in enumerate(goals):
        print(f"  Dialog {i+1}/{len(goals)} ...")
        try:
            user_model.init_session(goal)
            system_model.init_session()

            sys_msg = "Hello, how may I help you today?"
            conversation = [{"turn": 0, "system": sys_msg}]
            turns = 0

            for _ in range(20):
                user_msg = user_model.response(sys_msg)
                if user_model.is_terminated:
                    conversation.append({"turn": turns, "user": user_msg})
                    break
                sys_msg = system_model.response(user_msg)
                turns += 1
                conversation.append({"turn": turns, "user": user_msg, "system": sys_msg})

            completed = user_model.is_terminated and any(
                "[END]" in str(c.get("user", "")) for c in conversation
            )
            reward = user_model.get_reward()
        except Exception as e:
            print(f"    Error: {e}")
            completed, reward, conversation, turns = False, None, [], 0

        results.append({
            "dialogue_id": i,
            "goal_description": goal.get("description", ""),
            "turns": turns,
            "completed": completed,
            "success": completed,
            "reward": reward,
            "conversation": conversation,
        })
        status = "OK" if completed else "INCOMPLETE"
        print(f"    turns={turns} {status} reward={reward}")

    return _save(os.path.join(OUT_ROOT, "llm_us_llm_rg"), "llm_us_llm_rg", results,
                 extra_summary={"model": LLM_MODEL})


# =========================================================================
# 3. LLM US + Rule pipeline system (NLU→DST→Policy→NLG)
# =========================================================================
def run_llm_us_rule_sys():
    from convlab.policy.rule.multiwoz import RulePolicy
    from convlab.nlg.template.multiwoz import TemplateNLG
    from convlab.dialog_agent import PipelineAgent
    from convlab.base_models.llm.nlu import LLM_NLU
    from convlab.base_models.llm.user_simulator import LLM_US
    from convlab.util.unified_datasets_util import load_dataset

    dataset = load_dataset("multiwoz21")
    example_dialogs = dataset["train"][:3]
    goals = _load_test_goals()

    sys_nlu = LLM_NLU("multiwoz21", "litellm", LLM_MODEL, "user", example_dialogs)
    sys_agent = PipelineAgent(
        sys_nlu, CompatRuleDST(), RulePolicy(), TemplateNLG(is_user=False), name="sys"
    )
    user_model = LLM_US("litellm", LLM_MODEL)

    results = []
    for i, goal in enumerate(goals):
        print(f"  Dialog {i+1}/{len(goals)} ...")
        try:
            user_model.init_session(goal)
            sys_agent.init_session()

            sys_msg = "Hello, how may I help you today?"
            conversation = [{"turn": 0, "system": sys_msg}]
            turns = 0

            for _ in range(20):
                user_msg = user_model.response(sys_msg)
                if user_model.is_terminated:
                    conversation.append({"turn": turns, "user": user_msg})
                    break
                sys_msg = sys_agent.response(user_msg)
                turns += 1
                conversation.append({"turn": turns, "user": user_msg, "system": sys_msg})

            completed = user_model.is_terminated and any(
                "[END]" in str(c.get("user", "")) for c in conversation
            )
            reward = user_model.get_reward()
        except Exception as e:
            print(f"    Error: {e}")
            traceback.print_exc()
            completed, reward, conversation, turns = False, None, [], 0

        results.append({
            "dialogue_id": i,
            "goal_description": goal.get("description", ""),
            "turns": turns,
            "completed": completed,
            "success": completed,
            "reward": reward,
            "conversation": conversation,
        })
        status = "OK" if completed else "INCOMPLETE"
        print(f"    turns={turns} {status} reward={reward}")

    return _save(os.path.join(OUT_ROOT, "llm_us_rule_sys"), "llm_us_rule_sys", results,
                 extra_summary={"model": LLM_MODEL})


# =========================================================================
# Main
# =========================================================================
COMBOS = [
    ("rule_us_rule_sys",  run_rule_us_rule_sys),
    ("llm_us_llm_rg",    run_llm_us_llm_rg),
    ("llm_us_rule_sys",   run_llm_us_rule_sys),
]


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_summaries = {}

    for name, func in COMBOS:
        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"{'#'*70}")
        try:
            all_summaries[name] = func()
        except Exception:
            traceback.print_exc()
            all_summaries[name] = {"error": traceback.format_exc()[:500]}
            print(f"  FAILED: {name}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_ROOT, f"all_summaries_{ts}.json")
    with open(path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*70}")
    print("FINAL RESULTS  (10 test dialogues each)")
    print(f"{'='*70}")
    print(f"{'Combo':<25} {'Completion':>12} {'Success':>12} {'Avg Turns':>12}")
    print("-" * 61)
    for name, s in all_summaries.items():
        if "error" in s:
            print(f"{name:<25} {'ERROR':>12}")
        else:
            print(f"{name:<25} {s['completion_rate']:>11.0%} {s['success_rate']:>11.0%} {s['avg_turns']:>11.1f}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
