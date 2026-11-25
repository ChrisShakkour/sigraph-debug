"""Phase 3: run agent arms over bugs with Devstral. Resumable: one JSON per (arm, bug) under runs/<tag>/<arm>/.

    python -m sgd.s06_run_agents --tag pilot --pilot                 # 20 dev bugs x A, B, T
    python -m sgd.s06_run_agents --tag main --split test --arms A B T T-win T-graph
    python -m sgd.s06_run_agents --tag dbg --bugs bug111 --arms T

Run long jobs in a detached tmux session; the runner skips (arm, bug) pairs that already have a record.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

from .agents import ARMS, run_agent
from .common import BUGS, ROOT
from .llm import Ollama

RUNS = ROOT / "runs"


def pilot_bugs(n: int = 20, seed: int = 0) -> list[str]:
    """n dev bugs: half with the root cause off the failing output (gt_hops > 0), half on it."""
    ix = [b for b in json.loads((BUGS / "index.json").read_text()) if b["split"] == "dev"]
    rng = random.Random(seed)
    far = sorted(b["bug_id"] for b in ix if b["gt_hops"])
    near = sorted(b["bug_id"] for b in ix if not b["gt_hops"])
    pick = rng.sample(far, min(n // 2, len(far)))
    pick += rng.sample(near, n - len(pick))
    return sorted(pick)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "T"], choices=ARMS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--split", choices=["dev", "test"])
    g.add_argument("--bugs", nargs="+")
    ap.add_argument("--seed", type=int, default=0, help="LLM sampling seed")
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    if a.pilot:
        bugs = pilot_bugs()
    elif a.split:
        bugs = sorted(b["bug_id"] for b in json.loads((BUGS / "index.json").read_text()) if b["split"] == a.split)
    else:
        bugs = a.bugs
    llm = Ollama(seed=a.seed, temperature=a.temperature)
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"arms": a.arms, "bugs": bugs, "model": llm.model,
                                                 "options": llm.options}, indent=1))
    todo = [(arm, b) for b in bugs for arm in a.arms if not (out / arm / f"{b}.json").exists()]
    print(f"{len(todo)} runs to do ({len(bugs)} bugs x {len(a.arms)} arms, {len(bugs) * len(a.arms) - len(todo)} done)",
          flush=True)
    t0 = time.time()
    for i, (arm, b) in enumerate(todo, 1):
        try:
            rec = run_agent(arm, BUGS / b, llm)
        except Exception as e:  # noqa: BLE001 - keep going; the error is recorded and the pair retried next time
            print(f"[{i}/{len(todo)}] {arm:7s} {b}: ERROR {type(e).__name__}: {e}", flush=True)
            (out / "errors.log").open("a").write(f"{arm} {b}\n{traceback.format_exc()}\n")
            continue
        (out / arm).mkdir(exist_ok=True)
        (out / arm / f"{b}.json").write_text(json.dumps(rec, indent=1))
        f = rec["final"] or {}
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {arm:7s} {b}: {rec['stop']:10s} steps={rec['n_steps']:2d} "
              f"line={f.get('root_cause_line')} sig={f.get('root_cause_signal')} "
              f"{rec['wall_seconds']:.0f}s  eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
