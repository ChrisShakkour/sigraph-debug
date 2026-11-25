"""Phase 4: score agent runs against the private ground truth. Writes runs/<tag>/scores.jsonl and prints a
per-arm summary (overall, root cause off the failing output, long-horizon).

Per run:
  line_exact / line_pm1      root_cause_line vs gt_line
  signal_gt                  root_cause_signal in gt_signals (the mutated statement's targets)
  signal_div                 root_cause_signal in gt_signals or any signal that diverged from the golden model
  blamed_symptom             blamed a failing output that is not the root cause
  act_pm1                    cycle of activation_time within +-1 of the true activation cycle
  fix_compiles / fix_pass    fixed_module compiles / passes the original verilog-eval TB (read-only)
  fix_robust                 also passes both reseeded TB copies
  off_cone_visits            inspected signals outside the backward cone of the failing outputs, or clk/reset
  steps, prompt_tokens, output_tokens, wall_seconds

    python -m sgd.s07_score --tag pilot
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import BUGS, TB_RESEED, WORK, simulate
from .graph import backward_cone
from .s01_copy_refs import RESEED_SEEDS
from .s06_run_agents import RUNS
from .waves import DutWaves


def _int(x) -> int | None:
    try:
        return int(str(x).strip().split()[0].rstrip(",;ps"))
    except (ValueError, IndexError):
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else None


def check_fix(meta: dict, text: str, run_dir: Path) -> dict:
    out = {"fix_compiles": False, "fix_pass": False, "fix_robust": False}
    if not isinstance(text, str) or "module" not in text:
        return out
    if not re.search(r"\bmodule\s+TopModule\b", text):
        text = re.sub(r"\bmodule\s+\w+", "module TopModule", text, count=1)
    run_dir.mkdir(parents=True, exist_ok=True)
    dut = run_dir / "fixed.sv"
    dut.write_text(text)
    r = simulate(meta["problem"], dut, run_dir / "orig")
    out["fix_compiles"] = r.compiled
    out["fix_pass"] = bool(r.passed)
    if r.passed:
        out["fix_robust"] = all(
            simulate(meta["problem"], dut, run_dir / f"seed{s}",
                     tb=TB_RESEED / f"{meta['problem']}_test_seed{s}.sv").passed for s in RESEED_SEEDS)
    return out


def score_run(rec: dict, tag: str) -> dict:
    bug = BUGS / rec["bug_id"]
    meta = json.loads((bug / "private" / "meta.json").read_text())
    g = json.loads((bug / "public" / "graph.json").read_text())
    f = rec["final"] or {}
    line = _int(f.get("root_cause_line"))
    sig = str(f.get("root_cause_signal") or "").strip()
    sig = re.sub(r"\[.*$", "", sig.split(".")[-1]).strip()
    gt = set(meta["gt_signals"])
    s = {"arm": rec["arm"], "bug_id": rec["bug_id"], "answered": bool(rec["final"]),
         "line_exact": line == meta["gt_line"], "line_pm1": line is not None and abs(line - meta["gt_line"]) <= 1,
         "signal_gt": sig in gt, "signal_div": sig in gt | set(meta["div_signals"]),
         "blamed_symptom": sig in meta["failing_outputs"] and sig not in gt}
    t_act = _int(f.get("activation_time"))
    if t_act is not None:
        w = DutWaves(bug / "public" / "dut_wave.vcd")
        s["act_pm1"] = abs(w.cycle(t_act, change=True) - meta["cyc_act"]) <= 1
    else:
        s["act_pm1"] = False
    s.update(check_fix(meta, f.get("fixed_module"), WORK / "score" / tag / rec["arm"] / rec["bug_id"]))
    cone = backward_cone(g, meta["failing_outputs"], skip_roles=("clock", "reset"))
    s["off_cone_visits"] = sum(1 for v in rec.get("visited", []) if v not in cone)
    for k in ("n_steps", "prompt_tokens", "output_tokens", "wall_seconds", "parse_errors"):
        s[k] = rec.get(k)
    return s


def summarize(rows: list[dict]) -> str:
    keys = ["answered", "line_exact", "line_pm1", "signal_gt", "signal_div", "blamed_symptom", "act_pm1",
            "fix_compiles", "fix_pass", "fix_robust"]
    cost = ["n_steps", "prompt_tokens", "wall_seconds", "off_cone_visits"]
    arms = sorted({r["arm"] for r in rows}, key=["A", "B", "T", "T-win", "T-graph"].index)
    subsets = {"all": lambda r: True, "hops>0": lambda r: (r["gt_hops"] or 0) > 0,
               "long_horizon": lambda r: r["long_horizon"]}
    lines = []
    for name, pred in subsets.items():
        sub = [r for r in rows if pred(r)]
        if not sub:
            continue
        lines.append(f"\n== {name} ==")
        lines.append(f"{'metric':16s}" + "".join(f"{a:>9s}" for a in arms))
        by = {a: [r for r in sub if r["arm"] == a] for a in arms}
        lines.append(f"{'n':16s}" + "".join(f"{len(by[a]):9d}" for a in arms))
        for k in keys:
            lines.append(f"{k:16s}" + "".join(
                f"{sum(bool(r[k]) for r in by[a]) / max(1, len(by[a])):9.2f}" for a in arms))
        for k in cost:
            lines.append(f"{k:16s}" + "".join(
                f"{sum(r[k] or 0 for r in by[a]) / max(1, len(by[a])):9.1f}" for a in arms))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = RUNS / a.tag
    index = {b["bug_id"]: b for b in json.loads((BUGS / "index.json").read_text())}
    rows = []
    for p in sorted(out.glob("*/*.json")):
        rec = json.loads(p.read_text())
        s = score_run(rec, a.tag)
        s["gt_hops"] = index[rec["bug_id"]]["gt_hops"]
        s["long_horizon"] = index[rec["bug_id"]]["long_horizon"]
        s["category"] = index[rec["bug_id"]]["category"]
        rows.append(s)
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    text = summarize(rows)
    (out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
