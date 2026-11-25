"""Phase 1c: generate every single-point mutant of every ref, validate it against the verilog-eval TB
(read-only), and compute ground truth from a full-depth VCD (DUT vs golden internals).

Output: work/candidates.jsonl  (one record per mutant, valid or not)

    python -m sgd.s03_mutants [--jobs 16] [--only Prob120_fsm3s]
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor

from .common import (REFS, TB_RESEED, WORK, Vcd, first_divergence, posedges, problems, ref_as_dut,
                     simulate, time_to_cycle)
from .mutate import generate, stmt_lhs, tokenize, annotate
from .s01_copy_refs import RESEED_SEEDS

DUT_SCOPE, GOLD_SCOPE = "tb.top_module1", "tb.good1"


def design_stats(src: str) -> dict:
    toks = tokenize(src)
    annotate(src, toks)
    body_lines = {t.line for t in toks if t.region in ("seq", "comb", "other")}
    return {"body_lines": len(body_lines), "sequential": any(t.region == "seq" for t in toks),
            "n_lines": len(src.splitlines())}


def ground_truth(vcd_path) -> dict:
    v = Vcd(vcd_path)
    dut, gold = v.scope_vars(DUT_SCOPE), v.scope_vars(GOLD_SCOPE)
    clk = v.find("tb.stim1", "clk") or v.find("tb", "clk")
    edges = posedges(v.changes[clk.code]) if clk else []
    div = {}
    for name, dv in dut.items():
        gv = gold.get(name)
        if gv is None:
            continue
        t = first_divergence(v.changes[dv.code], v.changes[gv.code])
        if t is not None:
            div[name] = t
    if not div:
        return {"t_act": None, "div_signals": [], "div_order": [], "clk_edges": len(edges)}
    t_act = min(div.values())
    order = sorted(div, key=lambda n: (div[n], n))
    return {"t_act": t_act, "cyc_act": time_to_cycle(t_act, edges),
            "div_signals": [n for n in order if div[n] == t_act],
            "div_order": [[n, div[n], time_to_cycle(div[n], edges)] for n in order],
            "clk_edges": len(edges), "_edges": edges}


def run_one(task: tuple[str, int]) -> dict:
    p, k = task
    src = ref_as_dut((REFS / f"{p}_ref.sv").read_text())
    stats = design_stats(src)
    m = generate(src)[k]
    d = WORK / "cand" / p / f"m{k:03d}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    dut = d / "buggy.sv"
    dut.write_text(m.src)
    rec = {"problem": p, "k": k, **m.meta(), **stats, "src": m.src, "text_lhs": stmt_lhs(m.src, m.line)}
    r = simulate(p, dut, d, dump_internals=True)
    rec.update(compiled=r.compiled, wall_timeout=r.timeout, tb_watchdog=r.tb_watchdog,
               mismatches=r.mismatches, samples=r.samples, t_fail=r.t_fail,
               out_first_mismatch=r.out_first_mismatch, out_mismatch_count=r.out_mismatch_count)
    rec["valid"] = bool(r.compiled and not r.timeout and r.mismatches)
    if not r.compiled:
        rec["compile_err"] = r.compile_log[-400:]
    if rec["valid"]:
        gt = ground_truth(d / "wave.vcd")
        edges = gt.pop("_edges", [])
        rec.update(gt)
        rec["cyc_fail"] = time_to_cycle(r.t_fail, edges)
        rec["failing_outputs"] = sorted(r.out_first_mismatch, key=r.out_first_mismatch.get)
        rec["latency_cycles"] = (rec["cyc_fail"] - rec["cyc_act"]) if gt["t_act"] is not None else None
        for s in RESEED_SEEDS:
            rr = simulate(p, dut, d / f"seed{s}", tb=TB_RESEED / f"{p}_test_seed{s}.sv")
            rec[f"fails_seed{s}"] = bool(rr.compiled and not rr.timeout and rr.mismatches)
    # keep only the mutant source; VCDs are regenerated for selected bugs in s04
    for f in sorted(d.rglob("*"), reverse=True):
        if f.is_file() and f.name != "buggy.sv":
            f.unlink()
        elif f.is_dir():
            f.rmdir()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    probs = a.only or problems()
    WORK.mkdir(exist_ok=True)
    n = nv = 0
    tasks = [(p, k) for p in probs for k in range(len(generate(ref_as_dut((REFS / f"{p}_ref.sv").read_text()))))]
    print(f"{len(tasks)} candidate mutants over {len(probs)} problems", flush=True)
    with open(WORK / "candidates.jsonl", "w") as f, ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(run_one, tasks, chunksize=4):
            f.write(json.dumps(r) + "\n")
            n += 1
            nv += r["valid"]
            if n % 100 == 0:
                print(f"{n}/{len(tasks)} done, {nv} valid", flush=True)
    print(f"TOTAL {nv}/{n} valid mutants")


if __name__ == "__main__":
    main()
