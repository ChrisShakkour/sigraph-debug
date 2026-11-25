"""Phase 1d: select the benchmark from work/candidates.jsonl, split dev/test by problem, and materialize
bugs/<bug_id>/{public,private}.

public/  (what agents may see)
  buggy.sv      - DUT (module TopModule), comment-free, normalized
  spec.txt      - verilog-eval problem prompt
  tb_log.txt    - output of the ORIGINAL verilog-eval TB on the buggy DUT
  dut_wave.vcd  - waveform filtered to the DUT scope + expected output values (scope tb_expected);
                  golden internals (tb.good1) are removed
private/ (ground truth - never shown to agents)
  meta.json, reference_fix.sv, wave_full.vcd

    python -m sgd.s04_select [--per-problem 3] [--long K]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .common import BUGS, REFS, WORK, ref_as_dut, simulate
from .mutate import ASSIGN_LHS, annotate, tokenize

SEED = 0
DEV_FRAC = 0.2
CATS = ["control", "datapath", "state"]


def seq_signals(src: str) -> set[str]:
    """Names assigned inside sequential (clocked) blocks."""
    toks = tokenize(src)
    annotate(src, toks)
    seq_lines = {t.line for t in toks if t.region == "seq"}
    lines = src.splitlines()
    return {m.group(1) for ln in seq_lines for m in ASSIGN_LHS.finditer(lines[ln - 1])}


def effective_activation(r: dict) -> None:
    """Activation = first divergence of a *state* (sequentially assigned) signal, capped at T_fail.
    A combinational divergence that is never latched (masked) does not count. Raw value kept as cyc_act_any."""
    r["cyc_act_any"] = r.get("cyc_act")
    regs = seq_signals(r["src"])
    cyc = [c for n, _, c in r.get("div_order", []) if n in regs and c <= r["cyc_fail"]]
    r["cyc_act"] = min(cyc) if cyc else r["cyc_fail"]
    r["latency_cycles"] = r["cyc_fail"] - r["cyc_act"]


def eligible(r: dict) -> bool:
    return r["valid"] and (r["sequential"] or r["body_lines"] >= 5)


def pick(cands: list[dict], n: int, long_k: int, rng: random.Random) -> list[dict]:
    rng.shuffle(cands)
    chosen: list[dict] = []
    used_lines: set[int] = set()

    def take(c: dict) -> None:
        chosen.append(c)
        used_lines.add(c["line"])

    longs = [c for c in cands if (c.get("latency_cycles") or 0) >= long_k]
    if longs:
        take(max(longs, key=lambda c: c["latency_cycles"]))
    for pass_ in range(2):  # pass 0: new category + new line; pass 1: any new line
        for cat in CATS:
            if len(chosen) >= n:
                break
            have = {c["category"] for c in chosen}
            for c in cands:
                if c in chosen or c["line"] in used_lines:
                    continue
                if c["category"] == cat and (pass_ == 1 or cat not in have):
                    take(c)
                    break
    for c in cands:
        if len(chosen) >= n:
            break
        if c not in chosen and c["line"] not in used_lines:
            take(c)
    return chosen[:n]


def filter_vcd(src: Path, dst: Path, dut_scope: str = "tb.top_module1") -> None:
    """Write a VCD containing only the DUT scope plus TB '<out>_ref' signals renamed to tb_expected.<out>."""
    keep: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # scope -> [(type width code name...)]
    keep_codes: set[str] = set()
    scope: list[str] = []
    header_meta: list[str] = []
    body_start = None
    lines = src.read_text().splitlines()
    for i, line in enumerate(lines):
        tok = line.split()
        if not tok:
            continue
        if tok[0] in ("$timescale", "$version", "$date") and "$end" in tok:
            header_meta.append(line)
        if tok[0] == "$scope":
            scope.append(tok[2])
        elif tok[0] == "$upscope":
            scope.pop()
        elif tok[0] == "$var":
            sc = ".".join(scope)
            vtype, width, code, name = tok[1], tok[2], tok[3], " ".join(tok[4:-1])
            if sc == dut_scope:
                keep["top_module1"].append((vtype, width, code, name))
                keep_codes.add(code)
            elif sc == "tb" and name.split()[0].endswith("_ref"):
                keep["tb_expected"].append((vtype, width, code, name.replace("_ref", "", 1)))
                keep_codes.add(code)
        elif tok[0] == "$enddefinitions":
            body_start = i + 1
            break
    out = header_meta[:]
    for sc, vs in keep.items():
        out.append(f"$scope module {sc} $end")
        seen = set()
        for vtype, width, code, name in vs:
            if (code, name) in seen:
                continue
            seen.add((code, name))
            out.append(f"$var {vtype} {width} {code} {name} $end")
        out.append("$upscope $end")
    out.append("$enddefinitions $end")
    for line in lines[body_start:]:
        if not line:
            continue
        c = line[0]
        if c == "#" or line.startswith("$"):
            out.append(line)
        elif c in "01xzXZ":
            if line[1:] in keep_codes:
                out.append(line)
        elif c in "bBrR":
            parts = line.split()
            if len(parts) == 2 and parts[1] in keep_codes:
                out.append(line)
    dst.write_text("\n".join(out) + "\n")


def materialize(bug_id: str, r: dict, split: str) -> dict:
    d = BUGS / bug_id
    if d.exists():
        shutil.rmtree(d)
    pub, priv = d / "public", d / "private"
    pub.mkdir(parents=True)
    priv.mkdir()
    (pub / "buggy.sv").write_text(r["src"])
    shutil.copy2(REFS / f"{r['problem']}_prompt.txt", pub / "spec.txt")
    run = WORK / "golden_runs" / bug_id
    if run.exists():
        shutil.rmtree(run)
    res = simulate(r["problem"], pub / "buggy.sv", run, dump_internals=True)
    assert res.mismatches == r["mismatches"], (bug_id, res.mismatches, r["mismatches"])
    (pub / "tb_log.txt").write_text(res.sim_log)
    shutil.move(run / "wave.vcd", priv / "wave_full.vcd")
    filter_vcd(priv / "wave_full.vcd", pub / "dut_wave.vcd")
    shutil.rmtree(run)
    (priv / "reference_fix.sv").write_text(ref_as_dut((REFS / f"{r['problem']}_ref.sv").read_text()))
    meta = {k: v for k, v in r.items() if k != "src"}
    meta.update(bug_id=bug_id, split=split, gt_line=r["line"], gt_signals=r["text_lhs"])
    (priv / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-problem", type=int, default=3)
    ap.add_argument("--long", type=int, default=8, help="latency (cycles) that marks a long-horizon bug")
    a = ap.parse_args()
    rng = random.Random(SEED)

    by_prob: dict[str, list[dict]] = defaultdict(list)
    for line in open(WORK / "candidates.jsonl"):
        r = json.loads(line)
        if eligible(r):
            effective_activation(r)
            by_prob[r["problem"]].append(r)
    probs = sorted(by_prob)

    seq = [p for p in probs if by_prob[p][0]["sequential"]]
    comb = [p for p in probs if not by_prob[p][0]["sequential"]]
    dev: set[str] = set()
    for group in (seq, comb):
        g = group[:]
        rng.shuffle(g)
        dev |= set(g[: round(len(g) * DEV_FRAC)])

    if BUGS.exists():
        shutil.rmtree(BUGS)
    BUGS.mkdir()
    index = []
    n = 0
    for p in probs:
        for r in pick(by_prob[p], a.per_problem, a.long, rng):
            n += 1
            bug_id = f"bug{n:03d}"
            m = materialize(bug_id, r, "dev" if p in dev else "test")
            m["long_horizon"] = (m.get("latency_cycles") or 0) >= a.long
            (BUGS / bug_id / "private" / "meta.json").write_text(json.dumps(m, indent=1))
            index.append({k: m.get(k) for k in ("bug_id", "problem", "split", "op", "category", "gt_line",
                                                 "gt_signals", "latency_cycles", "long_horizon", "mismatches",
                                                 "failing_outputs", "sequential", "body_lines")})
    (BUGS / "index.json").write_text(json.dumps(index, indent=1))

    from collections import Counter
    print(f"{len(probs)} eligible problems ({len(seq)} seq / {len(comb)} comb), {len(dev)} in dev")
    print(f"{len(index)} bugs: split={dict(Counter(i['split'] for i in index))}")
    print(f"category={dict(Counter(i['category'] for i in index))}")
    print(f"op={dict(Counter(i['op'] for i in index))}")
    print(f"long_horizon(>={a.long} cyc)={sum(i['long_horizon'] for i in index)}")


if __name__ == "__main__":
    main()
