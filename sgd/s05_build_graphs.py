"""Phase 2b: build bugs/<id>/public/graph.json for every bug and run structural sanity checks.

Checks (reported, and asserted by tests/test_graph.py):
  coverage      every assignment line in buggy.sv appears as a driver line in the graph
  vcd_names     every non-memory graph node exists in public/dut_wave.vcd (scope top_module1)
  reachability  the ground-truth signal is in the backward cone of the failing outputs
                and the ground-truth line is a driver or guard line of a node in that cone

    python -m sgd.s05_build_graphs
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import BUGS
from .graph import backward_cone, build_graph
from .mutate import annotate, tokenize


def assignment_lines(src: str) -> set[int]:
    toks = tokenize(src)
    annotate(src, toks)
    lines = set()
    for i, t in enumerate(toks):
        if t.region not in ("seq", "comb", "other"):
            continue
        if t.text == "=" or (t.text == "<=" and t.pdepth == 0):
            # skip for-loop headers (inside parens) and declarations
            if t.pdepth == 0:
                lines.add(t.line)
    return lines


def vcd_names(vcd: Path) -> set[str]:
    names, in_dut = set(), False
    for line in vcd.open():
        if line.startswith("$scope"):
            in_dut = "top_module1" in line
        elif line.startswith("$var") and in_dut:
            names.add(line.split()[4])
        elif line.startswith("$enddefinitions"):
            break
    return names


def stmt_signals(g: dict, line: int) -> list[str]:
    """Targets of every driver whose statement span, guard, or always-header covers `line`."""
    out = []
    blocks = {b["id"] for b in g["blocks"] if b["line"] == line}
    for name, n in g["nodes"].items():
        for d in n["drivers"]:
            if d["line"] <= line <= d["line_end"] or line in d["guard_lines"] or d["block"] in blocks:
                if name not in out:
                    out.append(name)
    return out


def check(bug_dir: Path, g: dict) -> dict:
    meta_path = bug_dir / "private" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["gt_signals"] = stmt_signals(g, meta["gt_line"]) or meta["text_lhs"]
    meta_path.write_text(json.dumps(meta, indent=1))
    src = (bug_dir / "public" / "buggy.sv").read_text()
    driver_lines = {d["line"] for n in g["nodes"].values() for d in n["drivers"]}
    missing = sorted(assignment_lines(src) - driver_lines)
    res = {"bug_id": meta["bug_id"], "nodes": len(g["nodes"]), "edges": len(g["edges"]),
           "missing_driver_lines": missing, "diagnostics": len(g["diagnostics"])}
    vcd = bug_dir / "public" / "dut_wave.vcd"
    if vcd.exists():
        names = vcd_names(vcd)
        res["missing_in_vcd"] = sorted(n for n, v in g["nodes"].items()
                                       if not v["memory"] and "unused" not in v["roles"] and n not in names)
    cone = backward_cone(g, meta["failing_outputs"], skip_roles=())
    res["gt_signal_in_cone"] = any(s in cone for s in meta["gt_signals"])
    res["gt_hops"] = min((cone[s] for s in meta["gt_signals"] if s in cone), default=None)
    cone_lines = set()
    for n in cone:
        for d in g["nodes"].get(n, {}).get("drivers", []):
            cone_lines.update(range(d["line"], d["line_end"] + 1))
            cone_lines.update(d["guard_lines"])
    for b in g["blocks"]:
        cone_lines.add(b["line"])  # always-header mutations (sync/async reset)
    res["gt_line_in_cone"] = meta["gt_line"] in cone_lines
    return res


def main() -> None:
    rows = []
    for bug_dir in sorted(p for p in BUGS.iterdir() if p.is_dir()):
        src = (bug_dir / "public" / "buggy.sv").read_text()
        g = build_graph(src)
        (bug_dir / "public" / "graph.json").write_text(json.dumps(g, indent=1))
        rows.append(check(bug_dir, g))
    (BUGS / "graph_checks.json").write_text(json.dumps(rows, indent=1))
    # keep index.json in sync with the graph-derived ground truth
    index = json.loads((BUGS / "index.json").read_text())
    hops = {r["bug_id"]: r["gt_hops"] for r in rows}
    for it in index:
        m = json.loads((BUGS / it["bug_id"] / "private" / "meta.json").read_text())
        it["gt_signals"] = m["gt_signals"]
        it["gt_hops"] = hops[it["bug_id"]]
    (BUGS / "index.json").write_text(json.dumps(index, indent=1))
    n = len(rows)
    cov = [r for r in rows if r["missing_driver_lines"]]
    vcdm = [r for r in rows if r.get("missing_in_vcd")]
    print(f"{n} graphs | nodes avg {sum(r['nodes'] for r in rows)/n:.1f} max {max(r['nodes'] for r in rows)}"
          f" | edges avg {sum(r['edges'] for r in rows)/n:.1f} max {max(r['edges'] for r in rows)}")
    print(f"coverage: {n-len(cov)}/{n} bugs have every assignment line as a driver"
          + (f"  e.g. {[(r['bug_id'], r['missing_driver_lines']) for r in cov[:5]]}" if cov else ""))
    print(f"vcd names: {n-len(vcdm)}/{n} ok" + (f"  e.g. {[(r['bug_id'], r['missing_in_vcd']) for r in vcdm[:5]]}" if vcdm else ""))
    print(f"gt signal in backward cone of failing outputs: {sum(r['gt_signal_in_cone'] for r in rows)}/{n}")
    print(f"gt line is a driver/guard/block line in that cone: {sum(r['gt_line_in_cone'] for r in rows)}/{n}")
    hops = [r["gt_hops"] for r in rows if r["gt_hops"] is not None]
    from collections import Counter
    print(f"hops from failing output to gt signal: {dict(sorted(Counter(hops).items()))}")


if __name__ == "__main__":
    main()
