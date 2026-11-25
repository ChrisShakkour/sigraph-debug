"""Phase 1b: sanity - each ref (renamed TopModule, normalized) must pass the original verilog-eval TB
and the reseeded TB copies. Writes work/baseline.json."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from .common import REFS, TB_RESEED, WORK, problems, ref_as_dut, simulate
from .s01_copy_refs import RESEED_SEEDS


def check(p: str) -> tuple[str, dict]:
    d = WORK / "baseline" / p
    d.mkdir(parents=True, exist_ok=True)
    dut = d / "ref_dut.sv"
    dut.write_text(ref_as_dut((REFS / f"{p}_ref.sv").read_text()))
    res = {"orig": simulate(p, dut, d / "orig").passed}
    for s in RESEED_SEEDS:
        r = simulate(p, dut, d / f"seed{s}", tb=TB_RESEED / f"{p}_test_seed{s}.sv")
        res[f"seed{s}"] = r.passed
        res[f"seed{s}_compiled"] = r.compiled
    return p, res


if __name__ == "__main__":
    with ProcessPoolExecutor(16) as ex:
        out = dict(ex.map(check, problems()))
    (WORK / "baseline.json").write_text(json.dumps(out, indent=1))
    bad_orig = [p for p, r in out.items() if not r["orig"]]
    bad_seed = [p for p, r in out.items() if not all(r[f"seed{s}"] for s in RESEED_SEEDS)]
    print(f"{len(out)} problems; ref fails orig TB: {bad_orig}")
    print(f"ref fails a reseeded TB ({len(bad_seed)}): {bad_seed}")
