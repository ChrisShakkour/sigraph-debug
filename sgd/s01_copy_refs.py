"""Phase 1a: copy verilog-eval refs + prompts locally, record sha256 of every verilog-eval file we use,
and build reseeded TB *copies* (tb_reseed/) for the fix-robustness check.

    python -m sgd.s01_copy_refs            # copy + manifest + reseeded TBs
    python -m sgd.s01_copy_refs --verify   # check verilog-eval files are unchanged vs manifest
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

from .common import REFS, ROOT, TB_RESEED, VE_DATASET, VE_ROOT, problems, sha256

MANIFEST = ROOT / "refs" / "manifest.json"
RESEED_SEEDS = [1234, 98765]


def make_reseed_tb(src: str, seed: int) -> str:
    """Seed $random/$urandom inside stimulus_gen only. Leaves the checker (module tb) untouched."""
    i = src.index("module stimulus_gen")
    j = src.index("module tb")
    head, stim, tail = src[:i], src[i:j], src[j:]
    hdr_end = stim.index(");") + 2
    stim = stim[:hdr_end] + f"\n\tinteger sg_seed = {seed};\n" + stim[hdr_end:]
    stim = re.sub(r"\$random\b(?!\s*\()", "$random(sg_seed)", stim)
    stim = re.sub(r"\$urandom\b(?!_|\s*\()", "$urandom(sg_seed)", stim)
    return head + stim + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        man = json.loads(MANIFEST.read_text())
        bad = [f for f, h in man["verilog_eval_sha256"].items() if sha256(VE_DATASET / f) != h]
        print("verilog-eval unchanged" if not bad else f"CHANGED: {bad}")
        sys.exit(1 if bad else 0)

    REFS.mkdir(exist_ok=True)
    TB_RESEED.mkdir(exist_ok=True)
    hashes = {}
    for p in problems():
        for suffix in ("_ref.sv", "_prompt.txt", "_test.sv"):
            f = VE_DATASET / f"{p}{suffix}"
            hashes[f.name] = sha256(f)
            if suffix != "_test.sv":
                shutil.copy2(f, REFS / f.name)
        tb_src = (VE_DATASET / f"{p}_test.sv").read_text()
        for seed in RESEED_SEEDS:
            (TB_RESEED / f"{p}_test_seed{seed}.sv").write_text(make_reseed_tb(tb_src, seed))
    rev = subprocess.run(["git", "-C", str(VE_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    MANIFEST.write_text(json.dumps({"source": "https://github.com/NVlabs/verilog-eval (dataset_spec-to-rtl)",
                                    "commit": rev, "verilog_eval_sha256": hashes}, indent=1))
    print(f"copied {len(problems())} refs+prompts, {len(hashes)} hashes, reseeded TBs seeds={RESEED_SEEDS}")


if __name__ == "__main__":
    main()
