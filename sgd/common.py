"""Shared paths, simulation runner, TB-log parser and a minimal VCD reader.

verilog-eval is used READ-ONLY: its *_test.sv files are passed to iverilog by
absolute path and never copied or modified. Refs/prompts are copied into refs/.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# verilog-eval checkout (https://github.com/NVlabs/verilog-eval), default: sibling of this repo
VE_ROOT = Path(os.environ.get("VERILOG_EVAL_DIR", ROOT.parent / "verilog-eval")).resolve()
VE_DATASET = VE_ROOT / "dataset_spec-to-rtl"
REFS = ROOT / "refs"
TB_RESEED = ROOT / "tb_reseed"
BUGS = ROOT / "bugs"
WORK = ROOT / "work"
DUMP_SV = ROOT / "sgd" / "sg_dump.sv"   # extra top: dumps tb.top_module1 + tb.good1 internals

IVERILOG_FLAGS = ["-g2012", "-Wall", "-Wno-timescale", "-Winfloop"]
SIM_TIMEOUT_S = 15


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def problems() -> list[str]:
    return sorted(p.name[: -len("_ref.sv")] for p in VE_DATASET.glob("Prob*_ref.sv"))


# --------------------------------------------------------------------------- RTL text

def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def normalize(src: str) -> str:
    """Comment-free, no trailing spaces, no blank-line runs. Used for both ref-as-DUT and mutants
    so line numbers are comparable and the diff carries no formatting hints."""
    lines = [l.rstrip().replace("\t", "  ") for l in strip_comments(src).splitlines()]
    out, blank = [], False
    for l in lines:
        if not l.strip():
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(l)
        blank = False
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def ref_as_dut(ref_src: str) -> str:
    return normalize(re.sub(r"\bmodule\s+RefModule\b", "module TopModule", ref_src, count=1))


# --------------------------------------------------------------------------- simulation

@dataclass
class SimResult:
    compiled: bool
    compile_log: str = ""
    sim_log: str = ""
    timeout: bool = False        # wall-clock hang (vvp killed)
    tb_watchdog: bool = False    # TB's own #1000000 "TIMEOUT" fired; mismatch count up to then is still valid
    mismatches: int | None = None
    samples: int | None = None
    out_first_mismatch: dict[str, int] = field(default_factory=dict)  # output -> first mismatch time (ps)
    out_mismatch_count: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.compiled and not self.timeout and self.mismatches == 0

    @property
    def t_fail(self) -> int | None:
        return min(self.out_first_mismatch.values()) if self.out_first_mismatch else None


_HINT = re.compile(r"Hint: Output '([^']+)' has (\d+) mismatches\. First mismatch occurred at time (\d+)\.")
_MM = re.compile(r"Mismatches: (\d+) in (\d+) samples")


def scrub_paths(log: str) -> str:
    """Replace absolute local paths in simulator output with neutral placeholders."""
    return log.replace(str(VE_DATASET) + "/", "<verilog-eval>/").replace(str(ROOT) + "/", "")


def parse_tb_log(log: str, r: SimResult) -> SimResult:
    for name, cnt, t in _HINT.findall(log):
        r.out_mismatch_count[name] = int(cnt)
        r.out_first_mismatch[name] = int(t)
    m = _MM.search(log)
    if m:
        r.mismatches, r.samples = int(m.group(1)), int(m.group(2))
    r.tb_watchdog = "TIMEOUT" in log
    return r


def simulate(problem: str, dut_path: Path, run_dir: Path, *, tb: Path | None = None,
             dump_internals: bool = False) -> SimResult:
    """Compile verilog-eval TB (read-only) + copied ref + DUT; run in run_dir (wave.vcd lands there)."""
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    tb = tb or VE_DATASET / f"{problem}_test.sv"
    srcs = [str(tb), str(REFS / f"{problem}_ref.sv"), str(dut_path.resolve())]
    tops = ["-s", "tb"]
    if dump_internals:
        srcs.append(str(DUMP_SV))
        tops += ["-s", "sg_dump"]
    simbin = run_dir / "sim.vvp"
    c = subprocess.run(["iverilog", *IVERILOG_FLAGS, *tops, "-o", str(simbin), *srcs],
                       capture_output=True, text=True, cwd=run_dir)
    r = SimResult(compiled=c.returncode == 0, compile_log=scrub_paths(c.stdout + c.stderr))
    if not r.compiled:
        return r
    try:
        s = subprocess.run(["vvp", "-n", str(simbin)], capture_output=True, text=True,
                           cwd=run_dir, timeout=SIM_TIMEOUT_S)
        r.sim_log = scrub_paths(s.stdout + s.stderr)
    except subprocess.TimeoutExpired:
        r.timeout = True
        return r
    return parse_tb_log(r.sim_log, r)


# --------------------------------------------------------------------------- VCD

@dataclass
class VcdVar:
    scope: str      # e.g. "tb.top_module1"
    name: str
    width: int
    code: str


class Vcd:
    """Minimal VCD reader: per id-code list of (time, value-string), values left-padded to width."""

    def __init__(self, path: Path):
        self.vars: list[VcdVar] = []
        self.changes: dict[str, list[tuple[int, str]]] = {}
        widths: dict[str, int] = {}
        scope: list[str] = []
        t = 0
        in_header = True
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if in_header:
                    tok = line.split()
                    if tok[0] == "$scope":
                        scope.append(tok[2])
                    elif tok[0] == "$upscope":
                        scope.pop()
                    elif tok[0] == "$var":
                        w, code, name = int(tok[2]), tok[3], tok[4]
                        self.vars.append(VcdVar(".".join(scope), name, w, code))
                        widths[code] = w
                        self.changes.setdefault(code, [])
                    elif tok[0] == "$enddefinitions":
                        in_header = False
                    continue
                c = line[0]
                if c == "#":
                    t = int(line[1:])
                elif c in "01xzXZ":
                    self._add(line[1:], t, line[0].lower(), widths)
                elif c in "bB":
                    val, code = line[1:].split()
                    self._add(code, t, val.lower(), widths)
                elif c in "rR":
                    val, code = line[1:].split()
                    self._add(code, t, val, widths)
                # $dumpvars/$end/$comment lines in body are ignored

    def _add(self, code: str, t: int, v: str, widths: dict[str, int]) -> None:
        w = widths.get(code, 1)
        if len(v) < w and v[0] in "01":
            v = v.rjust(w, "0")
        elif len(v) < w:
            v = v.rjust(w, v[0])
        ch = self.changes.setdefault(code, [])
        if ch and ch[-1][0] == t:
            ch[-1] = (t, v)
        else:
            ch.append((t, v))

    def scope_vars(self, scope: str) -> dict[str, VcdVar]:
        return {v.name: v for v in self.vars if v.scope == scope}

    def find(self, scope: str, name: str) -> VcdVar | None:
        return self.scope_vars(scope).get(name)


def tb_mismatch(dut: str | None, exp: str | None) -> bool:
    """verilog-eval scoreboard rule: an x/z bit in the expected value matches anything; otherwise the DUT
    bit must equal it exactly (an x in the DUT does not match 0/1)."""
    if dut is None or exp is None:
        return False
    n = max(len(dut), len(exp))
    dut, exp = dut.rjust(n, dut[0] if dut[0] in "xz" else "0"), exp.rjust(n, exp[0] if exp[0] in "xz" else "0")
    return any(e in "01" and d != e for d, e in zip(dut, exp))


def first_divergence(a: list[tuple[int, str]], b: list[tuple[int, str]]) -> int | None:
    """Earliest time at which trace a (DUT) mismatches trace b (golden) under the testbench rule: an x/z
    bit in the golden value is a don't-care, so an uninitialized golden register is not a divergence."""
    times = sorted({t for t, _ in a} | {t for t, _ in b})
    ia = ib = 0
    va = vb = None
    for t in times:
        while ia < len(a) and a[ia][0] <= t:
            va = a[ia][1]; ia += 1
        while ib < len(b) and b[ib][0] <= t:
            vb = b[ib][1]; ib += 1
        if va is not None and tb_mismatch(va, vb):
            return t
    return None


def posedges(trace: list[tuple[int, str]]) -> list[int]:
    out, prev = [], None
    for t, v in trace:
        if v == "1" and prev == "0":
            out.append(t)
        prev = v
    return out


def time_to_cycle(t: int, edges: list[int]) -> int:
    """Number of clk posedges at or before t (cycle index, 0 = before first edge)."""
    import bisect
    return bisect.bisect_right(edges, t)
