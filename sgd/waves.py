"""Phase 2c: value windows over the *public* DUT waveform (bugs/<id>/public/dut_wave.vcd).

Sample points are the edges (both) of the TB sampling clock, exactly when the verilog-eval scoreboard
compares DUT vs. expected. The scoreboard sees the values settled *just before* the edge, so
`sampled(name, t)` returns the value at t-1. A window is the last K sample points up to and
including T_fail. Cycle of a sample = sampling-clock posedges strictly before t; cycle of a value change
= posedges at or before t (matches meta.json).
"""
from __future__ import annotations

import bisect
import re
from pathlib import Path

from .common import Vcd, tb_mismatch  # noqa: F401  (re-exported)

DUT, EXP, SAMPLE = "top_module1", "tb_expected", "tb_sample"


def const_to_int(text: str) -> int | None:
    t = text.replace("_", "").strip()
    m = re.fullmatch(r"(?:\d+)?'[sS]?([bodhBODH])([0-9a-fA-F]+)", t)
    if m:
        return int(m.group(2), {"b": 2, "o": 8, "d": 10, "h": 16}[m.group(1).lower()])
    return int(t) if t.isdigit() else None


def fmt(val: str | None, width: int) -> str:
    if val is None:
        return "?"
    if width <= 1 or len(val) == 1:
        return val
    if re.fullmatch(r"[01]+", val) and width > 4:
        return f"{width}'h{int(val, 2):0{(width + 3) // 4}x}"
    return f"{width}'b{val}"


def _dedupe(tr: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop re-recordings of an unchanged value (iverilog re-dumps on every reassignment)."""
    out: list[tuple[int, str]] = []
    for t, v in tr:
        if not out or out[-1][1] != v:
            out.append((t, v))
    return out


class DutWaves:
    def __init__(self, vcd_path: Path):
        v = Vcd(vcd_path)
        self.width: dict[str, int] = {}
        self.trace: dict[str, list[tuple[int, str]]] = {}
        self.expected: dict[str, list[tuple[int, str]]] = {}
        sample_clk = None
        for var in v.vars:
            if var.scope == DUT:
                self.trace[var.name] = _dedupe(v.changes[var.code])
                self.width[var.name] = var.width
            elif var.scope == EXP:
                self.expected[var.name] = _dedupe(v.changes[var.code])
            elif var.scope == SAMPLE and var.name == "clk":
                sample_clk = _dedupe(v.changes[var.code])
        self._times = {n: [t for t, _ in tr] for n, tr in {**self.trace, **{f"exp:{k}": x for k, x in self.expected.items()}}.items()}
        self.clocked = "clk" in self.trace  # does the DUT itself have a clock?
        clk = sample_clk or self.trace.get("clk") or []
        self.samples = [t for t, _ in clk if t > 0]
        self.posedges = [t for t, val in clk if val == "1"]

    def has(self, name: str) -> bool:
        return name in self.trace

    def _at(self, tr: list[tuple[int, str]], times: list[int], t: int) -> str | None:
        i = bisect.bisect_right(times, t) - 1
        return tr[i][1] if i >= 0 else None

    def value(self, name: str, t: int) -> str | None:
        return self._at(self.trace[name], self._times[name], t) if name in self.trace else None

    def expected_value(self, name: str, t: int) -> str | None:
        if name not in self.expected:
            return None
        return self._at(self.expected[name], self._times[f"exp:{name}"], t)

    def sampled(self, name: str, t: int) -> str | None:
        """Value the scoreboard sees at sample time t (settled just before the edge)."""
        return self.value(name, t - 1)

    def sampled_expected(self, name: str, t: int) -> str | None:
        return self.expected_value(name, t - 1)

    def cycle(self, t: int, change: bool = False) -> int:
        """Cycle of a sample at t (pre-edge value -> posedges strictly before t), or, with change=True,
        of a value change at t (posedges at or before t; same convention as meta.json cyc_act/cyc_fail)."""
        return (bisect.bisect_right if change else bisect.bisect_left)(self.posedges, t)

    def edge(self, t: int) -> str:
        i = bisect.bisect_left(self.posedges, t)
        return "posedge" if i < len(self.posedges) and self.posedges[i] == t else "negedge"

    def sample_times(self, t_end: int, k: int) -> list[int]:
        i = bisect.bisect_right(self.samples, t_end)
        return self.samples[max(0, i - k): i]

    def history(self, name: str, t0: int, t1: int) -> list[tuple[int, str]]:
        """Value at t0 followed by every change in (t0, t1]."""
        if name not in self.trace:
            return []
        tr, times = self.trace[name], self._times[name]
        out = [(t0, self.value(name, t0))]
        i = bisect.bisect_right(times, t0)
        while i < len(tr) and tr[i][0] <= t1:
            out.append(tr[i])
            i += 1
        return out

    def last_change(self, name: str, t: int) -> int | None:
        if name not in self.trace:
            return None
        i = bisect.bisect_right(self._times[name], t) - 1
        return self.trace[name][i][0] if i >= 0 else None
