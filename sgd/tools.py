"""Phase 2d: the treatment agent's tool surface over one bug: drive graph + pre-failure value windows.

Every tool returns plain text sized for a 16k-context model. Only bugs/<id>/public/ is read.
Ablation switches: with_windows=False (graph only, no values), with_edges=False (windows only, no
drivers/fan-in/fan-out).

    env = GraphEnv(Path("bugs/bug111"), k=8)
    print(env.failure_report()); print(env.get_node("state"))
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import SimResult, parse_tb_log
from .waves import DutWaves, const_to_int, fmt

MAX_HISTORY = 40


class GraphEnv:
    def __init__(self, bug_dir: Path, k: int = 8, with_windows: bool = True, with_edges: bool = True,
                 hide_reset: bool = True):
        pub = bug_dir / "public"
        self.src = (pub / "buggy.sv").read_text()
        self.spec = (pub / "spec.txt").read_text()
        self.g = json.loads((pub / "graph.json").read_text())
        self.w = DutWaves(pub / "dut_wave.vcd")
        self.k, self.with_windows, self.with_edges, self.hide_reset = k, with_windows, with_edges, hide_reset
        tb = parse_tb_log((pub / "tb_log.txt").read_text(), SimResult(compiled=True))
        self.failing = sorted(tb.out_first_mismatch, key=tb.out_first_mismatch.get)
        self.first_mismatch = tb.out_first_mismatch
        self.t_fail = tb.t_fail
        self.mismatch_count = tb.out_mismatch_count
        self.state_names = {}
        for c in self.g["state_constants"]:
            v = const_to_int(self.g["constants"].get(c, ""))
            if v is not None:
                self.state_names.setdefault(v, c)
        self.visited: list[str] = []  # trajectory: which nodes the agent inspected

    # ------------------------------------------------------------------ formatting
    def _node(self, sig: str) -> dict | None:
        return self.g["nodes"].get(sig)

    def _val(self, sig: str, t: int, sampled: bool = True) -> str:
        """Formatted value; sampled=True gives the pre-edge value the scoreboard compares."""
        n = self._node(sig)
        w = self.w.width.get(sig, n["width"] if n else 1)
        raw = self.w.sampled(sig, t) if sampled else self.w.value(sig, t)
        s = fmt(raw, w)
        if n and "fsm_state" in n["roles"] and raw and set(raw) <= {"0", "1"}:
            name = self.state_names.get(int(raw, 2))
            if name:
                s += f" ({name})"
        return s

    def _tlabel(self, t: int, change: bool = False) -> str:
        if change:
            return f"t={t} cyc={self.w.cycle(t, change=True)}"
        return f"t={t} cyc={self.w.cycle(t)} {self.w.edge(t)}"

    def _is_reset(self, sig: str) -> bool:
        n = self._node(sig)
        return bool(n and "reset" in n["roles"])

    # ------------------------------------------------------------------ tools
    def failure_report(self) -> str:
        lines = [f"Testbench result: {sum(self.mismatch_count.values())} output mismatches."]
        for o in self.failing:
            t = self.first_mismatch[o]
            exp = self.w.sampled_expected(o, t)
            line = f"- output '{o}': {self.mismatch_count[o]} mismatches, first at {self._tlabel(t)}"
            if self.with_windows:
                line += f": DUT={self._val(o, t)} expected={fmt(exp, self.w.width.get(o, 1))}"
            lines.append(line)
        lines.append(f"T_fail = {self.t_fail} ps (first mismatch). Values are sampled just before each clock edge, "
                     "as the testbench compares them. Start debugging from the failing output(s).")
        return "\n".join(lines)

    def list_signals(self) -> str:
        rows = []
        for name, n in self.g["nodes"].items():
            if "unused" in n["roles"]:
                continue
            extra = f" depth={n.get('depth')}" if n["memory"] else ""
            roles = f" roles={','.join(n['roles'])}" if n["roles"] else ""
            rows.append(f"{name}: {n['dir']}, {n['width']} bit{extra}{roles}")
        return "\n".join(rows)

    def get_node(self, sig: str) -> str:
        n = self._node(sig)
        if not n:
            return f"No signal named '{sig}'. Known: {', '.join(self.g['nodes'])}"
        self.visited.append(sig)
        out = [f"signal {sig}: {n['dir']}, {n['width']} bit" + (f", memory depth {n.get('depth')}" if n["memory"] else "")
               + (f", roles: {', '.join(n['roles'])}" if n["roles"] else "")]
        if self.with_edges:
            if n["drivers"]:
                out.append("drivers:")
                for d in n["drivers"]:
                    g = f"   when: {d['guard']}" if d["guard"] else ""
                    out.append(f"  L{d['line']} [{d['kind']}] {d['stmt']};{g}")
            elif n["dir"] == "in":
                out.append("drivers: primary input")
            fi = self._fanin_names(sig)
            fo = sorted({e["dst"] for e in self.g["edges"] if e["src"] == sig})
            out.append(f"fan-in : {', '.join(fi) or '-'}")
            out.append(f"fan-out: {', '.join(fo) or '-'}")
        if self.with_windows:
            out.append(self._window_table([sig]))
        return "\n".join(out)

    def _fanin_names(self, sig: str) -> list[str]:
        names = []
        for e in self.g["edges"]:
            if e["dst"] != sig:
                continue
            tag = f"{e['src']}({e['dep']})"
            if self.hide_reset and self._is_reset(e["src"]):
                tag = f"{e['src']}(reset)"
            if tag not in names:
                names.append(tag)
        return names

    def _window_table(self, sigs: list[str]) -> str:
        if self.t_fail is None:
            return "no failure time"
        ts = self.w.sample_times(self.t_fail, self.k)
        sigs = [s for s in sigs if self.w.has(s)]
        if not sigs:
            return "values: not in waveform (memory or optimized away)"
        head = f"values in the last {len(ts)} samples up to T_fail:"
        rows = [head]
        for t in ts:
            vals = "  ".join(f"{s}={self._val(s, t)}" for s in sigs)
            mark = "  <- T_fail" if t == self.t_fail else ""
            rows.append(f"  {self._tlabel(t)}: {vals}{mark}")
        return "\n".join(rows)

    def driven_by(self, sig: str) -> str:
        if not self.with_edges:
            return "tool unavailable in this configuration"
        n = self._node(sig)
        if not n:
            return f"No signal named '{sig}'."
        self.visited.append(sig)
        rows = [f"{sig} is driven by:"]
        seen = set()
        for e in self.g["edges"]:
            if e["dst"] != sig or (e["src"], e["dep"]) in seen:
                continue
            seen.add((e["src"], e["dep"]))
            lines = sorted({x["line"] for x in self.g["edges"] if x["dst"] == sig and x["src"] == e["src"]})
            tag = "reset" if self.hide_reset and self._is_reset(e["src"]) else e["dep"]
            row = f"  {e['src']} [{tag}, {e['kind']}] at L{','.join(map(str, lines))}"
            if self.with_windows and self.t_fail is not None and self.w.has(e["src"]):
                lc = self.w.last_change(e["src"], self.t_fail)
                row += f"; at T_fail = {self._val(e['src'], self.t_fail)} (last changed t={lc})"
            rows.append(row)
        if len(rows) == 1:
            rows.append("  (nothing: primary input or constant-only drivers)")
        return "\n".join(rows)

    def drives(self, sig: str) -> str:
        if not self.with_edges:
            return "tool unavailable in this configuration"
        if not self._node(sig):
            return f"No signal named '{sig}'."
        self.visited.append(sig)
        rows = [f"{sig} drives:"]
        seen = set()
        for e in self.g["edges"]:
            if e["src"] == sig and (e["dst"], e["dep"]) not in seen:
                seen.add((e["dst"], e["dep"]))
                rows.append(f"  {e['dst']} [{e['dep']}, {e['kind']}] at L{e['line']}")
        return "\n".join(rows) if len(rows) > 1 else f"{sig} drives nothing (output or unused)."

    def expand_history(self, sig: str, t0: int, t1: int) -> str:
        if not self.with_windows:
            return "tool unavailable in this configuration"
        if not self.w.has(sig):
            return f"'{sig}' has no waveform (unknown, memory, or optimized away)."
        self.visited.append(sig)
        h = self.w.history(sig, int(t0), int(t1))
        rows = [f"{sig} from t={t0} to t={t1} (value at start, then the time of every change):"]
        for t, v in h[:MAX_HISTORY]:
            n = self._node(sig)
            w = self.w.width.get(sig, 1)
            rows.append(f"  {self._tlabel(t, change=True)}: {self._val(sig, t, sampled=False) if v is not None else fmt(v, w)}")
        if len(h) > MAX_HISTORY:
            rows.append(f"  ... {len(h) - MAX_HISTORY} more changes truncated; narrow the range")
        return "\n".join(rows)

    def get_spec(self) -> str:
        return self.spec

    def read_rtl(self) -> str:
        return "\n".join(f"{i:3d}| {l}" for i, l in enumerate(self.src.splitlines(), 1))
