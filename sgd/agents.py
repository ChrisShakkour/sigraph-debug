"""Phase 3: the agent arms and the shared JSON-action ReAct loop.

Arms (same spec, same testbench log, same final-answer schema, same step cap; only the tools differ):
  A        raw dump: buggy RTL + a raw VCD slice ending at T_fail in one prompt, answer in one shot
  B        standard agent: read_file, grep, list_signals, get_value(signal, t0, t1) over the VCD
  T        graph + windows: the Phase 2 tools (sgd/tools.py)
  T-win    graph only: same tools with value windows off (no values anywhere)
  T-graph  windows only: same tools with drivers/edges off

Only bugs/<id>/public/ is read. The agent answers with
  {"action": "final", "root_cause_signal", "root_cause_line", "activation_time", "explanation", "fixed_module"}

    rec = run_agent("T", Path("bugs/bug111"), Ollama())
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .common import Vcd
from .llm import NUM_CTX, Ollama
from .tools import GraphEnv
from .waves import DutWaves, fmt

ARMS = ("A", "B", "T", "T-win", "T-graph")
MAX_STEPS = 15
OBS_CHARS = 3000             # one observation is truncated to this many characters
PROMPT_BUDGET = NUM_CTX - 2048 - 512   # tokens left for the prompt after output + safety margin
RAW_VCD_CHARS = 24000        # arm A: size of the raw VCD slice

FINAL_SCHEMA = """When you are confident, answer with
{"thought": "...", "action": "final",
 "root_cause_signal": "<the signal whose driver is wrong>",
 "root_cause_line": <line number of the buggy statement in buggy.sv>,
 "activation_time": <simulation time when that signal first took a wrong value>,
 "explanation": "<what is wrong, why it makes the output fail, and the fix>",
 "fixed_module": "<the complete corrected module TopModule, as SystemVerilog source>"}"""

SYSTEM = """You are an expert RTL verification engineer debugging a SystemVerilog module named TopModule.
The module was written from the specification below and fails its testbench. There is exactly one bug:
one small edit (an operator, a constant, a condition, a state, an index, or a reset style) breaks it.
Find the root cause, which is not necessarily on the failing output itself, and fix it.

The testbench compares the DUT outputs with a reference model on both edges of clk, using the values
settled just before each edge. Times are in ps.

Reply with ONE JSON object per message and nothing else.
{tools}
{final}"""

TOOL_STYLE = """To use a tool reply {"thought": "<short reasoning>", "action": "<tool>", "args": {...}}.
You will get the tool output back. Tools:
"""

TOOLS = {
    "B": {
        "read_file": ('{"path": "buggy.sv" | "spec.txt" | "tb_log.txt"}', "file contents (buggy.sv with line numbers)"),
        "grep": ('{"pattern": "<python regex>"}', "matching lines of buggy.sv with line numbers"),
        "list_signals": ("{}", "every signal in the waveform with its width, plus expected.<output> for the reference outputs"),
        "get_value": ('{"signal": "<name>", "t0": <time>, "t1": <time>}', "value at t0 and every change up to t1"),
    },
    "T": {
        "failure_report": ("{}", "failing outputs, first mismatch time and values"),
        "get_node": ('{"signal": "<name>"}', "drivers (line, statement, guard), fan-in, fan-out, and the values in the last samples up to the failure"),
        "driven_by": ('{"signal": "<name>"}', "signals feeding this one, with edge kind, lines and value at failure"),
        "drives": ('{"signal": "<name>"}', "signals this one feeds"),
        "expand_history": ('{"signal": "<name>", "t0": <time>, "t1": <time>}', "value at t0 and every change up to t1"),
        "list_signals": ("{}", "all signals with width and role"),
        "get_spec": ("{}", "the specification"),
        "read_rtl": ("{}", "buggy.sv with line numbers"),
    },
}
T_WIN_TOOLS = ("failure_report", "get_node", "driven_by", "drives", "list_signals", "get_spec", "read_rtl")
T_GRAPH_TOOLS = ("failure_report", "get_node", "expand_history", "list_signals", "get_spec", "read_rtl")


# ---------------------------------------------------------------------------------------------- envs
class BaseEnv:
    tools: dict[str, tuple[str, str]] = {}
    signal_arg = "signal"
    ALIASES = ("signal", "sig", "name", "signal_name", "node")

    def call(self, name: str, args: dict) -> str:
        args = dict(args) if isinstance(args, dict) else {}
        for a in self.ALIASES:
            if a in args and a != self.signal_arg:
                args.setdefault(self.signal_arg, args.pop(a))
        if name not in self.tools:
            return f"Unknown action '{name}'. Available: {', '.join(self.tools)}, final."
        try:
            return str(getattr(self, "t_" + name)(**args))
        except TypeError as e:
            return f"Bad arguments for {name}: {e}. Expected {self.tools[name][0]}."
        except Exception as e:  # noqa: BLE001 - tool errors are shown to the agent
            return f"Tool error: {type(e).__name__}: {e}"


def _numbered(src: str) -> str:
    return "\n".join(f"{i:3d}| {l}" for i, l in enumerate(src.splitlines(), 1))


class StdEnv(BaseEnv):
    """Arm B: files, grep and a plain waveform query. No graph, no sampling semantics."""
    tools = TOOLS["B"]

    def __init__(self, bug_dir: Path):
        pub = bug_dir / "public"
        self.files = {p: (pub / p).read_text() for p in ("buggy.sv", "spec.txt", "tb_log.txt")}
        self.w = DutWaves(pub / "dut_wave.vcd")
        self.visited: list[str] = []

    def t_read_file(self, path: str) -> str:
        path = Path(path).name
        if path not in self.files:
            return f"No such file. Available: {', '.join(self.files)}"
        return _numbered(self.files[path]) if path.endswith(".sv") else self.files[path]

    def t_grep(self, pattern: str) -> str:
        rx = re.compile(pattern)
        hits = [f"{i:3d}| {l}" for i, l in enumerate(self.files["buggy.sv"].splitlines(), 1) if rx.search(l)]
        return "\n".join(hits) or "no match"

    def t_list_signals(self) -> str:
        rows = [f"{n}: {self.w.width[n]} bit" for n in self.w.trace]
        rows += [f"expected.{n}: reference value of output {n}" for n in self.w.expected]
        return "\n".join(rows)

    def t_get_value(self, signal: str, t0: int, t1: int) -> str:
        if signal.startswith("expected."):
            name = signal.split(".", 1)[1]
            tr = self.w.expected.get(name)
            w = self.w.width.get(name, 1)
        else:
            name, tr = signal, self.w.trace.get(signal)
            w = self.w.width.get(signal, 1)
        if tr is None:
            return f"No signal '{signal}' in the waveform. Use list_signals."
        self.visited.append(name)
        t0, t1 = int(t0), int(t1)
        start = None
        for t, v in tr:
            if t <= t0:
                start = v
        rows = [f"{signal} at t={t0}: {fmt(start, w)}"]
        ch = [(t, v) for t, v in tr if t0 < t <= t1]
        rows += [f"  t={t}: {fmt(v, w)}" for t, v in ch[:40]]
        if len(ch) > 40:
            rows.append(f"  ... {len(ch) - 40} more changes; narrow the range")
        return "\n".join(rows)


class GraphArmEnv(BaseEnv):
    """Arms T, T-win, T-graph: thin wrapper over sgd.tools.GraphEnv."""
    signal_arg = "sig"

    def __init__(self, bug_dir: Path, arm: str, k: int = 8):
        self.g = GraphEnv(bug_dir, k=k, with_windows=(arm != "T-win"), with_edges=(arm != "T-graph"))
        names = {"T": tuple(TOOLS["T"]), "T-win": T_WIN_TOOLS, "T-graph": T_GRAPH_TOOLS}[arm]
        self.tools = {n: TOOLS["T"][n] for n in names}

    @property
    def visited(self) -> list[str]:
        return self.g.visited

    def __getattr__(self, item):
        if item.startswith("t_"):
            return getattr(self.g, item[2:])
        raise AttributeError(item)


# ---------------------------------------------------------------------------------------------- arm A
def raw_vcd_slice(vcd_path: Path, t_end: int, budget: int = RAW_VCD_CHARS) -> str:
    """Raw VCD text (DUT + expected outputs + TB clock): header, the state at the slice start, then every
    change up to t_end; the start is moved as late as needed to fit the budget."""
    v = Vcd(vcd_path)
    keep = [x for x in v.vars if x.scope.split(".")[-1] in ("top_module1", "tb_expected", "tb_sample")
            and not (x.width == 32 and x.name[:1].isupper() and len(v.changes[x.code]) <= 1)]
    head, scope = [], None
    for x in keep:
        sc = x.scope.split(".")[-1]
        if sc != scope:
            if scope:
                head.append("$upscope $end")
            head.append(f"$scope module {sc} $end")
            scope = sc
        head.append(f"$var wire {x.width} {x.code} {x.name} $end")
    head += ["$upscope $end", "$enddefinitions $end"]

    def line(code: str, val: str, w: int) -> str:
        return f"{val}{code}" if w == 1 else f"b{val} {code}"

    codes = {x.code: x.width for x in keep}
    events: dict[int, list[str]] = {}
    for c, w in codes.items():
        for t, val in v.changes[c]:
            if t <= t_end:
                events.setdefault(t, []).append(line(c, val, w))
    times = sorted(events)
    body: list[str] = []
    size = sum(len(h) + 1 for h in head) + 200
    start = 0
    for i in range(len(times) - 1, -1, -1):
        blk = [f"#{times[i]}"] + events[times[i]]
        n = sum(len(b) + 1 for b in blk)
        if size + n + 30 * len(codes) > budget:
            start = i + 1
            break
        body = blk + body
        size += n
    if start > 0:  # state just before the slice
        t0 = times[start]
        init = []
        for c, w in codes.items():
            prev = [val for t, val in v.changes[c] if t < t0]
            if prev:
                init.append(line(c, prev[-1], w))
        body = [f"$comment slice starts at #{t0}; values before it: $end", "$dumpvars"] + init + ["$end"] + body
    return "\n".join(head + body)


# ---------------------------------------------------------------------------------------------- loop
def parse_json(text: str) -> dict | None:
    """First balanced {...} object in the text, tolerating code fences and trailing prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    while i >= 0:
        depth, in_s, esc = 0, False, False
        for j in range(i, len(t)):
            ch = t[j]
            if in_s:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_s = False
                continue
            if ch == '"':
                in_s = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(t[i:j + 1], strict=False)
                        if isinstance(d, dict):
                            return d
                    except json.JSONDecodeError:
                        break
                    break
        i = t.find("{", i + 1)
    return None


def _tokens(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages) // 3  # conservative chars-per-token for code/VCD


def _fit(messages: list[dict]) -> None:
    """Elide the oldest tool observations until the prompt fits the context budget."""
    for m in messages[2:]:
        if _tokens(messages) <= PROMPT_BUDGET:
            return
        if m["role"] == "user" and not m["content"].startswith("[elided"):
            m["content"] = "[elided: older tool output removed to fit the context]"


def system_prompt(arm: str, env: BaseEnv | None) -> str:
    if arm == "A":
        tools = "You have no tools: answer directly with the final JSON."
    else:
        tools = TOOL_STYLE + "\n".join(f"- {n} {a}: {d}" for n, (a, d) in env.tools.items())
        tools += f"\nYou have at most {MAX_STEPS} messages including the final answer."
    return SYSTEM.replace("{tools}", tools).replace("{final}", FINAL_SCHEMA)


def first_message(arm: str, bug_dir: Path, env) -> str:
    pub = bug_dir / "public"
    spec = (pub / "spec.txt").read_text().strip()
    tb_log = (pub / "tb_log.txt").read_text().strip()
    parts = [f"Specification:\n{spec}", f"Testbench log:\n{tb_log}"]
    if arm == "A":
        t_fail = GraphEnv(bug_dir).t_fail or 0
        parts.append(f"buggy.sv:\n{_numbered((pub / 'buggy.sv').read_text())}")
        parts.append("Waveform (VCD; scope top_module1 is the DUT, tb_expected holds the reference outputs, "
                     f"tb_sample.clk is the testbench clock), up to the first mismatch at t={t_fail}:\n"
                     + raw_vcd_slice(pub / "dut_wave.vcd", t_fail))
    elif arm.startswith("T"):
        parts.append("Failure report:\n" + env.call("failure_report", {}))
    return "\n\n".join(parts)


def run_agent(arm: str, bug_dir: Path, llm: Ollama, max_steps: int = MAX_STEPS) -> dict:
    env = None if arm == "A" else StdEnv(bug_dir) if arm == "B" else GraphArmEnv(bug_dir, arm)
    messages = [{"role": "system", "content": system_prompt(arm, env)},
                {"role": "user", "content": first_message(arm, bug_dir, env)}]
    rec = {"arm": arm, "bug_id": bug_dir.name, "model": llm.model, "steps": [], "final": None,
           "prompt_tokens": 0, "output_tokens": 0, "llm_seconds": 0.0, "parse_errors": 0, "stop": None}
    t_start = time.time()
    limit = 2 if arm == "A" else max_steps  # A: one shot, plus one retry if the reply is not valid JSON
    for step in range(limit):
        if step == limit - 1 and arm != "A":
            messages.append({"role": "user", "content": "Step limit reached. Reply now with the final JSON answer."})
        _fit(messages)
        r = llm.chat(messages)
        rec["prompt_tokens"] += r.prompt_tokens
        rec["output_tokens"] += r.output_tokens
        rec["llm_seconds"] += r.seconds
        act = parse_json(r.text)
        s = {"step": step, "raw": r.text, "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
             "seconds": round(r.seconds, 2)}
        rec["steps"].append(s)
        messages.append({"role": "assistant", "content": r.text})
        if act is None:
            rec["parse_errors"] += 1
            messages.append({"role": "user", "content": "Your reply was not a single JSON object. Reply with exactly one JSON object."})
            continue
        name = str(act.get("action", "")).strip()
        s["action"] = name
        if name == "final" or (arm == "A" and "root_cause_line" in act):
            rec["final"] = act
            rec["stop"] = "final"
            break
        s["args"] = act.get("args") or {}
        obs = env.call(name, s["args"]) if env else "No tools are available. Reply with the final JSON."
        if len(obs) > OBS_CHARS:
            obs = obs[:OBS_CHARS] + "\n... [truncated]"
        s["observation"] = obs
        messages.append({"role": "user", "content": obs})
    else:
        rec["stop"] = "step_limit"
    rec["n_steps"] = len(rec["steps"])
    rec["visited"] = list(env.visited) if env else []
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    return rec
