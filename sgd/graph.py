"""Phase 2a: static signal drive graph from a (single-module) SystemVerilog DUT, via pyslang's semantic AST.

Nodes are signals (ports, nets, variables). Every assignment becomes a *driver* record on its target node
(line, statement text, guard path, comb/seq/init). Edges src -> dst are derived from drivers:
  dep="data": src is read on the right-hand side (or in an LHS index)
  dep="ctrl": src is read in an enclosing if/case condition (the guard)
Roles: "clock" (edge signal of a clocked block), "reset" (async edge signal, or a reset-like name used
in a guard of a clocked block), "fsm_state" (case subject or a register assigned symbolic state constants).

    graph = build_graph(src_text)        # -> JSON-serializable dict
"""
from __future__ import annotations

import re

import pyslang
from pyslang import ast, syntax

SymK, StmtK, ExprK = ast.SymbolKind, ast.StatementKind, ast.ExpressionKind
SIGNAL_KINDS = {SymK.Net, SymK.Variable}
RESET_NAME = re.compile(r"^(a?reset\w*|rst\w*|a?resetn|ar|r)$", re.I)


class _Builder:
    def __init__(self, src: str):
        self.src = src
        self.tree = syntax.SyntaxTree.fromText(src)
        self.comp = ast.Compilation()
        self.comp.addSyntaxTree(self.tree)
        diags = self.comp.getAllDiagnostics()
        self.diags = [l for l in pyslang.DiagnosticEngine.reportAll(self.tree.sourceManager, diags).splitlines()
                      if ": error:" in l or ": warning:" in l] if diags else []
        self.inst = self.comp.getRoot().topInstances[0]
        self.body = self.inst.body
        self.nodes: dict[str, dict] = {}
        self.constants: dict[str, str] = {}
        self.state_consts: set[str] = set()
        self.case_subjects: set[str] = set()
        self.blocks: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def line(self, node) -> int:
        return self.tree.sourceManager.getLineNumber(node.sourceRange.start)

    def line_end(self, node) -> int:
        return self.tree.sourceManager.getLineNumber(node.sourceRange.end)

    def text(self, node) -> str:
        r = node.sourceRange
        return " ".join(self.src[r.start.offset: r.end.offset].split())

    def refs(self, expr) -> list[str]:
        """Signal names read anywhere inside expr (constants/loop vars excluded), in order of appearance."""
        out: list[str] = []

        def cb(n):
            if getattr(n, "kind", None) == ExprK.NamedValue:
                s = n.symbol
                if s.kind in SIGNAL_KINDS and s.name in self.nodes and s.name not in out:
                    out.append(s.name)
            return True

        expr.visit(cb)
        return out

    def lhs(self, expr) -> tuple[list[str], list[str]]:
        """(target base names, signals read in LHS selectors)."""
        k = expr.kind
        if k == ExprK.NamedValue:
            return ([expr.symbol.name] if expr.symbol.name in self.nodes else []), []
        if k == ExprK.ElementSelect:
            t, r = self.lhs(expr.value)
            return t, r + self.refs(expr.selector)
        if k == ExprK.RangeSelect:
            t, r = self.lhs(expr.value)
            return t, r + self.refs(expr.left) + self.refs(expr.right)
        if k == ExprK.Concatenation:
            ts, rs = [], []
            for op in expr.operands:
                t, r = self.lhs(op)
                ts += t
                rs += r
            return ts, rs
        if k == ExprK.Conversion:
            return self.lhs(expr.operand)
        return [], self.refs(expr)

    # ---------------------------------------------------------------- symbols
    def collect_symbols(self) -> None:
        for m in self.body:
            if m.kind == SymK.Port:
                n = self._node(m.name)
                n["dir"] = str(m.direction).split(".")[-1].lower()
            elif m.kind in SIGNAL_KINDS:
                n = self._node(m.name)
                t = m.type
                n["type"] = str(t)
                n["memory"] = bool(t.isUnpackedArray)
                if n["memory"]:
                    n["width"] = t.arrayElementType.bitWidth
                    n["depth"] = t.fixedRange.width if t.hasFixedRange else None
                else:
                    n["width"] = t.bitWidth
            elif m.kind == SymK.Parameter:
                self.constants[m.name] = str(m.value)
            elif m.kind == SymK.TransparentMember:
                w = m.wrapped
                if w.kind == SymK.EnumValue:
                    self.constants[w.name] = str(w.value)
                    self.state_consts.add(w.name)
        # parameter lists with >= 3 names declared together are treated as state encodings
        params = [m for m in self.body if m.kind == SymK.Parameter]
        by_line: dict[int, list[str]] = {}
        for p in params:
            by_line.setdefault(self.tree.sourceManager.getLineNumber(p.location), []).append(p.name)
        for names in by_line.values():
            if len(names) >= 3:
                self.state_consts.update(names)

    def _node(self, name: str) -> dict:
        return self.nodes.setdefault(name, {"name": name, "dir": "internal", "width": 1, "type": "",
                                            "memory": False, "roles": [], "drivers": []})

    # ---------------------------------------------------------------- statements
    def collect_drivers(self) -> None:
        for m in self.body:
            if m.kind == SymK.ContinuousAssign:
                self.assign(m.assignment, [], "comb", block=f"assign@{self.line(m.assignment)}")
            elif m.kind == SymK.Net and m.initializer is not None:
                self.add_driver(m.name, self.text(m.initializer), self.refs(m.initializer), [], [],
                                "comb", self.tree.sourceManager.getLineNumber(m.location),
                                f"{m.name} = {self.text(m.initializer)}", f"wire@{m.name}")
            elif m.kind == SymK.ProceduralBlock:
                self.procedural(m)

    def procedural(self, m) -> None:
        pk = str(m.procedureKind).split(".")[-1]
        stmt = m.body
        kind, clock, async_rst = ("init" if pk == "Initial" else "comb"), None, []
        if stmt.kind == StmtK.Timed:
            events = []
            tc = stmt.timing
            if tc.kind == ast.TimingControlKind.SignalEvent:
                events = [tc]
            elif tc.kind == ast.TimingControlKind.EventList:
                events = list(tc.events)
            edged = [e for e in events if str(e.edge).split(".")[-1] in ("PosEdge", "NegEdge")]
            if edged:
                kind = "seq"
                names = [self.refs(e.expr)[0] for e in edged if self.refs(e.expr)]
                if names:
                    clock = names[0]
                    async_rst = names[1:]
            stmt = stmt.stmt
        if pk == "AlwaysFF":
            kind = "seq"
        line = self.line(m.body)
        blk = {"id": f"{pk.lower()}@{line}", "kind": kind, "line": line, "clock": clock, "async_reset": async_rst}
        self.blocks.append(blk)
        if clock:
            self._node(clock)["roles"].append("clock")
        for r in async_rst:
            self._node(r)["roles"].append("reset")
        self.walk(stmt, [], kind, blk["id"])

    def walk(self, s, guards: list[tuple[str, list[str], int]], kind: str, block: str) -> None:
        k = s.kind
        if k == StmtK.Block:
            self.walk(s.body, guards, kind, block)
        elif k == StmtK.List:
            for x in s.list:
                self.walk(x, guards, kind, block)
        elif k == StmtK.Timed:
            self.walk(s.stmt, guards, kind, block)
        elif k == StmtK.Conditional:
            cond = s.conditions[0].expr
            ct, ids, ln = self.text(cond), self.refs(cond), self.line(cond)
            if kind == "seq" and ids and all(RESET_NAME.match(i) for i in ids):
                for i in ids:
                    self._node(i)["roles"].append("reset")
            self.walk(s.ifTrue, guards + [(f"if ({ct})", ids, ln)], kind, block)
            if s.ifFalse is not None:
                self.walk(s.ifFalse, guards + [(f"!({ct})", ids, ln)], kind, block)
        elif k == StmtK.Case:
            e = s.expr
            et, ids = self.text(e), self.refs(e)
            self.case_subjects.update(ids)
            for item in s.items:
                labels = ", ".join(self.text(x) for x in item.expressions)
                lids = [i for x in item.expressions for i in self.refs(x)]
                self.walk(item.stmt, guards + [(f"case {et} == {labels}", ids + lids, self.line(item.stmt))],
                          kind, block)
            if s.defaultCase is not None:
                self.walk(s.defaultCase, guards + [(f"case {et} == default", ids, self.line(s.defaultCase))],
                          kind, block)
        elif k == StmtK.ExpressionStatement and s.expr.kind == ExprK.Assignment:
            self.assign(s.expr, guards, kind, block)
        elif k == StmtK.ForLoop:
            self.walk(s.body, guards + [(f"for ({self.text(s.stopExpr)})", [], self.line(s))], kind, block)

    def assign(self, a, guards, kind: str, block: str) -> None:
        targets, idx_reads = self.lhs(a.left)
        data = self.refs(a.right)
        for i in idx_reads:
            if i not in data:
                data.append(i)
        rhs = self.text(a.right)
        stmt = self.text(a)
        ln = self.line(a)
        for t in targets:
            self.add_driver(t, rhs, data, guards, idx_reads, kind, ln, stmt, block, nonblocking=a.isNonBlocking,
                            line_end=self.line_end(a))

    def add_driver(self, dst, rhs, data, guards, idx_reads, kind, line, stmt, block, nonblocking=False,
                   line_end=None) -> None:
        ctrl = []
        for _, ids, _ in guards:
            for i in ids:
                if i not in ctrl and i not in data:
                    ctrl.append(i)
        self._node(dst)["drivers"].append({
            "line": line, "line_end": line_end or line, "stmt": stmt, "rhs": rhs, "kind": kind, "block": block,
            "guard": " && ".join(g for g, _, _ in guards), "guard_lines": sorted({l for _, _, l in guards}),
            "data": data, "ctrl": ctrl, "nonblocking": nonblocking,
            "const_rhs": rhs if rhs in self.constants else None,
        })

    # ---------------------------------------------------------------- finish
    def finish(self) -> dict:
        for name in self.case_subjects:
            self._node(name)["roles"].append("fsm_state")
        for n in self.nodes.values():
            if not n["memory"] and any(d["kind"] == "seq" and d["const_rhs"] in self.state_consts for d in n["drivers"]):
                n["roles"].append("fsm_state")
            n["roles"] = sorted(set(n["roles"]))
        edges = []
        for n in self.nodes.values():
            seen = set()
            if n["dir"] == "internal" and not n["drivers"]:
                used = any(n["name"] in d["data"] + d["ctrl"] for m in self.nodes.values() for d in m["drivers"])
                if not used:
                    n["roles"].append("unused")
            for d in n["drivers"]:
                for dep in ("data", "ctrl"):
                    for s in d[dep]:
                        key = (s, dep, d["line"])
                        if key in seen:
                            continue
                        seen.add(key)
                        edges.append({"src": s, "dst": n["name"], "dep": dep, "kind": d["kind"],
                                      "line": d["line"], "guard": d["guard"]})
        return {"module": self.inst.name, "nodes": self.nodes, "edges": edges, "constants": self.constants,
                "state_constants": sorted(self.state_consts), "blocks": self.blocks, "diagnostics": self.diags}


def build_graph(src: str) -> dict:
    b = _Builder(src)
    b.collect_symbols()
    b.collect_drivers()
    return b.finish()


# ------------------------------------------------------------------- queries on the JSON graph

def fanin(g: dict, sig: str) -> list[dict]:
    return [e for e in g["edges"] if e["dst"] == sig]


def fanout(g: dict, sig: str) -> list[dict]:
    return [e for e in g["edges"] if e["src"] == sig]


def backward_cone(g: dict, roots: list[str], skip_roles=("clock",)) -> dict[str, int]:
    """Signals reachable backward from roots -> hop distance (BFS over data+ctrl edges)."""
    dist = {r: 0 for r in roots}
    frontier = list(roots)
    while frontier:
        nxt = []
        for s in frontier:
            for e in fanin(g, s):
                src = e["src"]
                if src in dist or set(g["nodes"].get(src, {}).get("roles", [])) & set(skip_roles):
                    continue
                dist[src] = dist[s] + 1
                nxt.append(src)
        frontier = nxt
    return dist
