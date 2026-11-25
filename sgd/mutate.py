"""Token-level single-point mutation engine for verilog-eval refs.

Works on the *normalized* ref text (ref_as_dut), so the line numbers recorded as ground truth are
the line numbers of the buggy.sv the agents see. Each Mutant is one minimal edit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<sized>\d*\s*'[sS]?[bBoOdDhH]\s*[0-9a-fA-FxXzZ_?]+)
  | (?P<unsz>'[01xXzZ])
  | (?P<num>\d+(?:\.\d+)?)
  | (?P<sys>\$[A-Za-z_]\w*)
  | (?P<id>[A-Za-z_][\w$]*)
  | (?P<str>"[^"]*")
  | (?P<op>===|!==|==|!=|<=|>=|&&|\|\||<<<|>>>|<<|>>|\*\*|~\^|\^~|~&|~\||->|::|\+\+|--|\+:|-:|.)
""", re.X | re.S)

DECL_KW = {"input", "output", "inout", "wire", "reg", "logic", "integer", "parameter", "localparam",
           "typedef", "genvar", "bit", "int"}
ITEM_KW = {"always", "always_ff", "always_comb", "always_latch", "assign", "initial", "endmodule",
           "generate", "endgenerate"} | DECL_KW
RESET_RE = re.compile(r"^(a?reset\w*|rst\w*|ar|r|resetn|aresetn)$")
OPERAND_END = {"id", "num", "sized", "unsz"}


@dataclass
class Tok:
    kind: str
    text: str
    start: int
    line: int          # 1-based
    pdepth: int = 0    # paren depth BEFORE this token
    bdepth: int = 0    # bracket depth BEFORE this token
    region: str = ""   # "header" | "decl" | "seq" | "comb" | "other"
    reset_ctx: bool = False


@dataclass
class Mutant:
    op: str
    category: str      # control | datapath | state
    line: int
    old: str           # original line text
    new: str           # mutated line text (single-line edits) or "" for multi-line
    src: str           # full mutated source
    note: str = ""

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("src")
        return d


def tokenize(src: str) -> list[Tok]:
    toks, line = [], 1
    for m in TOKEN_RE.finditer(src):
        kind = m.lastgroup
        if kind == "ws":
            line += m.group().count("\n")
            continue
        toks.append(Tok(kind, m.group(), m.start(), line))
        line += m.group().count("\n")
    return toks


def annotate(src: str, toks: list[Tok]) -> None:
    """Fill paren/bracket depth, module-item region (header/decl/seq/comb) and reset context."""
    lines = src.splitlines()
    # header ends at the first ');' after 'module'
    hdr_end_line = next(i + 1 for i, l in enumerate(lines) if re.search(r"\)\s*;", l))
    p = b = 0
    region = "header"
    for i, t in enumerate(toks):
        t.pdepth, t.bdepth = p, b
        if t.text == "(":
            p += 1
        elif t.text == ")":
            p -= 1
        elif t.text == "[":
            b += 1
        elif t.text == "]":
            b -= 1
        if t.line <= hdr_end_line:
            t.region = "header"
            continue
        first_on_line = i == 0 or toks[i - 1].line != t.line
        if first_on_line and p == 0 and t.text in ITEM_KW:
            if t.text in DECL_KW:
                region = "decl"
            elif t.text in ("always_ff",):
                region = "seq"
            elif t.text == "always":
                hdr = " ".join(x.text for x in toks[i:i + 12] if x.line == t.line)
                region = "seq" if re.search(r"\b(posedge|negedge)\b", hdr) else "comb"
            elif t.text in ("always_comb", "always_latch", "assign"):
                region = "comb"
            else:
                region = "other"
        t.region = region
    # reset context: statement(s) guarded by `if (<reset-like id>)` inside seq regions
    for i, t in enumerate(toks):
        if t.text == "if" and t.region == "seq" and i + 1 < len(toks) and toks[i + 1].text == "(":
            j = match_paren(toks, i + 1)
            ids = [x.text for x in toks[i + 2:j] if x.kind == "id"]
            if ids and all(RESET_RE.match(x) for x in ids):
                k = j + 1
                end = stmt_end(toks, k)
                for x in toks[i:end + 1]:
                    x.reset_ctx = True


def match_paren(toks: list[Tok], i: int) -> int:
    opn, cls = toks[i].text, {"(": ")", "[": "]", "{": "}"}[toks[i].text]
    d = 0
    for j in range(i, len(toks)):
        if toks[j].text == opn:
            d += 1
        elif toks[j].text == cls:
            d -= 1
            if d == 0:
                return j
    return len(toks) - 1


def stmt_end(toks: list[Tok], k: int) -> int:
    """Index of last token of the statement starting at k (handles begin/end and case/endcase)."""
    if k >= len(toks):
        return len(toks) - 1
    if toks[k].text in ("begin", "case", "casez", "casex"):
        opn = "begin" if toks[k].text == "begin" else None
        d = 0
        for j in range(k, len(toks)):
            tx = toks[j].text
            if (opn and tx == "begin") or (not opn and tx in ("case", "casez", "casex")):
                d += 1
            elif (opn and tx == "end") or (not opn and tx == "endcase"):
                d -= 1
                if d == 0:
                    return j
        return len(toks) - 1
    d = 0
    for j in range(k, len(toks)):
        if toks[j].text in "([{":
            d += 1
        elif toks[j].text in ")]}":
            d -= 1
        elif toks[j].text == ";" and d == 0:
            return j
    return len(toks) - 1


def state_names(src: str) -> list[str]:
    """Symbolic state names: enum members, or a parameter/localparam list with >= 3 names."""
    names: list[str] = []
    for m in re.finditer(r"enum[^{]*\{([^}]*)\}", src):
        names += [n.split("=")[0].strip() for n in m.group(1).split(",") if n.strip()]
    if not names:
        for m in re.finditer(r"\b(?:parameter|localparam)\b[^;]*;", src):
            ns = re.findall(r"([A-Za-z_]\w*)\s*=", m.group())
            if len(ns) >= 3:
                names += ns
    return names


def _replace(src: str, toks: list[Tok], i: int, new: str, j: int | None = None) -> str:
    j = i if j is None else j
    return src[: toks[i].start] + new + src[toks[j].start + len(toks[j].text):]


def _drop(src: str, toks: list[Tok], i: int, j: int) -> str:
    """Delete tokens i..j together with the whitespace before token i (no formatting trace)."""
    a = toks[i - 1].start + len(toks[i - 1].text)
    return src[:a] + src[toks[j].start + len(toks[j].text):]


def _sized_pm1(text: str) -> list[str]:
    m = re.match(r"(\d*)\s*'([sS]?)([bBoOdDhH])\s*([0-9a-fA-F_]+)$", text)
    if not m:
        return []
    w, s, base, digits = m.groups()
    radix = {"b": 2, "o": 8, "d": 10, "h": 16}[base.lower()]
    v = int(digits.replace("_", ""), radix)
    width = int(w) if w else 32
    out = []
    for nv in ((v + 1) % (1 << width), (v - 1) % (1 << width)):
        if nv == v:
            continue
        lit = {2: format(nv, "b"), 8: format(nv, "o"), 10: str(nv), 16: format(nv, "x")}[radix]
        out.append(f"{w}'{s}{base}{lit}")
    return list(dict.fromkeys(out))[:1] if width == 1 else list(dict.fromkeys(out))


def generate(src: str) -> list[Mutant]:
    toks = tokenize(src)
    annotate(src, toks)
    lines = src.splitlines()
    states = state_names(src)
    muts: list[Mutant] = []

    def add(op: str, cat: str, t: Tok, new_src: str, note: str = "") -> None:
        if new_src == src:
            return
        nl = new_src.splitlines()
        new_line = nl[t.line - 1] if len(nl) == len(lines) else ""
        if t.reset_ctx:
            cat = "state"
        muts.append(Mutant(op, cat, t.line, lines[t.line - 1], new_line, new_src, note))

    body = [i for i, t in enumerate(toks) if t.region in ("seq", "comb", "other")]
    for i in body:
        t = toks[i]
        prev = toks[i - 1] if i else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        binary = prev is not None and (prev.kind in OPERAND_END or prev.text in (")", "]", "}"))
        seq = t.region == "seq"

        # --- comparison / logic operators
        if t.text in ("==", "!=", "===", "!=="):
            add("eq_neq", "control", t, _replace(src, toks, i, {"==": "!=", "!=": "==", "===": "!==", "!==": "==="}[t.text]))
        elif t.text in ("<", ">", ">=") or (t.text == "<=" and t.pdepth > 0):
            for r in {"<": ["<=", ">"], ">": [">=", "<"], ">=": [">"], "<=": ["<"]}[t.text]:
                add("rel_op", "datapath", t, _replace(src, toks, i, r))
        elif t.text in ("&&", "||"):
            add("logic_op", "control", t, _replace(src, toks, i, "||" if t.text == "&&" else "&&"))
            # drop the right operand if it is simple:  X && Y  ->  X
            if nxt and nxt.kind == "id" and (i + 2 >= len(toks) or toks[i + 2].text in (")", ";", "&&", "||", "?")):
                add("drop_term", "control", t, _drop(src, toks, i, i + 1))
            elif nxt and nxt.text == "!" and i + 2 < len(toks) and toks[i + 2].kind == "id":
                add("drop_term", "control", t, _drop(src, toks, i, i + 2))
        elif t.text in ("&", "|", "^") and binary:
            for r in {"&": ["|"], "|": ["&"], "^": ["|"]}[t.text]:
                add("bit_op", "datapath", t, _replace(src, toks, i, r))
        elif t.text in ("+", "-") and binary and t.bdepth == 0:
            add("arith_op", "state" if seq else "datapath", t, _replace(src, toks, i, "-" if t.text == "+" else "+"))
        elif t.text in ("!", "~") and not binary and nxt and nxt.text not in ("=", "&", "|", "^"):
            add("drop_not", "control" if t.text == "!" else "datapath", t, _replace(src, toks, i, ""))

        # --- constants / indices
        elif t.kind in ("num", "sized") and t.region != "decl" and prev and prev.text not in ("#",):
            first_on_line = prev.line != t.line
            if first_on_line and nxt and nxt.text in (":", ","):
                continue  # case label
            if prev.text == "[" and nxt and nxt.text == "]" and t.kind == "num":
                v = int(t.text)
                add("index_shift", "datapath", t, _replace(src, toks, i, str(v - 1 if v > 0 else v + 1)))
            elif t.bdepth == 0:
                cands = _sized_pm1(t.text) if t.kind == "sized" else (
                    [str(int(t.text) + 1)] + ([str(int(t.text) - 1)] if t.text != "0" else []) if "." not in t.text else [])
                for c in cands:
                    add("const_pm1", "state" if seq else "datapath", t, _replace(src, toks, i, c))

        # --- state names
        elif t.kind == "id" and t.text in states and t.region != "decl":
            first_on_line = prev is None or prev.line != t.line
            if first_on_line and nxt and nxt.text in (":", ","):
                continue  # case label
            alt = states[(states.index(t.text) + 1) % len(states)]
            add("state_swap", "control", t, _replace(src, toks, i, alt), f"{t.text}->{alt}")

        # --- ternary branch swap
        elif t.text == "?":
            d, colon = 0, None
            for j in range(i + 1, len(toks)):
                x = toks[j].text
                if x in "([{":
                    d += 1
                elif x in ")]}":
                    if d == 0:
                        end = j - 1
                        break
                    d -= 1
                elif x == "?" and d == 0:
                    colon = None
                    end = None
                    break
                elif x == ":" and d == 0 and colon is None:
                    colon = j
                elif x in (";", ",") and d == 0:
                    end = j - 1
                    break
            else:
                end = None
            if colon and end and colon > i + 1 and end > colon:
                a_txt = src[toks[i + 1].start: toks[colon - 1].start + len(toks[colon - 1].text)]
                b_txt = src[toks[colon + 1].start: toks[end].start + len(toks[end].text)]
                if a_txt.strip() != b_txt.strip():
                    new = src[: toks[i + 1].start] + b_txt + " : " + a_txt + src[toks[end].start + len(toks[end].text):]
                    add("ternary_swap", "control", t, new)

        # --- if condition inversion
        elif t.text == "if" and nxt and nxt.text == "(":
            j = match_paren(toks, i + 1)
            cond = src[toks[i + 1].start + 1: toks[j].start]
            new = src[: toks[i + 1].start] + f"(!({cond}))" + src[toks[j].start + 1:]
            add("cond_invert", "control", t, new)

        # --- nonblocking -> blocking in sequential logic
        elif t.text == "<=" and t.pdepth == 0 and seq:
            add("nb_to_b", "state", t, _replace(src, toks, i, "="))

        # --- async <-> sync reset
        elif t.text in ("always", "always_ff") and seq:
            hdr_line = lines[t.line - 1]
            m = re.search(r"@\s*\(\s*posedge\s+(\w+)\s*(,|or)\s*(posedge|negedge)\s+(\w+)\s*\)", hdr_line)
            if m:
                new_line = hdr_line[: m.start()] + f"@(posedge {m.group(1)})" + hdr_line[m.end():]
                add("async_to_sync", "state", t, src.replace(hdr_line, new_line, 1))
            else:
                m = re.search(r"@\s*\(\s*posedge\s+(\w+)\s*\)", hdr_line)
                blk = src[t.start: toks[min(len(toks) - 1, stmt_end(toks, i + 5))].start]
                r = re.search(r"if\s*\(\s*(!?)\s*(a?reset\w*|rst\w*|resetn|aresetn)\s*\)", blk)
                if m and r:
                    edge = "negedge" if r.group(1) else "posedge"
                    new_line = hdr_line[: m.start()] + f"@(posedge {m.group(1)}, {edge} {r.group(2)})" + hdr_line[m.end():]
                    add("sync_to_async", "state", t, src.replace(hdr_line, new_line, 1))

    # dedupe identical mutated sources
    seen, uniq = set(), []
    for m in muts:
        if m.src not in seen:
            seen.add(m.src)
            uniq.append(m)
    return uniq


ASSIGN_LHS = re.compile(r"(?<![=!<>])\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])*\s*(<=|=)(?!=)")


def stmt_lhs(src: str, line: int) -> list[str]:
    """Signals assigned by the statement(s) on/after the mutated line (text ground truth)."""
    ls = src.splitlines()
    out: list[str] = []
    for k in range(line - 1, min(len(ls), line + 6)):
        for m in ASSIGN_LHS.finditer(ls[k]):
            name = m.group(1)
            if name not in ("if", "for", "integer", "i") and name not in out:
                out.append(name)
        if out and ls[k].rstrip().endswith(";"):
            break
    return out
