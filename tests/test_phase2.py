"""Phase 2 checks: graph structure, waveform/TB time alignment, tool behaviour and ablation switches.
Tests needing VCDs are skipped if the waves were not regenerated (python -m sgd.s04_select)."""
import json
from pathlib import Path

import pytest

from sgd.common import BUGS, REFS, ref_as_dut
from sgd.graph import backward_cone, build_graph
from sgd.tools import GraphEnv
from sgd.waves import tb_mismatch

BUG_DIRS = sorted(p for p in BUGS.iterdir() if p.is_dir())
HAVE_WAVES = all((b / "public" / "dut_wave.vcd").exists() for b in BUG_DIRS)
needs_waves = pytest.mark.skipif(not HAVE_WAVES, reason="VCDs not generated; run python -m sgd.s04_select")


def edges(g):
    return {(e["src"], e["dst"], e["dep"], e["kind"]) for e in g["edges"]}


def test_graph_fsm3s_exact():
    g = build_graph(ref_as_dut((REFS / "Prob120_fsm3s_ref.sv").read_text()))
    assert edges(g) == {
        ("in", "next", "data", "comb"), ("state", "next", "ctrl", "comb"),
        ("next", "state", "data", "seq"), ("reset", "state", "ctrl", "seq"),
        ("state", "out", "data", "comb"),
    }
    assert g["nodes"]["clk"]["roles"] == ["clock"]
    assert g["nodes"]["reset"]["roles"] == ["reset"]
    assert "fsm_state" in g["nodes"]["state"]["roles"]
    guards = {d["guard"] for d in g["nodes"]["state"]["drivers"]}
    assert guards == {"if (reset)", "!(reset)"}


def test_graph_gshare_memory_and_async_reset():
    g = build_graph(ref_as_dut((REFS / "Prob153_gshare_ref.sv").read_text()))
    pht = g["nodes"]["pht"]
    assert pht["memory"] and pht["width"] == 2 and pht["depth"] == 128
    assert g["blocks"][0]["async_reset"] == ["areset"]
    assert "predict_history_r" in backward_cone(g, ["predict_history"])


def test_all_graphs_cover_ground_truth():
    rows = json.loads((BUGS / "graph_checks.json").read_text())
    assert len(rows) == len(BUG_DIRS)
    assert all(not r["missing_driver_lines"] for r in rows)
    assert all(r["gt_signal_in_cone"] and r["gt_line_in_cone"] for r in rows)


@needs_waves
@pytest.mark.parametrize("bug", BUG_DIRS, ids=lambda p: p.name)
def test_failure_visible_in_public_wave(bug):
    """At T_fail some failing output must differ from its expected value in the public VCD."""
    env = GraphEnv(bug)
    assert env.t_fail is not None
    diffs = [tb_mismatch(env.w.sampled(o, env.first_mismatch[o]), env.w.sampled_expected(o, env.first_mismatch[o]))
             for o in env.failing]
    assert all(diffs), f"mismatch not visible at first-mismatch time: {dict(zip(env.failing, diffs))}"
    ts = env.w.sample_times(env.t_fail, env.k)
    assert ts[-1] == env.t_fail and len(ts) <= env.k


@needs_waves
def test_tools_and_ablations():
    bug = BUGS / "bug111"
    full = GraphEnv(bug)
    assert "drivers:" in full.get_node("state") and "values in the last" in full.get_node("state")
    assert "(B0)" in full.get_node("state")
    no_win = GraphEnv(bug, with_windows=False)
    assert "values in the last" not in no_win.get_node("state")
    assert "unavailable" in no_win.expand_history("state", 0, 100)
    assert "DUT=" not in no_win.failure_report()
    no_edges = GraphEnv(bug, with_edges=False)
    s = no_edges.get_node("state")
    assert "drivers:" not in s and "fan-in" not in s and "values in the last" in s
    assert "unavailable" in no_edges.driven_by("state")
    h = full.expand_history("state", 0, 400)
    assert h.count("\n") >= 1 and "3'b000 (B0)" in h
