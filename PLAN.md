# Signal-graph + value-window RTL debugging vs. standard agent flow

**Question:** does a pre-built signal *drive graph* (signal → driven signals, with guard and line) with a
*pre-failure value window* attached to every node help Devstral localize, explain and fix RTL bugs better
or more cheaply than (A) a raw RTL+waveform dump or (B) a standard file/grep/waveform tool agent?

**Model:** `devstral-small-2` on local Ollama (`localhost:11434`), num_ctx 16384, temperature 0.
Every arm gets the same model, caps, spec text and TB log.

**Benchmarks:** verilog-eval spec-to-rtl (this study; single-module, ≤ 78 lines) → cvdp_benchmark next
(multi-module; tests the cross-hierarchy hypothesis that verilog-eval cannot).

## Phase 1: bug benchmark (done: `sgd/s01`–`s04`)
- verilog-eval is **read-only**: `_test.sv` is used in place, `_ref.sv`/`_prompt.txt` are copied to
  `refs/`, and `refs/manifest.json` holds sha256 hashes (`python3 -m sgd.s01_copy_refs --verify`).
- DUT = ref renamed `TopModule`, comments stripped, normalized, with exactly one token-level mutation
  (`sgd/mutate.py`). Simulated as `TB(verilog-eval) + RefModule(golden copy) + buggy TopModule`.
- Internal signals are captured by an extra sim top `sgd/sg_dump.sv` (`-s sg_dump`), not by editing the TB.
- Validity: compiles, no hang, `Mismatches > 0` on the original TB. A pass on `Mismatches == 0`
  ignores the TB's 1M-ps watchdog, which also fires on 3 refs.
- Ground truth: mutated line and statement LHS, T_fail and failing outputs (TB hints), and T_activation =
  first divergence from `good1` of a **sequentially-assigned** signal, capped at T_fail. Combinational
  divergence that never gets latched is masked and doesn't count; the raw value is kept as `cyc_act_any`.
  latency = cycles(T_fail − T_activation); `long_horizon` if ≥ 8.
- Result (2026-11-25): 1913 candidates → 1676 valid → 100 eligible problems (72 seq / 28 comb) →
  **269 bugs** (dev 56 / test 213; control 97 / datapath 98 / state 74; 69 with latency > 0, 30 long-horizon,
  max 52k cycles in fancytimer).
- Robustness oracle: `tb_reseed/` = TB **copies** with `$random(sg_seed)` in `stimulus_gen` only
  (seeds 1234, 98765). Every ref passes the original TB and both reseeded copies.
- Selection: problems that are sequential or have ≥ 5 body lines, up to 3 bugs each (long-horizon first,
  then a mix of control/datapath/state). Dev/test split 20/80 **by problem**, stratified by sequential/comb.
- Known limitation: iverilog does not dump memories (e.g. gshare `pht`). Ground truth falls back to the
  mutated statement's LHS.

## Phase 2: treatment system (done: `sgd/graph.py`, `waves.py`, `tools.py`, `s05`, `tests/`)
Implemented as below, with two corrections found while testing. (1) Windows are sampled on both edges of
the TB clock (`tb_sample.clk` in the public VCD) using the **pre-edge value**, matching the scoreboard.
(2) gt_signals = targets of the graph driver whose span, guard, or always-header covers the mutated line,
which fixes multi-line statements and concat LHS. Hops failing-output → gt: 0:177 1:49 2:40 3:3.

- Graph builder (pyslang): nodes are ports/regs/wires (width, kind, FSM-state flag); edges are one per
  driver `src → dst` with kind comb/seq, guard path (enclosing if/case), assignment text and line.
  clk/reset are tagged and hidden by default.
- Value windows from `public/dut_wave.vcd`: last K samples before T_fail per node (K = 8; sweep 4/8/16/32 on dev).
- Tools: `get_node`, `driven_by`, `drives`, `expand_history(sig,t0,t1)`, `get_spec`, `submit`.

## Phase 3: arms
| Arm | Context / tools |
|---|---|
| A raw dump | buggy.sv + VCD slice (DUT scope) in one prompt |
| B standard agent | read_file, grep, list_signals, get_value(sig,t0,t1) |
| T graph + windows | Phase 2 tools |
| T−win | graph only (port mismatches, no values) |
| T−graph | per-signal windows, no edges |

Output JSON: `root_cause_signal, root_cause_line, activation_cycle, explanation, fixed_module`.
The loop is JSON-action ReAct with tolerant parsing, 15-step cap. Primary result is single-shot;
secondary is pass@2 with sim feedback, given identically to every arm.

## Phase 4: metrics
1. Localization: line exact / ±1, signal ∈ gt_signals ∪ div_signals, hop distance in the graph.
2. Cause vs. symptom: activation_cycle within ±1 of truth; blamed-symptom rate.
3. Fix: compile, fail→pass on the original TB, pass on the reseeded TB copies.
4. Explanation: deterministic rubric (true signal, line, cycle, mutation type) plus a 20% manual sample.
5. Cost: tokens, steps, wall time, distraction (visits outside the true backward cone, clk/reset).

Stats: paired by bug. McNemar for binary outcomes, bootstrap 95% CI, broken down by category and long_horizon.

## Phase 5: order
1. Phase 1 (done) → 2. graph + windows with unit checks → 3. pilot: 20 dev bugs × A/B/T →
4. tune on dev only, then freeze → 5. main run: test split × 5 arms (tmux, resumable), 3 seeds on a
40-bug subset for A/B/T → 6. K sweep on long_horizon bugs → 7. port to cvdp_benchmark.

## Layout
```
refs/        copied refs + prompts + manifest.json (hashes of verilog-eval files)
tb_reseed/   reseeded TB copies
sgd/         code (common, mutate, s01..s04)
work/        candidates.jsonl, baseline.json, logs (scratch)
bugs/<id>/public   buggy.sv spec.txt tb_log.txt dut_wave.vcd   <- agents see only this
bugs/<id>/private  meta.json reference_fix.sv wave_full.vcd    <- ground truth
bugs/index.json
```
