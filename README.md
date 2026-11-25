# sigraph-debug

**Does a pre-built signal drive graph with pre-failure value windows help an LLM debug RTL?**

The usual agentic RTL debugging setup has the model read raw RTL and waveform dumps, or grep files and
query waveforms one call at a time. This project tests a third option. We pre-compile a directed **signal
drive graph** where each edge is `src → dst` and carries its guard condition, its kind (comb or seq) and
its `file:line`. Every node also gets the **last K values before the failure**. The agent then walks the
graph backward from the failing output to the root cause, explains the bug, and patches it.

The LLM is **Devstral** (`devstral-small-2`) served by a local [Ollama](https://ollama.com).

Full experiment design: [PLAN.md](PLAN.md).

## Status

| Phase | What | State |
|---|---|---|
| 1 | Bug benchmark mutated from verilog-eval golden refs | done: 269 bugs |
| 2 | Signal-graph builder + value windows + agent tools | done: 269 graphs, 273 tests |
| 3 | Agents: raw dump / standard tools / graph (+ ablations) | todo |
| 4 | Scoring: localization, cause vs. symptom, fix, explanation, cost | todo |
| next | Port to multi-module designs (CVDP benchmark) | planned |

## Benchmark (Phase 1)

The bugs come from the golden reference designs of
[NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval) (spec-to-rtl).

- **verilog-eval itself is never modified.** Its testbenches (`*_test.sv`, golden model vs. DUT
  scoreboard) are used in place as the pass/fail oracle. Refs and prompts are copied into `refs/`, and
  `refs/manifest.json` records the upstream commit and the sha256 of every file used.
- Each bug is a copy of a ref, renamed to `TopModule`, with comments stripped and **exactly one
  token-level mutation** ([`sgd/mutate.py`](sgd/mutate.py)): wrong FSM state, inverted condition, ±1
  constant, operator swap, sync↔async reset, `<=`→`=`, swapped ternary branches, and others.
- A mutant is kept only if it compiles, doesn't hang, and **fails the original verilog-eval TB**.
- **Ground truth** per bug:
  - the mutated line and the signal it assigns
  - the TB's first failing time and outputs
  - the activation cycle: the first cycle where a DUT register differs from the golden model, taken from
    a full-depth VCD captured by an extra simulation top ([`sgd/sg_dump.sv`](sgd/sg_dump.sv)) so the TB
    file stays untouched
  - the latency between activation and failure
- **Fix-robustness oracle:** reseeded *copies* of each TB (`tb_reseed/`, generated). This catches
  patches that only pass the original stimulus.

| | |
|---|---|
| candidate mutants | 1913 (1676 valid) |
| designs used | 100 (72 sequential / 28 combinational) |
| bugs | **269**: dev 56 / test 213 (split by design) |
| categories | control 97 · datapath 98 · state 74 |
| long-horizon (activation ≥ 8 cycles before failure) | 30 (max ≈ 52k cycles) |

Each bug lives in `bugs/<bug_id>/`:

```
public/   buggy.sv  spec.txt  tb_log.txt  dut_wave.vcd*     <- the only things an agent may see
private/  meta.json  reference_fix.sv  wave_full.vcd*        <- ground truth, scoring only
```

`*` = not committed (≈ 210 MB); `python -m sgd.s04_select` regenerates them. `dut_wave.vcd` holds only
the DUT scope plus the TB's expected output values. Golden-model internals are removed.

## Treatment system (Phase 2)

- **Drive graph** ([`sgd/graph.py`](sgd/graph.py), pyslang semantic AST). Each assignment becomes a driver
  record on its target: line span, statement, guard path (`if (...) && case state == S1 ...`), and
  comb/seq/init. Edges `src → dst` are `data` (read on the RHS) or `ctrl` (read in an enclosing condition).
  Nodes are tagged `clock`, `reset`, `fsm_state`, or `unused`. Stored per bug in `public/graph.json`.
- **Value windows** ([`sgd/waves.py`](sgd/waves.py)). Sampled on both edges of the TB clock using the
  value settled *just before* the edge, which is exactly what the verilog-eval scoreboard compares
  (sampling at the edge itself hid the mismatch in 97/269 bugs). FSM values are shown with state names.
- **Agent tools** ([`sgd/tools.py`](sgd/tools.py)): `failure_report`, `get_node`, `driven_by`, `drives`,
  `expand_history`, `list_signals`, `get_spec`, `read_rtl`, plus `with_windows` / `with_edges` switches
  for the ablations.

```text
signal state: internal, 3 bit, roles: fsm_state
drivers:
  L22 [seq] state <= B0;   when: if (!(reset))
  L23 [seq] state <= next;   when: !(!(reset))
fan-in : reset(reset), next(data)
fan-out: next, shift_ena
values in the last 8 samples up to T_fail:
  t=355 cyc=35 posedge: state=3'b010 (B2)
  t=360 cyc=36 negedge: state=3'b000 (B0)
  ...
  t=390 cyc=39 negedge: state=3'b000 (B0)  <- T_fail
```

Checks on all 269 bugs ([`sgd/s05_build_graphs.py`](sgd/s05_build_graphs.py), [`tests/`](tests/)):
every assignment line is a driver, node names match the VCD, the root cause lies in the backward cone of
the failing outputs, and each failing output's mismatch is visible in the public wave at its
first-mismatch time.

Hops from the failing output to the root-cause signal: **0: 177 · 1: 49 · 2: 40 · 3: 3**. On
verilog-eval most bugs sit on the output's own driver, so graph traversal can only pay off on the
92 bugs with 1–3 hops. Results are reported separately for that subset, and CVDP (multi-module) is
the real test of traversal.

## Reproduce

Requirements: Icarus Verilog 12 (`iverilog`, `vvp`), Python ≥ 3.10, and a checkout of verilog-eval
(default location `../verilog-eval`, or set `VERILOG_EVAL_DIR`).

```bash
git clone https://github.com/NVlabs/verilog-eval ../verilog-eval   # if not present
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

python -m sgd.s01_copy_refs            # copy refs/prompts, hash manifest, reseeded TB copies
python -m sgd.s01_copy_refs --verify   # verilog-eval untouched?
python -m sgd.s02_baseline             # every ref must pass original + reseeded TBs
python -m sgd.s03_mutants --jobs 16    # generate + validate all mutants   (~10 min)
python -m sgd.s04_select               # pick bugs, split, write bugs/ (incl. VCDs)
python -m sgd.s05_build_graphs         # drive graphs + checks; refines gt_signals in meta.json
pytest -q tests
```

## Layout

```
sgd/        pipeline code (common, mutate, s01..s05, graph, waves, tools)
tests/      Phase 2 tests
refs/       copied verilog-eval refs + prompts, manifest, upstream license
bugs/       the benchmark (index.json + one folder per bug)
PLAN.md     full experiment plan
```

## Attribution

The reference designs, problem prompts and testbenches are from
[NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval), © 2023-2024 NVIDIA Research Projects,
MIT License (see [`refs/LICENSE-verilog-eval`](refs/LICENSE-verilog-eval)). Every file in `bugs/` is a
modified version of those refs.
