# Reproducibility — *The Free Inference Dimension Beyond Nested Worlds*

Four protocols and the level structure of $d_{\mathrm{FI}}$.

This folder is a **self-contained** copy of exactly the code needed to reproduce the
five experiment groups listed in the paper's *Experimental Basis and Reproducibility*
section. Every module imports its dependencies by bare name, so **keep all `.py`
files in this one directory and run the scripts from inside it.**

The whole set compiles under Python 3.12 (the version the authors used; 3.10+ should
work) and imports cleanly with the packages in `requirements.txt`.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

All experiments use a Random-Maze generator with wall/step cost ratio
`r_wall/r_step = 100`, start `s0 = (0, H-1)`, goal `g = (W-1, 0)`, per-environment
Q-functions by value iteration, and Free Inference scored by the coherence criterion
`FI ⇔ C(V) ≥ 0` (topological failures recorded as `C = -∞`).

---

## Quick start: the `run_all.py` driver

`run_all.py` runs the five groups in dependency order, logs each step under
`run_all_logs/`, and prints a PASS / FAIL / SKIP summary.

```bash
python run_all.py --smoke     # fast wiring check (~30 s), then:
python run_all.py             # full reproduction at paper settings
python run_all.py --list      # print the plan without running
python run_all.py --groups 1 2 3   # restrict to selected groups
```

- **`--smoke`** validates the whole pipeline in about half a minute: it executes
  the geometry check at tiny scale (20×10, a few seeds) and **import-checks** every
  other module, so you confirm the code is wired and all dependencies resolve before
  committing to the long run. A green smoke run means every module loads and the
  substrate/geometry path works.
- **`--full`** (default) runs each group's real entry point. Analysis/reanalysis
  steps that consume CSVs from earlier groups are **SKIPPED with a note** (not failed)
  when their inputs are not present yet — run the upstream sweeps first.

Two heavy steps are import-checked even in smoke because their calibration grid is
hardcoded (~1,200 cells) and not shrinkable by a flag: the lightest *real* run of each
is `python phase2_shared_skel_calibration_and_sweep.py --step calibrate`
(and likewise `phase3_…`), ~10–15 min each; the full sweeps use `--step all`.

---

## What maps to what

### Core library (shared by every experiment — do not run directly)

| File | Role |
|---|---|
| `GridWorldCanonicalBenchmarkSuite.py` | Random-Maze substrate, families (`RandomMazeFamily`, `RandomScatterFamily`, `SizeSpec`), `build_canonical_suite` |
| `Q_learning_Agent_plus_plus.py` | value-iteration / Q-learning and trajectory rollout (`q_learning`, `get_trajectory`) |
| `MetaAgent2NIGSelfSup.py` | the Meta-Agent: Value-Mixture (VM) and Posterior-Mode-Selection (PMS) action rules |
| `MultiAgentMainRecupF.py` / `…RecupE.py` / `…Recup.py` | Q-table I/O (`load_qtable`), seeding (`doSeed`), rollout drivers |
| `navigational_coherence_new.py` | the navigational-coherence scorer `C(V)` that defines FI success |

### Experiment group 1 — Protocol-geometry verification
*(60×30, 20 master seeds/cell: inclusion fractions, the |F\*|/|F_νN| hierarchy, shared-skeleton saturation, closed-form calibration — Result "Closed-form calibration maps".)*

```bash
python phase1_corrected_protocol_geometry_check.py        # writes ./phase1_geometry_check/
```

### Experiment group 2 — Flat-schedule sweeps
*(40×20 and 60×30, n = 30/cell, 21 calibrated cells, ≈4,350 Q-tables: the level thresholds φ_c^(ℓ), the deep-failure bridge to the nested ρ_c, and the topology collapse — Result "Topology is governed by φ".)*

```bash
python fixed_rho_sweep.py                                 # runs the sweep -> CSV
python fixed_rho_sweep_analyze.py <sweep_csv_path>         # thresholds + plots
```
`fixed_rho_sweep_helpers.py` is imported by the sweep (training + VM/PMS rollouts); not run directly.

### Experiment group 3 — Dense orthogonalized (φ, N) sweep
*(60×30, n = 50, 19 cells: the joint logistic surface and flat-schedule coefficients.)*

```bash
python analyze_dense_sweep_N_effect.py                    # logistic surface fit
```

### Experiment group 4 — Nested re-run + matched-φ shared-skeleton / non-nested sweeps
*(60×30, n = 50, nine cells per protocol at common realized-φ targets: the graduated-class comparison, Results "Surface compatibility" and "The schedule selects the surface".)*

Each phase runs in two steps (`--step calibrate` to pick cells, then `--step all` to sweep):

```bash
python phase2_shared_skel_calibration_and_sweep.py --step calibrate
python phase2_shared_skel_calibration_and_sweep.py --step all
python phase3_non_nested_calibration_and_sweep.py  --step calibrate
python phase3_non_nested_calibration_and_sweep.py  --step all
```
`targeted_f_star_sweep.py` provides the matched-φ targeting these phases import.

### Experiment group 5 — Reanalyses (no new sweeps)
*(pooled graduated-class fit, leave-one-protocol-out transfer, and the single-cell / design-wide Hoeffding certificates.)*

These **consume the CSV outputs of groups 2–4**, so run those first.

```bash
python bootstrap_protocol_comparison.py                   # 2000-iter bootstrap on the per-protocol CSVs
python design_effect_reanalysis.py                        # pooled fit / design-effect reanalysis
python experiment_b_generalization_bound_v2.py            # worst-case growth-function certificate (Thm 7.2)
```

`bootstrap_protocol_comparison.py` reads per-protocol sweep CSVs from
`./logs/bootstrap_protocol_comparison/` (override with `--shared_skel`, `--non_nested`,
`--nested`, …). Its optional `--shared_skel_stoch` input (the stochastic-severity
protocol) belongs to the companion paper and is **not required** for the dFI
graduated-class comparison.

---

## Suggested run order

```
group 1 (geometry)  ─┐
group 2 (flat sweep) ─┤─► group 3 (dense surface) ─► group 4 (matched-φ phases) ─► group 5 (reanalyses/certificates)
```
Groups 1–3 are independent; group 4 produces the per-protocol CSVs that group 5 reanalyzes.

Most scripts accept `--help`. Sweeps are compute-heavy (thousands of Q-tables); start
with reduced seed counts (e.g. `--calibration_seeds`, `--n_per_N` on the phase scripts)
for a quick smoke test before a full run.

