#!/usr/bin/env python3
"""
run_all.py — pipeline driver for the dFI reproducibility package
================================================================

Runs the five experiment groups of "The Free Inference Dimension Beyond
Nested Worlds" in dependency order, writing a per-step log and a final
PASS / FAIL / SKIP summary.

Two modes
---------
--smoke   Fast end-to-end wiring check (minutes). Executes the three
          parametrizable sweeps at tiny scale (phase1 with a few seeds,
          phase2 / phase3 calibrate-only with 2 seeds) and *import-checks*
          every other module. Use this to confirm the pipeline is wired
          correctly before committing to a full run.

--full    (default) Runs each group's real entry point at paper settings.
          Analysis / reanalysis steps that consume CSVs produced by earlier
          groups are SKIPPED (not failed) when their inputs are not present
          yet, with a note telling you which sweep to run first.

Usage
-----
    python run_all.py --smoke
    python run_all.py                 # full run
    python run_all.py --groups 1 2    # only groups 1 and 2
    python run_all.py --list          # print the plan and exit
    python run_all.py --full --stop-on-error

Everything is run from this directory (the modules import each other by
bare name), using the same Python that launched this script.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import glob
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

IMPORT_ONLY = object()  # sentinel: in this mode, just `import` the module


@dataclasses.dataclass
class Step:
    group: int
    name: str
    module: str                       # importable module name (no .py)
    full: object                      # argv list, or IMPORT_ONLY
    smoke: object                     # argv list, or IMPORT_ONLY
    requires: tuple[str, ...] = ()     # path globs that must exist for `full`
    note: str = ""                    # shown when SKIPPED for missing inputs


def _imp(module: str) -> list[str]:
    return [PY, "-c", f"import {module}"]


# --- the plan -------------------------------------------------------------
# Each step's `full` is the real invocation; `smoke` is the fast one.
# Steps whose input is a CSV from an earlier group use IMPORT_ONLY in smoke
# and declare `requires` so `full` SKIPs cleanly when the sweep hasn't run.

STEPS: list[Step] = [
    # ---- Group 1: protocol-geometry verification ----
    Step(
        1, "phase1_geometry", "phase1_corrected_protocol_geometry_check",
        full=[PY, "phase1_corrected_protocol_geometry_check.py"],
        smoke=[PY, "-c",
               "import phase1_corrected_protocol_geometry_check as m; "
               "m.main(grid_w=20, grid_h=10, n_check_seeds=2, n_seeds=3)"],
    ),

    # ---- Group 2: flat-schedule sweeps ----
    Step(
        2, "fixed_rho_sweep", "fixed_rho_sweep",
        full=[PY, "fixed_rho_sweep.py"],
        smoke=IMPORT_ONLY,
    ),
    Step(
        2, "fixed_rho_analyze", "fixed_rho_sweep_analyze",
        full=IMPORT_ONLY,   # needs a sweep CSV as positional arg; run manually
        smoke=IMPORT_ONLY,
        note="run manually: python fixed_rho_sweep_analyze.py <sweep_csv>",
    ),

    # ---- Group 3: dense orthogonalized (phi, N) sweep ----
    Step(
        3, "dense_surface", "analyze_dense_sweep_N_effect",
        full=[PY, "analyze_dense_sweep_N_effect.py"],
        smoke=IMPORT_ONLY,
    ),

    # ---- Group 4: matched-phi cross-protocol sweeps ----
    # The calibration grid in phase2/phase3 is hardcoded (~1200 cells) and is
    # NOT shrunk by --calibration_seeds, so even "calibrate" is ~10-15 min.
    # Smoke therefore import-checks them; the lightest *real* execution is
    # `--step calibrate` (see note), and the full run is `--step all`.
    Step(
        4, "phase2_shared_skel", "phase2_shared_skel_calibration_and_sweep",
        full=[PY, "phase2_shared_skel_calibration_and_sweep.py", "--step", "all"],
        smoke=IMPORT_ONLY,
        note="lightest real run: python phase2_shared_skel_calibration_and_sweep.py "
             "--step calibrate  (~10-15 min); full uses --step all",
    ),
    Step(
        4, "phase3_non_nested", "phase3_non_nested_calibration_and_sweep",
        full=[PY, "phase3_non_nested_calibration_and_sweep.py", "--step", "all"],
        smoke=IMPORT_ONLY,
        note="lightest real run: python phase3_non_nested_calibration_and_sweep.py "
             "--step calibrate  (~10-15 min); full uses --step all",
    ),
    Step(
        4, "targeted_sweep", "targeted_f_star_sweep",
        full=IMPORT_ONLY,   # targeting helper driven by the phase scripts
        smoke=IMPORT_ONLY,
        note="imported by phase2/phase3; not a standalone full-run entry",
    ),

    # ---- Group 5: reanalyses + Hoeffding certificates ----
    Step(
        5, "bootstrap", "bootstrap_protocol_comparison",
        full=[PY, "bootstrap_protocol_comparison.py"],
        smoke=IMPORT_ONLY,
        requires=(
            "logs/bootstrap_protocol_comparison/phase2_shared_skel-*.csv",
            "logs/bootstrap_protocol_comparison/phase3_non_nested-*.csv",
            "logs/bootstrap_protocol_comparison/targeted_nested_sweep-*.csv",
        ),
        note="needs per-protocol sweep CSVs in logs/bootstrap_protocol_comparison/ "
             "(from groups 2-4)",
    ),
    Step(
        5, "design_effect", "design_effect_reanalysis",
        full=[PY, "design_effect_reanalysis.py"],
        smoke=IMPORT_ONLY,
        note="pooled graduated-class / design-effect reanalysis of sweep CSVs",
    ),
    Step(
        5, "experiment_b", "experiment_b_generalization_bound_v2",
        full=[PY, "experiment_b_generalization_bound_v2.py"],
        smoke=IMPORT_ONLY,
        note="worst-case growth-function certificate (Thm 7.2); standalone entry",
    ),
]


def _missing_requires(step: Step) -> list[str]:
    missing = []
    for pat in step.requires:
        if not glob.glob(str(HERE / pat), recursive=True):
            missing.append(pat)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="fast wiring check (tiny sweeps + import validation)")
    ap.add_argument("--full", action="store_true",
                    help="full run at paper settings (default if --smoke absent)")
    ap.add_argument("--groups", type=int, nargs="+", choices=[1, 2, 3, 4, 5],
                    help="restrict to these experiment groups")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="halt at the first failing step (default: continue)")
    ap.add_argument("--list", action="store_true",
                    help="print the execution plan and exit")
    args = ap.parse_args()

    smoke = args.smoke and not args.full
    mode = "SMOKE" if smoke else "FULL"

    steps = [s for s in STEPS if (not args.groups or s.group in args.groups)]

    if args.list:
        print(f"Plan ({mode} mode):")
        for s in steps:
            cmd = s.smoke if smoke else s.full
            how = "import-check" if cmd is IMPORT_ONLY else " ".join(cmd[1:] or cmd)
            print(f"  [G{s.group}] {s.name:22s} -> {how}")
        return 0

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = HERE / "run_all_logs" / f"{mode.lower()}_{ts}"
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"=== dFI reproducibility :: {mode} run ===")
    print(f"python : {PY}")
    print(f"logs   : {logdir}")
    print()

    results: list[tuple[Step, str, float, str]] = []
    for i, s in enumerate(steps, 1):
        cmd = s.smoke if smoke else s.full
        tag = f"[G{s.group}] {s.name}"

        # skip full-mode steps whose upstream CSVs aren't present
        if not smoke and cmd is not IMPORT_ONLY:
            missing = _missing_requires(s)
            if missing:
                reason = s.note or f"missing inputs: {', '.join(missing)}"
                print(f"--- {i}/{len(steps)} {tag}: SKIP ({reason})")
                results.append((s, "SKIP", 0.0, reason))
                continue

        is_import = cmd is IMPORT_ONLY
        run_cmd = _imp(s.module) if is_import else cmd
        label = "import-check" if is_import else "run"
        print(f"--- {i}/{len(steps)} {tag}: {label} ...", flush=True)

        log_path = logdir / f"{s.group}_{s.name}.log"
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"# {tag} ({label})\n# cmd: {' '.join(run_cmd)}\n\n")
            fh.flush()
            proc = subprocess.run(run_cmd, cwd=str(HERE), stdout=fh,
                                  stderr=subprocess.STDOUT)
        dt = time.time() - t0

        if proc.returncode == 0:
            status = "OK(import)" if is_import else "OK"
            print(f"    {status}  ({dt:.1f}s)  -> {log_path.name}")
            results.append((s, status, dt, ""))
        else:
            print(f"    FAIL (exit {proc.returncode}, {dt:.1f}s)  -> {log_path.name}")
            # show the last few log lines inline for quick triage
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
            for ln in tail:
                print(f"      | {ln}")
            results.append((s, "FAIL", dt, f"exit {proc.returncode}"))
            if args.stop_on_error:
                print("\nStopping at first error (--stop-on-error).")
                break

    # ---- summary ----
    print("\n=== summary ===")
    width = max(len(s.name) for s, *_ in results) if results else 10
    n_fail = 0
    for s, status, dt, extra in results:
        extra = f"  {extra}" if extra else ""
        print(f"  [G{s.group}] {s.name:{width}s}  {status:11s} {dt:6.1f}s{extra}")
        n_fail += status == "FAIL"

    print()
    if n_fail:
        print(f"{n_fail} step(s) FAILED. See logs in {logdir}")
    else:
        print(f"All steps OK/SKIP in {mode} mode. Logs in {logdir}")
        if smoke:
            print("Smoke passed: pipeline wiring validated. Run `python run_all.py` "
                  "for the full reproduction.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
