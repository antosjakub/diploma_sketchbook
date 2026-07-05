import itertools
import json
import subprocess
import time
from pathlib import Path


import os, sys
src_dir = os.path.join(os.path.dirname(__file__), '../src/')
sys.path.append(src_dir)
import utility


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default=None, type=str, help="suffix of the generated grid search folder")
parser.add_argument("--out_dir", default=None, type=str)
args = parser.parse_args()

CASE_DIR = Path(__file__).resolve().parent # case0_HeatEq/
if args.out_dir == None:
    timestamp = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    if args.suffix == None:
        search_root = CASE_DIR / f"gridsearch__{timestamp}"
    else:
        search_root = CASE_DIR / f"gridsearch__{timestamp}__{args.suffix}"
else:
    print("[ DBG: out_dir specified, ignoring passed suffix... ]")
    search_root = CASE_DIR / args.out_dir
print(f"[ DBG: Saving to {search_root} ]")

FIXED_PARAMS = {
    "description": "test",
    "clear_dir": False,
    "seed": 42,

    "d": 2,
    "layers": "64,64,64,64",
    #"layers": "128,128,128,128",
    #"layers": "192,192,192,192",
    #"layers": "256,256,256,256",
    #"layers": "512,512,512,512",

    "time_strategy": "none",
    "t_discr": "0.0, 0.5, 1.5, 3.5",
    "eps": 0.1,

    #"n_steps": 29_999,
    "n_steps": 109,
    "n_steps_decay": 2000, # = n_steps / 25
    # 3)

    "n_res_points": 8192,
    "bs": 128,
    "resampling_frequency": 100, # res_freq * bs = n_res_points
    "one_batch_per_epoch": True,

    "active_losses": "pde,ic",
    "use_hard_constrains": True,
    "lambda_strategy": "fixed",
    "lambda_pde": 1.0,
    "lambda_bc": 1.0,
    "lambda_ic": 1.0,

    "use_lbfgs": False,

    #"enable_profiler": True
    "enable_memory_tracking": True,
    "enable_testing": False,
    "n_test_points": 100_000,
}


# Add or remove search axes here.
# Each axis value is a list. Items may be either:
# - scalars, which set one parameter with the axis name
# - dicts, for grouped parameters such as trajectory sampling settings
SEARCH_AXES = {
    #"ic_type": ["cauchy", "gauss"],
    #"lambda_pde": [0.1, 1.0, 10.0],
    "lambda_bc": [0.1, 1.0, 10.0],
    #"lambda_ic": [0.1, 1.0, 10.0]
    #"d": [6, 4, 8],
    #"box": [
    #    {"L_min": -4.0, "L_max": 4.0},
    #    {"L_min": -6.0, "L_max": 6.0},
    #],
    "sampling": [
        {
            # full batch, fixed
            "n_res_points": 8192,
            "bs": 8192,
            "one_batch_per_epoch": True,
            "prevent_resampling": True
        },
        {
            # mini batch, fixed
            "n_res_points": 8192,
            "bs": 128,
            "one_batch_per_epoch": False,
            "prevent_resampling": True
        },
        {
            # full batch, resampl
            "n_res_points": 8192,
            "bs": 8192,
            "one_batch_per_epoch": True,
            "prevent_resampling": False
        },
        {
            # mini batch, resampl
            "n_res_points": 8192,
            "resampling_frequency": 1,
            "bs": 128,
            "one_batch_per_epoch": False,
            "prevent_resampling": False
        },
    ],
    #"T": [5.0, 6.0],
    #"bs": [10_000, 1_000]
    #"stepping": [
    #    {
    #        "n_steps": 19_999,
    #        #"n_steps": 19,
    #        "n_steps_decay": 800,
    #    },
    #    {
    #        "n_steps": 9_999,
    #        #"n_steps": 9,
    #        "n_steps_decay": 400,
    #    },
    #    {
    #        "n_steps": 39_999,
    #        #"n_steps": 39,
    #        "n_steps_decay": 1_600,
    #    },
    #],
    #"use_adaptive_weights": [True, False],
    #"n_steps_decay": [1_000, 2_000, 500], # ~ 20-5 times decay (10 times was best so far)
    #"layers": [
    #    "128,128,128,128",
    #    #"256,256,256,256",
    #    "64,64,64,64",
    ##    #"148,148,148,148"
    #],
    #"sampling": [
    #    {
    #        "f_ic_full_domain": 1,
    #        "f_ic_trajs": 1,
    #    },
    #    {
    #        "f_ic_full_domain": 4,
    #        "f_ic_trajs": 1,
    #    },
    #    {
    #        "f_ic_full_domain": 1,
    #        "f_ic_trajs": 4,
    #    },
    #],
    #"sampling": [
    #    {
    #        "sampling_type": "domain",
    #    },
    #    {
    #        "sampling_type": "trajectories",
    #        "n_trajs": 1_000,
    #        "nt_steps": 1_000,
    #    },
    #],
    #"active_losses": ["pde,bc,ic", "pde,ic"]
}




def json_dump(file_path: Path, obj) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def slug(value) -> str:
    if isinstance(value, float):
        return str(value).replace(".", "p").replace("-", "m")
    return str(value).replace("/", "-").replace(",", "_").replace(" ", "")


def normalize_axis_item(axis_name: str, item) -> dict:
    if isinstance(item, dict):
        return dict(item)
    return {axis_name: item}


def merge_dicts(parts: list[dict]) -> dict:
    merged = {}
    for part in parts:
        overlap = set(merged) & set(part)
        if overlap:
            raise ValueError(f"Duplicate keys while building combo: {sorted(overlap)}")
        merged.update(part)
    return merged


def iter_base_combos():
    axis_names = list(SEARCH_AXES.keys())
    axis_values = [SEARCH_AXES[name] for name in axis_names]
    for raw_items in itertools.product(*axis_values):
        parts = [
            normalize_axis_item(axis_name, item)
            for axis_name, item in zip(axis_names, raw_items)
        ]
        yield merge_dicts(parts)


def combo_name(combo: dict) -> str:
    parts = [f"{key}={slug(combo[key])}" for key in combo]
    return "__".join(parts)


def build_run_config(base_combo: dict, output_dir: Path):
    config = dict(FIXED_PARAMS)
    config.update(base_combo)
    config["output_dir"] = str(output_dir.relative_to(CASE_DIR))
    return config


def run_one(entrypoint: str, run_dir: Path, config: dict) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    json_dump(config_path, config)

    cmd = [
        "python",
        entrypoint,
        "--config",
        str(config_path.relative_to(CASE_DIR)),
    ]
    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()
    log_path = run_dir / "stdout_stderr.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        proc = subprocess.run(
            cmd,
            cwd=CASE_DIR,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start_time

    (run_dir / "time.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    (run_dir / "return_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")

    if proc.returncode == 0:
        print(f"OK  {run_dir.name}  {elapsed:.1f}s")
    else:
        print(f"FAIL {run_dir.name}  code={proc.returncode}")
    return proc.returncode


def main():
    search_root.mkdir(parents=True, exist_ok=True)

    entrypoint = "main.py"
    manifest = {
        "entrypoint": entrypoint,
        "fixed_params": FIXED_PARAMS,
        "search_axes": SEARCH_AXES,
    }
    json_dump(search_root / "manifest.json", manifest)

    base_combos = list(iter_base_combos())
    total_runs = len(base_combos)
    print(f"Grid search root: {search_root.relative_to(CASE_DIR)}")
    print(f"Base combos: {len(base_combos)}")
    print(f"Planned runs: {total_runs}")

    n_ok = 0
    n_fail = 0
    run_records = []

    gs_start_time = time.time()
    for combo_index, base_combo in enumerate(base_combos, start=1):
        combo_dir = search_root / combo_name(base_combo)
        combo_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[{combo_index}/{len(base_combos)}] {combo_dir.name}")

        run_dir = combo_dir
        run_config = build_run_config(
            base_combo=base_combo,
            output_dir=run_dir,
        )
        run_rc = run_one(entrypoint, run_dir, run_config)
        run_records.append(
            {
                "combo": base_combo,
                "run_dir": str(run_dir.relative_to(CASE_DIR)),
                "return_code": run_rc,
            }
        )
        if run_rc == 0:
            n_ok += 1
        else:
            n_fail += 1


    summary = {
        "n_ok": n_ok,
        "n_fail": n_fail,
        "records": run_records,
    }
    json_dump(search_root / "summary.json", summary)

    print()
    print(utility.get_duration_h_m_s(gs_start_time, time.time(), label="Grid search"))
    print(f"ok={n_ok}, failed={n_fail}")
    print(f"Summary: {search_root.relative_to(CASE_DIR) / 'summary.json'}")


if __name__ == "__main__":
    main()
