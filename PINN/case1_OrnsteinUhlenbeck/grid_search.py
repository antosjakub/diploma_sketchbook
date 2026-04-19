import itertools
import json
import subprocess
import time
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent


# Choose which training entrypoint to use.
# Supported values: "hardcoded", "3losses"
VARIANT = "hardcoded"


FIXED_PARAMS = {
    #"ic_type": "gauss",
    "description": "OU IC grid search",
    "seed": 42,
    "layers": "128,128,128,128",
    "n_steps": 9950,
    "n_steps_decay": 5_000,
    "gamma": 0.9,
    "lr": 1e-3,
    "bs": 1_000,
    "n_res_points": 100_000,
    "resampling_frequency": 500,
    "T": 1.0,
    "n_test_points": 10_000,
    "testing_frequency": 100,
    "L_min": -4.0,
    "L_max": 4.0
}


# Add or remove search axes here.
# Each axis value is a list. Items may be either:
# - scalars, which set one parameter with the axis name
# - dicts, for grouped parameters such as trajectory sampling settings
SEARCH_AXES = {
    "ic_type": ["gauss", "cauchy", "laplace"],
    "d": [2, 4],
    #"box": [
    #    {"L_min": -4.0, "L_max": 4.0},
    #    {"L_min": -6.0, "L_max": 6.0},
    #],
    "sampling": [
        {
            "sampling_type": "trajectories",
            "n_trajs": 1_000,
            "nt_steps": 1_000,
        },
        {
            "sampling_type": "domain",
        },
    ],
}


# `ll_ode` depends on a matching `score_pde` run, so modes are run in order
# for each base combo instead of being treated as an independent search axis.
MODES = ["score_pde", "ll_ode"]


def json_dump(file_path: Path, obj) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def get_entrypoint(variant: str) -> str:
    if variant == "hardcoded":
        return "main_score_pinn_hardcoded.py"
    if variant == "3losses":
        return "main_score_pinn_3losses.py"
    raise ValueError(f"Unsupported variant: {variant!r}")


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


def build_run_config(base_combo: dict, mode: str, output_dir: Path, linked_score_pde_dir: Path | None):
    config = dict(FIXED_PARAMS)
    config.update(base_combo)
    config["mode"] = mode
    config["output_dir"] = str(output_dir.relative_to(CASE_DIR))
    if linked_score_pde_dir is not None:
        config["linked_score_pde_dir"] = str(linked_score_pde_dir.relative_to(CASE_DIR))
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


def validate_search_setup():
    allowed_modes = {"score_pde", "ll_ode"}
    unknown_modes = sorted(set(MODES) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"Unsupported modes in MODES: {unknown_modes}")
    if "ll_ode" in MODES and "score_pde" not in MODES:
        raise ValueError("ll_ode requires score_pde because it depends on the trained score model.")


def main():
    validate_search_setup()

    timestamp = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    search_root = CASE_DIR / f"gridsearch__{VARIANT}__{timestamp}"
    search_root.mkdir(parents=True, exist_ok=True)

    entrypoint = get_entrypoint(VARIANT)
    manifest = {
        "variant": VARIANT,
        "entrypoint": entrypoint,
        "fixed_params": FIXED_PARAMS,
        "search_axes": SEARCH_AXES,
        "modes": MODES,
    }
    json_dump(search_root / "manifest.json", manifest)

    base_combos = list(iter_base_combos())
    total_runs = len(base_combos) * len(MODES)
    print(f"Grid search root: {search_root.relative_to(CASE_DIR)}")
    print(f"Base combos: {len(base_combos)}")
    print(f"Planned runs: {total_runs}")

    n_ok = 0
    n_fail = 0
    run_records = []

    for combo_index, base_combo in enumerate(base_combos, start=1):
        combo_dir = search_root / combo_name(base_combo)
        combo_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[{combo_index}/{len(base_combos)}] {combo_dir.name}")

        score_run_dir = combo_dir / "score_pde"
        ll_run_dir = combo_dir / "ll_ode"

        score_rc = None
        if "score_pde" in MODES:
            score_config = build_run_config(
                base_combo=base_combo,
                mode="score_pde",
                output_dir=score_run_dir,
                linked_score_pde_dir=None,
            )
            score_rc = run_one(entrypoint, score_run_dir, score_config)
            run_records.append(
                {
                    "combo": base_combo,
                    "mode": "score_pde",
                    "run_dir": str(score_run_dir.relative_to(CASE_DIR)),
                    "return_code": score_rc,
                }
            )
            if score_rc == 0:
                n_ok += 1
            else:
                n_fail += 1

        if "ll_ode" not in MODES:
            continue
        if score_rc != 0:
            print("Skipping ll_ode because score_pde failed.")
            run_records.append(
                {
                    "combo": base_combo,
                    "mode": "ll_ode",
                    "run_dir": str(ll_run_dir.relative_to(CASE_DIR)),
                    "return_code": None,
                    "skipped": "score_pde_failed",
                }
            )
            continue

        ll_config = build_run_config(
            base_combo=base_combo,
            mode="ll_ode",
            output_dir=ll_run_dir,
            linked_score_pde_dir=score_run_dir,
        )
        ll_rc = run_one(entrypoint, ll_run_dir, ll_config)
        run_records.append(
            {
                "combo": base_combo,
                "mode": "ll_ode",
                "run_dir": str(ll_run_dir.relative_to(CASE_DIR)),
                "return_code": ll_rc,
            }
        )
        if ll_rc == 0:
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
    print(f"Completed. ok={n_ok}, failed={n_fail}")
    print(f"Summary: {search_root.relative_to(CASE_DIR) / 'summary.json'}")


if __name__ == "__main__":
    main()
