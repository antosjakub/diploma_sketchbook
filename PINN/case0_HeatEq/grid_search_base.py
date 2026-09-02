import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path


src_dir = os.path.join(os.path.dirname(__file__), "../src/")
sys.path.append(src_dir)
import utility


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


def iter_base_combos(search_axes: dict):
    axis_names = list(search_axes.keys())
    axis_values = [search_axes[name] for name in axis_names]
    for raw_items in itertools.product(*axis_values):
        parts = [
            normalize_axis_item(axis_name, item)
            for axis_name, item in zip(axis_names, raw_items)
        ]
        yield merge_dicts(parts)


def combo_name(combo: dict) -> str:
    parts = [f"{key}={slug(combo[key])}" for key in combo]
    return "__".join(parts)


def build_run_config(
    fixed_params: dict,
    base_combo: dict,
    output_dir: Path,
    project_dir: Path,
):
    config = dict(fixed_params)
    config.update(base_combo)
    config["output_dir"] = str(output_dir.relative_to(project_dir))
    return config


def run_one(
    entrypoint: str,
    run_dir: Path,
    config: dict,
    *,
    project_dir: Path,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    json_dump(config_path, config)

    cmd = [
        "python",
        entrypoint,
        "--config",
        str(config_path.relative_to(project_dir)),
    ]
    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()
    log_path = run_dir / "stdout_stderr.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_fp:
        proc = subprocess.run(
            cmd,
            cwd=project_dir,
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


def resolve_search_root(base_dir: Path, *, suffix: str | None, out_dir: str | None) -> Path:
    if out_dir is None:
        timestamp = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
        if suffix is None:
            return base_dir / f"gridsearch__{timestamp}"
        return base_dir / f"gridsearch__{timestamp}__{suffix}"

    print("[ DBG: out_dir specified, ignoring passed suffix... ]")
    return base_dir / out_dir


def run_grid_search(
    *,
    fixed_params: dict,
    search_axes: dict,
    search_base_dir: str | Path,
    project_dir: str | Path | None = None,
    entrypoint: str = "main.py",
    suffix: str | None = None,
    out_dir: str | None = None,
) -> Path:
    search_base_dir = Path(search_base_dir).resolve()
    project_dir = Path(project_dir).resolve() if project_dir is not None else search_base_dir
    search_root = resolve_search_root(search_base_dir, suffix=suffix, out_dir=out_dir)
    print(f"[ DBG: Saving to {search_root} ]")

    search_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "entrypoint": entrypoint,
        "fixed_params": fixed_params,
        "search_axes": search_axes,
    }
    json_dump(search_root / "manifest.json", manifest)

    base_combos = list(iter_base_combos(search_axes))
    total_runs = len(base_combos)
    print(f"Grid search root: {search_root.relative_to(project_dir)}")
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

        run_config = build_run_config(
            fixed_params=fixed_params,
            base_combo=base_combo,
            output_dir=combo_dir,
            project_dir=project_dir,
        )
        run_rc = run_one(
            entrypoint,
            combo_dir,
            run_config,
            project_dir=project_dir,
        )
        run_records.append(
            {
                "combo": base_combo,
                "run_dir": str(combo_dir.relative_to(project_dir)),
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
    print(f"Summary: {search_root.relative_to(project_dir) / 'summary.json'}")
    return search_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default=None, type=str, help="suffix of the generated grid search folder")
    parser.add_argument("--out_dir", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(
        "grid_search_base.py is now an importable helper. "
        "Import run_grid_search(...) from another grid_search.py file."
    )
