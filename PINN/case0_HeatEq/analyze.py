import json
import os

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = "gridsearch__2026-07-07--12-36-01"
TERM = "pde"


def get_data(item_path, item, file_name):
    json_path = os.path.join(item_path, file_name)
    if not os.path.exists(json_path):
        print(f"!! No {file_name} found in: {item}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Error loading {item}/{file_name}: {exc}")
        return None


def collect_data(current_dir, term):
    data_full = {}
    report_names = []

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if not os.path.isdir(item_path):
            continue

        print(item_path)
        model_metadata = get_data(item_path, item, "model_metadata.json")
        if model_metadata is None:
            continue

        args = model_metadata.get("args", {})
        if not args.get("prevent_resampling", False):
            continue

        report = get_data(item_path, item, "report.json")
        if report is None:
            continue

        data_full[item] = {
            "lambda_ic": args["lambda_ic"],
            "bs": args["bs"],
            "linf": report["test_linf"][term],
            "rel_l2": report["test_rel_l2"][term],
            "runtime": report["runtime"],
        }
        report_names.append(item)

    print(f"\nTotal reports loaded: {len(report_names)}")
    print(list(data_full.values()))
    return data_full


def create_pivot_table(records, err_type):
    row_keys = sorted({record["lambda_ic"] for record in records.values()})
    col_keys = sorted({record["bs"] for record in records.values()})
    row_index = {value: idx for idx, value in enumerate(row_keys)}
    col_index = {value: idx for idx, value in enumerate(col_keys)}

    matrix = np.full((len(row_keys), len(col_keys)), np.nan, dtype=float)
    for record in records.values():
        matrix[row_index[record["lambda_ic"]], col_index[record["bs"]]] = record[err_type]

    return matrix, row_keys, col_keys


def plot_matrix(matrix, row_labels, col_labels, err_type, file_name, cmap="YlOrRd"):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap=cmap)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title("Grid Search")
    ax.set_xlabel("bs")
    ax.set_ylabel("lambda_ic")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", color="black")

    plt.colorbar(im, label=err_type)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def main():
    data_full = collect_data(CURRENT_DIR, TERM)
    if not data_full:
        print("No matching reports found.")
        return

    for err_type in ("linf", "rel_l2"):
        matrix, row_labels, col_labels = create_pivot_table(data_full, err_type)
        plot_matrix(matrix, row_labels, col_labels, err_type, f"analysis_{err_type}")


if __name__ == "__main__":
    main()
