"""Benchmark matrix and matrix-free solver backends.

Run from the repository root, for example:

    python examples/benchmark_backends.py --dimension 3 --level 4 --output results/ou_backend_benchmark.csv
"""

from __future__ import annotations

import argparse

from combination_technique.benchmark import run_backend_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/ou_backend_benchmark.csv")
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--domain-radius", type=float, default=4.0)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument("--max-component-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_backend_benchmark(
        output_path=args.output,
        dimension=args.dimension,
        level=args.level,
        final_time=args.final_time,
        dt=args.dt,
        domain_radius=args.domain_radius,
        rho=args.rho,
        max_workers=args.max_workers,
        repeats=args.repeats,
        min_level=args.min_level,
        max_component_size=args.max_component_size,
    )
    summary_rows = [row for row in rows if row["row_type"] == "summary"]
    print(f"wrote {len(rows)} rows to {args.output}")
    for row in summary_rows:
        print(
            f"{row['case']}: wall={row['wall_seconds']:.6f}s, "
            f"component_total={row['total_component_time']:.6f}s, "
            f"krylov={row['total_krylov_iterations']}"
        )


if __name__ == "__main__":
    main()
