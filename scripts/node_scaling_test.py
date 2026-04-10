#!/usr/bin/env python3
"""Benchmark MPI/OpenMP matrix multiplication programs across node/thread counts.

This script is intended for the report section "不同节点数量下MPI(openMP)的运行速度".
It runs a fixed square matrix size n*n, varies MPI process counts and/or OpenMP
thread counts, and records the [Time taken] value printed by each program.

Default programs:
- MM_RS_mpi
- MM_RS_openmp

The script is flexible enough to benchmark the CBLAS variants too if requested.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TIME_PATTERN = re.compile(
    r"\[Time taken\]\s*<([^>]+)>\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds"
)

DEFAULT_PROGRAMS = ["MM_RS_mpi", "MM_RS_openmp"]
DEFAULT_MPI_COUNTS = [4, 6, 8, 10, 12, 14]
DEFAULT_OMP_THREADS = [4, 6, 8, 10, 12, 14]
DEFAULT_N = 2000


@dataclass
class RunResult:
    program: str
    module_name: str
    kind: str
    count: int
    n: int
    runs_completed: int
    avg_time_sec: str
    raw_times_sec: str
    status: str


def parse_time_taken(output: str) -> Tuple[Optional[str], Optional[float]]:
    module_name: Optional[str] = None
    time_sec: Optional[float] = None
    for line in output.splitlines():
        match = TIME_PATTERN.search(line)
        if match:
            module_name = match.group(1).strip()
            time_sec = float(match.group(2))
    return module_name, time_sec


def _mean(values: Iterable[float]) -> Optional[float]:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    if count == 0:
        return None
    return total / count


def _program_kind(program: str) -> str:
    if program.endswith("_mpi"):
        return "mpi"
    if program.endswith("_openmp"):
        return "openmp"
    raise ValueError(f"Unsupported program name: {program}")


def run_once(
    program: str,
    exe_path: Path,
    count: int,
    n: int,
    mpirun_cmd: str,
) -> Tuple[Optional[str], Optional[float], str]:
    dims = [str(n), str(n), str(n), str(n)]
    kind = _program_kind(program)

    if kind == "mpi":
        cmd = [mpirun_cmd, "-np", str(count), str(exe_path), *dims]
    else:
        cmd = [str(exe_path), "-t", str(count), *dims]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    mixed_output = completed.stdout + "\n" + completed.stderr
    module_name, time_sec = parse_time_taken(mixed_output)

    if completed.returncode != 0:
        return module_name, None, f"error(exit={completed.returncode})"
    if time_sec is None:
        return module_name, None, "error(no_time_taken_line)"
    return module_name, time_sec, "ok"


def benchmark_program(
    program: str,
    exe_path: Path,
    counts: Sequence[int],
    n: int,
    runs: int,
    mpirun_cmd: str,
) -> List[RunResult]:
    rows: List[RunResult] = []
    module_name_for_program = program
    kind = _program_kind(program)

    for count in counts:
        times: List[float] = []
        runs_completed = 0
        status = "ok"
        last_module_name: Optional[str] = None

        for _ in range(runs):
            module_name, time_sec, run_status = run_once(
                program=program,
                exe_path=exe_path,
                count=count,
                n=n,
                mpirun_cmd=mpirun_cmd,
            )

            if module_name:
                last_module_name = module_name

            if run_status != "ok":
                status = run_status
                break

            if time_sec is None:
                status = "error(no_time_taken_line)"
                break

            times.append(time_sec)
            runs_completed += 1

        if last_module_name:
            module_name_for_program = last_module_name

        avg_time = _mean(times)
        rows.append(
            RunResult(
                program=program,
                module_name=module_name_for_program,
                kind=kind,
                count=count,
                n=n,
                runs_completed=runs_completed,
                avg_time_sec="" if avg_time is None else f"{avg_time:.6f}",
                raw_times_sec=";".join(f"{t:.6f}" for t in times),
                status=status,
            )
        )

        print(
            f"[{program}] n={n}, {kind}={count}: module={module_name_for_program}, "
            f"runs={runs_completed}/{runs}, status={status}, "
            f"avg={'' if avg_time is None else f'{avg_time:.6f}'}"
        )

    return rows


def write_csv(path: Path, rows: List[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "program",
                "module_name",
                "kind",
                "count",
                "n",
                "runs_completed",
                "avg_time_sec",
                "status",
                "raw_times_sec",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.program,
                    row.module_name,
                    row.kind,
                    row.count,
                    row.n,
                    row.runs_completed,
                    row.avg_time_sec,
                    row.status,
                    row.raw_times_sec,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MPI/OpenMP programs across process/thread counts."
    )
    parser.add_argument(
        "--programs",
        nargs="+",
        default=DEFAULT_PROGRAMS,
        choices=["MM_RS_mpi", "MM_RS_openmp", "MM_cblas_mpi", "MM_cblas_openmp"],
        help="Programs to benchmark. Default: MM_RS_mpi MM_RS_openmp.",
    )
    parser.add_argument(
        "--mpi-counts",
        nargs="+",
        type=int,
        default=DEFAULT_MPI_COUNTS,
        help="MPI process counts to test. Default: 4 6 8 10 12 14.",
    )
    parser.add_argument(
        "--omp-threads",
        nargs="+",
        type=int,
        default=DEFAULT_OMP_THREADS,
        help="OpenMP thread counts to test. Default: 4 6 8 10 12 14.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help="Square matrix size n for A(n,n) and B(n,n). Default: 2000.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Runs per (program, count). Default: 5.",
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="Directory containing executables. Default: build.",
    )
    parser.add_argument(
        "--mpirun",
        default="mpirun",
        help="MPI launcher command. Default: mpirun.",
    )
    parser.add_argument(
        "--out",
        default="report/data/perf_results_node_scaling.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = (repo_root / args.build_dir).resolve()
    out_path = (repo_root / args.out).resolve()

    missing = [program for program in args.programs if not (build_dir / program).exists()]
    if missing:
        print("Missing executables in build dir: " + ", ".join(missing), file=sys.stderr)
        return 1

    all_rows: List[RunResult] = []
    for program in args.programs:
        counts = args.mpi_counts if _program_kind(program) == "mpi" else args.omp_threads
        rows = benchmark_program(
            program=program,
            exe_path=build_dir / program,
            counts=counts,
            n=args.n,
            runs=args.runs,
            mpirun_cmd=args.mpirun,
        )
        all_rows.extend(rows)

    write_csv(out_path, all_rows)
    print(f"Wrote {len(all_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
