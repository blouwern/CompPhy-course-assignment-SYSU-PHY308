#!/usr/bin/env python3
"""Benchmark six matrix-multiplication executables across square sizes.

Requirements implemented:
1. Test programs:
   MM_cblas_mpi, MM_cblas_openmp, MM_cblas_seq,
   MM_RS_mpi, MM_RS_openmp, MM_RS_seq
2. Matrix sizes n in [10, 100, 500, 1000, 5000, 10000], using n*n by n*n.
3. Run each (program, n) 5 times and average the parsed [Time taken] value.
4. Use <module_name> in [Time taken]<module_name> as algorithm name in output.
5. For MM_RS_seq only: if one run exceeds 300s, interrupt remaining runs for
   that n and record result as >300sec.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TIME_LINE_PATTERN = re.compile(
	r"\[Time taken\]\s*<([^>]+)>\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds"
)

TARGET_PROGRAMS = [
	"MM_cblas_mpi",
	"MM_cblas_openmp",
	"MM_cblas_seq",
	"MM_RS_mpi",
	"MM_RS_openmp",
	"MM_RS_seq",
]

DEFAULT_N_VALUES = [10, 100, 500, 1000, 5000, 10000]


@dataclass
class BenchResult:
	program: str
	module_name: str
	n: int
	runs_completed: int
	avg_time_sec: str
	status: str
	raw_times_sec: str


def parse_time_taken(output: str) -> Tuple[Optional[str], Optional[float]]:
	module_name: Optional[str] = None
	time_sec: Optional[float] = None

	for line in output.splitlines():
		match = TIME_LINE_PATTERN.search(line)
		if match:
			module_name = match.group(1).strip()
			time_sec = float(match.group(2))

	return module_name, time_sec


def normalize_process_output(raw: object) -> str:
	if raw is None:
		return ""
	if isinstance(raw, bytes):
		return raw.decode("utf-8", errors="replace")
	if isinstance(raw, str):
		return raw
	return str(raw)


def run_one(
	program: str,
	exe_path: Path,
	n: int,
	mpirun_cmd: str,
	mpi_np: int,
	timeout_sec: Optional[int],
) -> Tuple[Optional[str], Optional[float], str]:
	dims = [str(n), str(n), str(n), str(n)]
	if program.endswith("_mpi"):
		cmd = [mpirun_cmd, "-np", str(mpi_np), str(exe_path), *dims]
	else:
		cmd = [str(exe_path), *dims]

	try:
		completed = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			check=False,
			timeout=timeout_sec,
		)
	except subprocess.TimeoutExpired as exc:
		mixed_output = (
			normalize_process_output(exc.stdout)
			+ "\n"
			+ normalize_process_output(exc.stderr)
		)
		module_name, _ = parse_time_taken(mixed_output)
		return module_name, None, "timeout"

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
	n_values: List[int],
	runs: int,
	mpirun_cmd: str,
	mpi_np: int,
	rs_seq_timeout_sec: int,
) -> List[BenchResult]:
	rows: List[BenchResult] = []
	last_module_name: Optional[str] = None

	for n in n_values:
		times: List[float] = []
		status = "ok"
		runs_completed = 0

		for _ in range(runs):
			timeout = rs_seq_timeout_sec if program == "MM_RS_seq" else None
			module_name, time_sec, run_status = run_one(
				program=program,
				exe_path=exe_path,
				n=n,
				mpirun_cmd=mpirun_cmd,
				mpi_np=mpi_np,
				timeout_sec=timeout,
			)

			if module_name:
				last_module_name = module_name

			if run_status != "ok":
				status = run_status
				if program == "MM_RS_seq" and run_status == "timeout":
					status = ">300sec"
				break

			if time_sec is None:
				status = "error(no_time_taken_line)"
				break

			if program == "MM_RS_seq" and time_sec > rs_seq_timeout_sec:
				status = ">300sec"
				break

			times.append(time_sec)
			runs_completed += 1

		module_name_for_row = last_module_name or program
		if status == ">300sec":
			avg_str = ">300sec"
		elif times:
			avg_str = f"{(sum(times) / len(times)):.6f}"
		else:
			avg_str = ""

		rows.append(
			BenchResult(
				program=program,
				module_name=module_name_for_row,
				n=n,
				runs_completed=runs_completed,
				avg_time_sec=avg_str,
				status=status,
				raw_times_sec=";".join(f"{t:.6f}" for t in times),
			)
		)

		print(
			f"[{program}] n={n}: module={module_name_for_row}, "
			f"runs={runs_completed}/{runs}, status={status}, avg={avg_str}"
		)

	return rows


def write_csv(path: Path, rows: List[BenchResult]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"program",
				"module_name",
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
					row.n,
					row.runs_completed,
					row.avg_time_sec,
					row.status,
					row.raw_times_sec,
				]
			)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Benchmark six assignment executables and parse [Time taken]."
	)
	parser.add_argument("--runs", type=int, default=5, help="Runs per (program, n).")
	parser.add_argument(
		"--n-values",
		type=int,
		nargs="+",
		default=DEFAULT_N_VALUES,
		help="Square sizes n. Default: 10 100 500 1000 5000 10000",
	)
	parser.add_argument(
		"--build-dir",
		default="build",
		help="Directory containing executables. Default: build",
	)
	parser.add_argument(
		"--out",
		default="report/perf_results_6_programs.csv",
		help="Output CSV path.",
	)
	parser.add_argument("--mpirun", default="mpirun", help="MPI launcher command.")
	parser.add_argument(
		"--mpi-np",
		type=int,
		# default=4,
		help="Process count for MPI executables.",
	)
	parser.add_argument(
		"--rs-seq-timeout-sec",
		type=int,
		default=300,
		help="Timeout used only for MM_RS_seq (seconds).",
	)

	args = parser.parse_args()

	repo_root = Path(__file__).resolve().parents[1]
	build_dir = (repo_root / args.build_dir).resolve()
	out_path = (repo_root / args.out).resolve()

	missing = [p for p in TARGET_PROGRAMS if not (build_dir / p).exists()]
	if missing:
		print(
			"Missing executables in build dir: " + ", ".join(missing),
			file=sys.stderr,
		)
		print("Please run build first, e.g. `cmake --build build`.", file=sys.stderr)
		return 1

	all_rows: List[BenchResult] = []
	for program in TARGET_PROGRAMS:
		exe_path = build_dir / program
		rows = benchmark_program(
			program=program,
			exe_path=exe_path,
			n_values=args.n_values,
			runs=args.runs,
			mpirun_cmd=args.mpirun,
			mpi_np=args.mpi_np,
			rs_seq_timeout_sec=args.rs_seq_timeout_sec,
		)
		all_rows.extend(rows)

	write_csv(out_path, all_rows)
	print(f"Wrote {len(all_rows)} rows to {out_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
