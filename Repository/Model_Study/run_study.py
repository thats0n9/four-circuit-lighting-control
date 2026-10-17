"""Public command line for the archived BSER&T lighting study.

The numerical kernels in Study implement the archived calculations. This controller
validates the archive and runs write-producing tools in separate directories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "Study"
sys.path.insert(0, str(STUDY))
from input_storage import materialise_inputs, verify_shared_inputs
CORE_PACKAGES = {"numpy": "2.3.5", "numba": "0.68.0", "llvmlite": "0.50.0"}
FIGURE_PACKAGES = {"numpy": "2.3.5", "matplotlib": "3.11.2"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict:
    versions = {}
    for name in sorted(CORE_PACKAGES.keys() | FIGURE_PACKAGES.keys()):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": sys.platform, "packages": versions,
            "initial_thread_environment": {name: os.environ.get(name) for name in
                ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}}


def check_environment(kind: str) -> dict:
    found = environment()
    expected = FIGURE_PACKAGES if kind == "draw" else CORE_PACKAGES
    failures = []
    if sys.version_info[:2] != (3, 12):
        failures.append("Python 3.12 is required for the declared reproduction environment")
    for name, version in expected.items():
        if found["packages"][name] != version:
            failures.append(f"{name}=={version} is required; found {found['packages'][name]}")
    if failures:
        raise ValueError("; ".join(failures) + ". Install the applicable requirements file.")
    return found


def manifest_check() -> dict:
    manifest = ROOT / "file_manifest.csv"
    if not manifest.is_file():
        raise ValueError("file_manifest.csv is missing")
    checked = 0
    failures = []
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "bytes", "sha256"]:
            raise ValueError("Unexpected manifest columns")
        for row in reader:
            relative = Path(row["path"])
            path = (ROOT / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(ROOT.resolve()):
                raise ValueError(f"Manifest path escapes the repository: {row['path']}")
            good = path.is_file() and path.stat().st_size == int(row["bytes"])
            good = good and digest(path) == row["sha256"]
            if not good:
                failures.append(row["path"])
            checked += 1
    if failures:
        raise ValueError("Archive integrity failed for " + ", ".join(failures[:12]))
    if checked < 602:
        raise ValueError("Manifest is incomplete; this archive contains 602 current scientific files")
    return {"status": "passed", "files_checked": checked, **verify_shared_inputs()}


def new_destination(value: str | None, command: str) -> Path:
    runs = ROOT / "runs"
    if value:
        destination = Path(value).expanduser().resolve()
        if destination.exists():
            raise ValueError("Choose a new destination; existing files are never overwritten")
        if destination.is_relative_to(ROOT.resolve()) and not destination.is_relative_to(runs.resolve()):
            raise ValueError("Inside the repository, outputs must be under runs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination
    runs.mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=command + "_", dir=runs)).resolve()
    assert destination.is_relative_to(runs.resolve())
    # The numerical runner itself creates its destination.
    destination.rmdir()
    return destination


def child_environment() -> dict:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Preserve the launch environment of the archived runner. Individual kernels
    # retain their original thread settings. Setting a new BLAS limit before
    # NumPy initialises changes last-bit optical calculations in this archive.
    return env


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    with log.open("x", encoding="utf-8") as output:
        child = subprocess.Popen(command, cwd=cwd, env=child_environment(), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert child.stdout is not None
        for line in child.stdout:
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def save_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def verify(destination: Path, software: dict, integrity: dict) -> None:
    sys.path.insert(0, str(ROOT / 'documentation'))
    from verify_data_catalog import verify_catalog
    catalog = verify_catalog(ROOT)
    destination.mkdir()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="work_", dir=destination) as name:
        scratch = Path(name).resolve()
        assert scratch.is_relative_to(destination.resolve())
        # Numerical verification runs on the archived data. Its generated
        # verification file and any JIT cache are confined to this scratch copy.
        shutil.copytree(STUDY, scratch / "Study", ignore=shutil.ignore_patterns(
            "figure_redraw", "figures_for_submission_en", "__pycache__", "*.pyc", "*.nbi", "*.nbc"))
        local = scratch / "Study"
        materialise_inputs(local)
        run_logged([sys.executable, "-B", "verify_analysis.py"], local, destination / "verification.log")
        result = json.loads((local / "analysis_results/verification.json").read_text("utf-8"))
        if result.get("status") != "passed":
            raise ValueError("Numerical invariant checks did not pass")
        save_report(destination / "numerical_invariants.json", result)
        # Explicitly check the resolved target before the temporary context
        # recursively removes its own newly created work directory.
        assert scratch.is_relative_to(destination.resolve())
    save_report(destination / "verification_report.json", {
        "status": "passed", "manifest": integrity, "invariants_passed": len(result["checks"]),
        "data_catalog": catalog,
        "software": software, "elapsed_seconds": time.perf_counter() - started,
        "scope": "Archive integrity and numerical invariant checks"})


def reproduce(destination: Path, workers: int, software: dict, integrity: dict, strict_bytes: bool) -> None:
    started = time.perf_counter()
    # Keep the log beside, rather than inside, the initially nonexistent target.
    log = destination.parent / (destination.name + ".log")
    if log.exists():
        raise ValueError("The adjacent log already exists; choose another destination")
    try:
        run_logged([sys.executable, "-B", str(STUDY / "run_analysis.py"), "--workers", str(workers),
                    "--destination", str(destination)], STUDY, log)
    except subprocess.CalledProcessError:
        # The analysis runner deliberately fails on any nonidentical byte.
        # Continue only when the complete scientific pipeline and its invariant
        # checks reached the final archive-comparison step.
        audit=destination / "reproduction_audit.json"
        invariant=destination / "analysis_results/verification.json"
        if not audit.is_file() or not invariant.is_file() or json.loads(invariant.read_text('utf-8')).get('status')!='passed':
            raise
    result = json.loads((destination / "reproduction_audit.json").read_text("utf-8"))
    from archive_comparison import compare_archives
    comparison=compare_archives(STUDY,destination)
    save_report(destination / "scientific_comparison.json",comparison)
    print(json.dumps({k:comparison[k] for k in ["science_files","byte_identical","numerically_equivalent","max_absolute_floating_difference"]}),flush=True)
    if comparison["science_files"]!=602 or comparison["numerically_equivalent"]!=602:
        raise ValueError("Inspect scientific_comparison.json: scientific values or discrete results differ")
    if strict_bytes and comparison["byte_identical"]!=602:
        raise ValueError("Strict byte mode requires all 602 file hashes to match")
    save_report(destination / "execution_report.json", {
        "status": "passed", "manifest": integrity, "software": software, "workers": workers,
        "science_files": 602, "byte_identical": comparison["byte_identical"],
        "numerically_equivalent":602,"strict_bytes":strict_bytes,
        "max_absolute_floating_difference":comparison["max_absolute_floating_difference"],
        "float_absolute_tolerance":comparison["float_absolute_tolerance"],
        "float_relative_tolerance":comparison["float_relative_tolerance"],
        "elapsed_seconds": time.perf_counter() - started})


def draw(destination: Path, figure: int | None, font_dir: str | None, software: dict, integrity: dict) -> None:
    started = time.perf_counter()
    log = destination.parent / (destination.name + ".log")
    if log.exists():
        raise ValueError("The adjacent log already exists; choose another destination")
    command = [sys.executable, "-B", str(STUDY / "figure_redraw/redraw_all.py"), "--output", str(destination)]
    if figure is not None:
        command += ["--figure", str(figure)]
    if font_dir:
        command += ["--font-dir", str(Path(font_dir).expanduser().resolve())]
    run_logged(command, STUDY, log)
    result = json.loads((destination / "figure_style_audit.json").read_text("utf-8"))
    expected = 11 if figure is None else 1
    if len(result["figures"]) != expected:
        raise ValueError("Unexpected number of redrawn figures")
    save_report(destination / "redraw_report.json", {
        "status": "passed", "manifest": integrity, "software": software,
        "figures_drawn": expected, "elapsed_seconds": time.perf_counter() - started,
        "scope": "Redraw from exact supplied plotting CSVs; the numerical study is not rerun"})


def positive_workers(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 36:
        raise argparse.ArgumentTypeError("workers must be between 1 and 36")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Show release and package versions without running the study")
    commands.add_parser("check-files", help="Check every archived file hash without scientific dependencies")
    comparison=commands.add_parser("compare",help="Compare an already generated numerical directory without rerunning calculations")
    comparison.add_argument("--generated",required=True)
    comparison.add_argument("--strict-bytes",action="store_true")
    for name, helptext in [("verify", "Check archived numerical invariants in an isolated copy"),
                           ("reproduce", "Rebuild and compare all 602 scientific files"),
                           ("draw", "Redraw figures from exact plotting tables")]:
        child = commands.add_parser(name, help=helptext)
        child.add_argument("--destination", help="New output directory; default is a new directory under runs")
        if name == "reproduce":
            child.add_argument("--workers", type=positive_workers, default=4)
            child.add_argument("--strict-bytes",action="store_true",help="Require exact file bytes as well as scientific equivalence")
        if name == "draw":
            child.add_argument("--figure", type=int, choices=range(1, 12), help="1–7: numerical-supplement figures M1–M7; 8–11: S1–S4")
            child.add_argument("--font-dir", help="Directory containing installed/licensed Times New Roman TTF files")
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            print(json.dumps({"release": json.loads((ROOT / "release.json").read_text("utf-8")),
                              "software": environment()}, ensure_ascii=False, indent=2))
            return 0
        integrity = manifest_check()
        if args.command == "check-files":
            print(json.dumps(integrity, indent=2))
            return 0
        software = check_environment(args.command)
        if args.command=="compare":
            from archive_comparison import compare_archives
            comparison=compare_archives(STUDY,Path(args.generated).expanduser().resolve())
            print(json.dumps({k:v for k,v in comparison.items() if k!="files"},indent=2))
            passed=comparison["science_files"]==comparison["numerically_equivalent"]==602
            passed=passed and (not args.strict_bytes or comparison["byte_identical"]==602)
            return 0 if passed else 1
        destination = new_destination(args.destination, args.command)
        if args.command == "verify":
            verify(destination, software, integrity)
        elif args.command == "reproduce":
            reproduce(destination, args.workers, software, integrity,args.strict_bytes)
        else:
            draw(destination, args.figure, args.font_dir, software, integrity)
        print("Completed. Results: " + str(destination), flush=True)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
