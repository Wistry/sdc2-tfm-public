#!/usr/bin/env python3
"""Audit equivalence between original 10GB SoFiA configs and 40 GB extended validation configs."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


VALIDATION_ROOT = Path(__file__).resolve().parents[1]


ALLOWED_KEYS = {
    "input.data",
    "input.cube",
    "input.file",
    "data.path",
    "cube.path",
    "output.directory",
    "output.filename",
    "output.prefix",
    "output.catalog",
    "output.catalogue",
    "catalog.output",
    "catalog.path",
    "log.path",
    "log.file",
}
ALLOWED_KEY_FRAGMENTS = ("input", "data", "cube", "fits", "output", "catalog", "catalogue", "prefix", "filename", "directory", "log")


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def normalise_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            line = f"{key.strip()} = {value.strip()}"
        else:
            line = " ".join(line.split())
        lines.append(line)
    return lines


def line_key(line: str) -> str:
    if "=" not in line:
        return ""
    return line.split("=", 1)[0].strip()


def is_allowed_key(key: str) -> bool:
    lower = key.lower()
    if lower in ALLOWED_KEYS:
        return True
    return any(fragment in lower for fragment in ALLOWED_KEY_FRAGMENTS) and (
        lower.startswith("input.") or lower.startswith("output.") or lower.startswith("log.") or "catalog" in lower
    )


def key_value_map(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        key = line_key(line)
        if key:
            out[key] = line.split("=", 1)[1].strip()
        else:
            out[line] = ""
    return out


def compare_configs(original: Path, new: Path, name: str, out_dir: Path) -> dict[str, Any]:
    original_lines = normalise_lines(read_text(original))
    new_lines = normalise_lines(read_text(new))
    full_diff = list(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=str(original),
            tofile=str(new),
            lineterm="",
        )
    )
    original_map = key_value_map(original_lines)
    new_map = key_value_map(new_lines)
    all_keys = sorted(set(original_map) | set(new_map))
    critical: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    filtered_lines: list[str] = []

    for key in all_keys:
        old = original_map.get(key)
        new_value = new_map.get(key)
        if old == new_value:
            continue
        row = {"key": key, "original": old, "new": new_value}
        if is_allowed_key(key):
            allowed.append(row)
            continue
        critical.append(row)
        filtered_lines.append(f"[CRITICAL] {key}")
        filtered_lines.append(f"  original: {old}")
        filtered_lines.append(f"  new:      {new_value}")

    full_path = out_dir / f"{name}_config_diff_full.txt"
    filtered_path = out_dir / f"{name}_config_diff_filtered.txt"
    full_path.write_text("\n".join(full_diff) + ("\n" if full_diff else "No differences.\n"), encoding="utf-8")
    if filtered_lines:
        filtered_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
    else:
        filtered_path.write_text("No critical differences after ignoring allowed path/output changes.\n", encoding="utf-8")

    return {
        "original_config": str(original),
        "new_config": str(new),
        "only_allowed_differences": not critical,
        "allowed_differences": allowed,
        "critical_differences": critical,
        "full_diff_path": str(full_path),
        "filtered_diff_path": str(filtered_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-baseline-config", type=Path, required=True)
    parser.add_argument("--original-sdc2-config", type=Path, required=True)
    parser.add_argument("--new-baseline-config", type=Path, required=True)
    parser.add_argument("--new-sdc2-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=VALIDATION_ROOT / "outputs" / "config_audit")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_current": compare_configs(
            args.original_baseline_config,
            args.new_baseline_config,
            "baseline_current",
            args.out_dir,
        ),
        "sdc2_team_sofia_like": compare_configs(
            args.original_sdc2_config,
            args.new_sdc2_config,
            "sdc2_team_sofia_like",
            args.out_dir,
        ),
    }
    summary_path = args.out_dir / "config_equivalence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    any_critical = False
    for name, result in summary.items():
        if result["only_allowed_differences"]:
            print(f"[OK] {name}_40gb is equivalent to {name} except allowed path/output changes")
        else:
            any_critical = True
            print("[WARN] Critical differences found. Do not run SoFiA until reviewed.")
            print(f"[WARN] {name}:")
            for diff in result["critical_differences"]:
                print(f"  - {diff['key']}: {diff['original']} -> {diff['new']}")
    print(f"Summary: {summary_path}")
    if any_critical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
