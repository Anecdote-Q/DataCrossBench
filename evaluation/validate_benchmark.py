"""Validate the publishable DataCrossBench directory without calling an API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(bench_dir: Path, classification_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    flags = sorted(bench_dir.glob("flag-*"), key=lambda p: p.name)
    if len(flags) != 200:
        errors.append(f"expected 200 flag directories, found {len(flags)}")

    seen: set[str] = set()
    for flag_dir in flags:
        flag_id = flag_dir.name
        if flag_id in seen:
            errors.append(f"duplicate flag directory: {flag_id}")
        seen.add(flag_id)
        meta_path = flag_dir / "meta-info.json"
        if not meta_path.exists():
            errors.append(f"{flag_id}: missing meta-info.json")
            continue
        try:
            meta = load_json(meta_path)
        except Exception as exc:
            errors.append(f"{meta_path}: invalid JSON ({exc})")
            continue
        if meta.get("flag_id") != flag_id:
            errors.append(f"{meta_path}: flag_id does not match directory")
        if not isinstance(meta.get("goal"), str) or not meta["goal"].strip():
            errors.append(f"{meta_path}: goal must be a non-empty string")
        if not isinstance(meta.get("insights"), list) or not all(
            isinstance(item, str) for item in meta.get("insights", [])
        ):
            errors.append(f"{meta_path}: insights must be a string list")
        for relative_path in meta.get("input_files", []):
            if not (flag_dir / relative_path).exists():
                errors.append(f"{flag_id}: missing input file {relative_path}")

        if (flag_dir / "output_csv_origin").exists():
            errors.append(f"{flag_id}: output_csv_origin should be removed")
        if (flag_dir / "output" / "processed").exists():
            errors.append(f"{flag_id}: output/processed should be removed")

    cleaned_path = bench_dir / "cleaned_insights_200.json"
    if cleaned_path.exists():
        try:
            cleaned = load_json(cleaned_path)
            cleaned_ids = {item.get("flag_id") for item in cleaned} if isinstance(cleaned, list) else set()
            if cleaned_ids != seen:
                errors.append("cleaned_insights_200.json does not cover exactly the flag directories")
        except Exception as exc:
            errors.append(f"{cleaned_path}: invalid JSON ({exc})")

    if classification_path and classification_path.exists():
        try:
            classification = load_json(classification_path).get("flags", {})
            if set(classification) != seen:
                errors.append("classification file does not cover exactly the flag directories")
            for flag_dir in flags:
                images = [
                    p for p in (flag_dir / "output").iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                ]
                expected = "hard" if images else "easy"
                if classification.get(flag_dir.name) != expected:
                    errors.append(f"{flag_dir.name}: easy/hard classification mismatch")
        except Exception as exc:
            errors.append(f"{classification_path}: invalid JSON ({exc})")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DataCrossBench files and metadata")
    parser.add_argument("bench", type=Path, help="DataCrossBench root directory")
    parser.add_argument("--classification", type=Path, default=None)
    args = parser.parse_args()
    errors = validate(args.bench.expanduser().resolve(), args.classification)
    if errors:
        print("Validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("DataCrossBench validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
