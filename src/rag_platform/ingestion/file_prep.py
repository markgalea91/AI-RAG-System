from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set, Union

import filetype

ALLOWED_EXTENSIONS: Set[str] = {"pdf", "json", "txt", "docx"}  # add others later if needed


def detect_extension(path: Union[str, Path]) -> Optional[str]:
    p = Path(path)
    kind = filetype.guess(str(p))
    if kind and kind.extension:
        return kind.extension.lower()
    suffix = p.suffix.lower().lstrip(".")
    return suffix if suffix else None


def safe_move(target_dir: Path, file_path: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / file_path.name

    if not dest.exists():
        shutil.move(str(file_path), str(dest))
        return dest

    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        candidate = target_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            shutil.move(str(file_path), str(candidate))
            return candidate
        i += 1


def write_extension_report(counter: Counter, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["extension", "count"])
        for ext, count in counter.most_common():
            writer.writerow([ext, count])


def prepare_files(
    directory: Union[str, Path],
    quarantine_dir: Union[str, Path],
    recursive: bool = True,
    overwrite: bool = False,
    report_file: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Walk directory, detect true extension, rename mismatched suffixes,
    quarantine unsupported/unknown files, and return list of accepted file paths.
    """
    base = Path(directory)
    quarantine = Path(quarantine_dir)
    report_path = Path(report_file) if report_file else None

    if not base.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    final_paths: List[str] = []
    quarantined_ext_counter: Counter = Counter()

    for p in base.glob(pattern):
        if not p.is_file():
            continue

        detected_ext = detect_extension(p)

        if not detected_ext or detected_ext not in ALLOWED_EXTENSIONS:
            ext_label = detected_ext if detected_ext else "unknown"
            quarantined_ext_counter[ext_label] += 1
            moved_path = safe_move(quarantine, p)
            # keep prints out of here; service layer should log
            continue

        desired_suffix = "." + detected_ext

        # rename file if suffix doesn't match detected extension
        if p.suffix.lower() != desired_suffix:
            if p.suffix == "":
                new_p = p.with_name(p.name + desired_suffix)
            else:
                new_p = p.with_suffix(desired_suffix)

            # avoid clobbering existing file unless overwrite=True
            if new_p.exists() and not overwrite:
                stem = new_p.stem
                suf = new_p.suffix
                i = 1
                candidate = new_p
                while candidate.exists():
                    candidate = new_p.with_name(f"{stem}_{i}{suf}")
                    i += 1
                new_p = candidate

            p.rename(new_p)
            p = new_p

        final_paths.append(str(p))

    if report_path and quarantined_ext_counter:
        write_extension_report(quarantined_ext_counter, report_path)

    return final_paths