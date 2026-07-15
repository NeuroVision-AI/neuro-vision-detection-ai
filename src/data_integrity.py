"""Dataset provenance, duplicate auditing, and leakage-resistant splitting.

The research pipeline must split at a patient or duplicate-group level.  This
module intentionally has no PyTorch dependency so it can be run before model
training and in lightweight CI checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm"}
SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "testing": "test",
    "test": "test",
}
CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "no_tumor",
    "no_tumor": "no_tumor",
    "no-tumor": "no_tumor",
    "pituitary": "pituitary",
}


@dataclass
class ManifestRecord:
    """One source image and its research-provenance fields."""

    relative_path: str
    class_name: str
    source_split: str
    source_name: str
    patient_id: str
    exact_sha256: str
    perceptual_hash: str
    group_id: str
    width: int
    height: int
    mode: str
    decode_status: str
    assigned_split: str = ""
    materialized_path: str = ""


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 hash for *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_grayscale(path: Path) -> Image.Image:
    if path.suffix.lower() != ".dcm":
        with Image.open(path) as image:
            return image.convert("L").copy()

    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array
    try:
        pixels = apply_voi_lut(pixels, ds)
    except Exception:
        pass
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim > 2:
        pixels = pixels[pixels.shape[0] // 2]
    lo, hi = float(np.min(pixels)), float(np.max(pixels))
    normalized = np.zeros_like(pixels) if hi <= lo else (pixels - lo) / (hi - lo)
    return Image.fromarray((normalized * 255).astype(np.uint8), mode="L")


def perceptual_dhash(path: Path, hash_size: int = 8) -> Tuple[str, int, int, str]:
    """Return a deterministic difference hash and basic decoded image metadata."""
    image = _load_grayscale(path)
    width, height = image.size
    mode = image.mode
    resized = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}", width, height, mode


def extract_patient_id(filename: str, patient_pattern: Optional[str]) -> str:
    """Extract a patient identifier using a named/first regex capture group."""
    if not patient_pattern:
        return ""
    match = re.search(patient_pattern, filename)
    if not match:
        return ""
    if "patient" in match.groupdict():
        return str(match.group("patient"))
    if match.groups():
        return str(match.group(1))
    return str(match.group(0))


def _class_and_source_split(relative: Path) -> Tuple[str, str]:
    parts = relative.parts
    if len(parts) < 2:
        return "", "unknown"
    first = parts[0].lower()
    if first in SPLIT_ALIASES and len(parts) >= 3:
        return CLASS_ALIASES.get(parts[1].lower(), parts[1].lower()), SPLIT_ALIASES[first]
    return CLASS_ALIASES.get(parts[0].lower(), parts[0].lower()), "unknown"


def build_manifest(
    source_root: Path,
    source_name: str = "local",
    patient_pattern: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
) -> List[ManifestRecord]:
    """Scan an image corpus and build a provenance/duplicate manifest.

    Supported layouts are ``root/class/image`` and
    ``root/{Training,Testing}/class/image``.  When patient identifiers are not
    available, exact/perceptual duplicate groups become the split unit.
    """
    source_root = Path(source_root).resolve()
    allowed = set(class_names or CLASS_ALIASES.values())
    records: List[ManifestRecord] = []

    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(source_root)
        class_name, source_split = _class_and_source_split(relative)
        if class_name not in allowed:
            continue

        exact_hash = sha256_file(path)
        patient_id = extract_patient_id(path.name, patient_pattern)
        try:
            dhash, width, height, mode = perceptual_dhash(path)
            decode_status = "ok"
        except Exception as exc:
            dhash, width, height, mode = "", 0, 0, ""
            decode_status = f"error:{type(exc).__name__}"

        if patient_id:
            group_id = f"patient:{patient_id}"
        elif dhash:
            group_id = f"dhash:{dhash}"
        else:
            group_id = f"sha256:{exact_hash}"

        records.append(
            ManifestRecord(
                relative_path=relative.as_posix(),
                class_name=class_name,
                source_split=source_split,
                source_name=source_name,
                patient_id=patient_id,
                exact_sha256=exact_hash,
                perceptual_hash=dhash,
                group_id=group_id,
                width=width,
                height=height,
                mode=mode,
                decode_status=decode_status,
            )
        )
    _assign_connected_group_ids(records)
    return records


def _assign_connected_group_ids(records: Sequence[ManifestRecord]) -> None:
    """Union patient, exact-hash, and perceptual-hash links transitively.

    A copied image may have a different filename/patient parse. Treating these
    identifiers independently would still allow the exact pixels to cross a
    split, so all available identity signals form one connected component.
    """
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field in ("patient_id", "exact_sha256", "perceptual_hash"):
        first_seen: Dict[str, int] = {}
        for index, record in enumerate(records):
            value = getattr(record, field)
            if not value:
                continue
            if value in first_seen:
                union(index, first_seen[value])
            else:
                first_seen[value] = index

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    for members in components.values():
        patient_ids = sorted({records[index].patient_id for index in members if records[index].patient_id})
        component_key = "|".join(sorted(records[index].relative_path for index in members))
        if len(patient_ids) == 1:
            group_id = f"patient:{patient_ids[0]}"
        else:
            group_id = f"component:{hashlib.sha256(component_key.encode('utf-8')).hexdigest()[:20]}"
        for index in members:
            records[index].group_id = group_id


def _duplicates(records: Sequence[ManifestRecord], field: str) -> Dict[str, List[ManifestRecord]]:
    grouped: Dict[str, List[ManifestRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, field)
        if value:
            grouped[value].append(record)
    return {key: value for key, value in grouped.items() if len(value) > 1}


def audit_manifest(records: Sequence[ManifestRecord]) -> Dict[str, object]:
    """Return leakage, duplicate, label-conflict, and decode diagnostics."""
    exact_duplicates = _duplicates(records, "exact_sha256")
    perceptual_duplicates = _duplicates(records, "perceptual_hash")
    group_splits: Dict[str, set] = defaultdict(set)
    hash_splits: Dict[str, set] = defaultdict(set)
    group_labels: Dict[str, set] = defaultdict(set)
    class_counts = Counter()
    split_counts = Counter()

    for record in records:
        class_counts[record.class_name] += 1
        if record.assigned_split:
            split_counts[record.assigned_split] += 1
            group_splits[record.group_id].add(record.assigned_split)
            hash_splits[record.exact_sha256].add(record.assigned_split)
        group_labels[record.group_id].add(record.class_name)

    cross_split_groups = sorted(key for key, splits in group_splits.items() if len(splits) > 1)
    cross_split_hashes = sorted(key for key, splits in hash_splits.items() if len(splits) > 1)
    cross_label_groups = sorted(key for key, labels in group_labels.items() if len(labels) > 1)
    invalid = [record.relative_path for record in records if record.decode_status != "ok"]

    return {
        "records": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "assigned_split_counts": dict(sorted(split_counts.items())),
        "unique_patients": len({record.patient_id for record in records if record.patient_id}),
        "unique_groups": len({record.group_id for record in records}),
        "exact_duplicate_groups": len(exact_duplicates),
        "perceptual_duplicate_groups": len(perceptual_duplicates),
        "cross_label_groups": cross_label_groups,
        "cross_split_groups": cross_split_groups,
        "cross_split_exact_hashes": cross_split_hashes,
        "invalid_images": invalid,
        "leakage_free": not cross_split_groups and not cross_split_hashes,
        "label_consistent": not cross_label_groups,
        "all_images_decodable": not invalid,
    }


def assign_grouped_splits(
    records: Sequence[ManifestRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    preserve_source_test: bool = True,
) -> List[ManifestRecord]:
    """Assign whole patient/duplicate groups to deterministic stratified splits.

    If an official source test split exists, every group touching it is locked to
    test and only the remaining source-training groups are divided into train/val.
    """
    if not records:
        raise ValueError("No image records were found")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios cannot be negative")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    grouped: Dict[str, List[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)

    official_test_exists = preserve_source_test and any(
        record.source_split == "test" for record in records
    )
    locked_test = {
        group_id for group_id, members in grouped.items()
        if official_test_exists and any(member.source_split == "test" for member in members)
    }

    for group_id in locked_test:
        for record in grouped[group_id]:
            record.assigned_split = "test"

    candidates = ["train", "val"] if official_test_exists else ["train", "val", "test"]
    raw_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    denominator = sum(raw_ratios[name] for name in candidates)
    ratios = {name: raw_ratios[name] / denominator for name in candidates}

    by_class: Dict[str, List[Tuple[str, List[ManifestRecord]]]] = defaultdict(list)
    for group_id, members in grouped.items():
        if group_id in locked_test:
            continue
        dominant_class = Counter(member.class_name for member in members).most_common(1)[0][0]
        by_class[dominant_class].append((group_id, members))

    rng = random.Random(seed)
    for class_name in sorted(by_class):
        groups = by_class[class_name]
        rng.shuffle(groups)
        groups.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(members) for _, members in groups)
        targets = {name: total * ratios[name] for name in candidates}
        counts = {name: 0 for name in candidates}

        for _, members in groups:
            def allocation_score(name: str) -> Tuple[float, float, str]:
                target = max(targets[name], 1.0)
                return (counts[name] / target, counts[name], name)

            chosen = min(candidates, key=allocation_score)
            for record in members:
                record.assigned_split = chosen
            counts[chosen] += len(members)

    return list(records)


def write_manifest(records: Sequence[ManifestRecord], path: Path) -> None:
    """Write the manifest as CSV for human and machine audit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ManifestRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> List[ManifestRecord]:
    """Read a CSV previously created by :func:`write_manifest`."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [ManifestRecord(**row) for row in csv.DictReader(handle)]


def group_ids_for_materialized_samples(
    samples: Sequence[Tuple[Path, int]],
    manifest_path: Path,
    materialized_root: Path,
) -> List[str]:
    """Align manifest group IDs to a dataset's deterministic sample order."""
    lookup: Dict[str, str] = {}
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            materialized = row.get("materialized_path", "")
            if materialized:
                lookup[Path(materialized).as_posix()] = row["group_id"]

    materialized_root = Path(materialized_root).resolve()
    relative_paths = [Path(path).resolve().relative_to(materialized_root).as_posix() for path, _ in samples]
    missing = [path for path in relative_paths if path not in lookup]
    if missing:
        raise ValueError(f"Manifest is missing {len(missing)} evaluated files; first missing: {missing[0]}")
    return [lookup[path] for path in relative_paths]


def materialize_splits(
    records: Sequence[ManifestRecord],
    source_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> None:
    """Copy assigned records into ``split/class`` folders without name collisions."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        if not overwrite:
            raise FileExistsError(
                f"{output_root} is not empty. Re-run with explicit overwrite after archiving it."
            )
        shutil.rmtree(output_root)

    for record in records:
        if record.assigned_split not in {"train", "val", "test"}:
            raise ValueError(f"Unassigned record: {record.relative_path}")
        source = source_root / record.relative_path
        prefix = hashlib.sha1(record.relative_path.encode("utf-8")).hexdigest()[:10]
        destination = output_root / record.assigned_split / record.class_name / f"{prefix}_{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record.materialized_path = destination.relative_to(output_root).as_posix()


def prepare_research_dataset(
    source_root: Path,
    output_root: Path,
    manifest_dir: Path,
    source_name: str,
    patient_pattern: Optional[str] = None,
    seed: int = 42,
    preserve_source_test: bool = True,
    overwrite: bool = False,
    exclude_conflicting_groups: bool = False,
) -> Dict[str, object]:
    """End-to-end manifest, grouped split, materialization, and audit."""
    records = build_manifest(
        source_root=source_root,
        source_name=source_name,
        patient_pattern=patient_pattern,
        class_names=sorted(set(CLASS_ALIASES.values())),
    )
    assign_grouped_splits(
        records,
        seed=seed,
        preserve_source_test=preserve_source_test,
    )
    pre_audit = audit_manifest(records)
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(records, manifest_dir / "dataset_manifest_pre_exclusion.csv")
    (manifest_dir / "dataset_audit_pre_exclusion.json").write_text(
        json.dumps(pre_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    conflicting_groups = set(pre_audit["cross_label_groups"])
    excluded_records = [record for record in records if record.group_id in conflicting_groups]
    if not pre_audit["label_consistent"]:
        if not exclude_conflicting_groups:
            raise ValueError(
                "The same patient/duplicate group has conflicting class labels. "
                "The failed audit was saved; adjudicate the records or re-run with "
                "explicit conflict-group exclusion."
            )
        records = [record for record in records if record.group_id not in conflicting_groups]
        write_manifest(excluded_records, manifest_dir / "excluded_conflicting_records.csv")
    if not pre_audit["all_images_decodable"]:
        raise ValueError("One or more images cannot be decoded; inspect the audit before training.")
    if not pre_audit["leakage_free"]:
        raise ValueError("Grouped split audit found cross-split leakage.")

    materialize_splits(records, source_root, output_root, overwrite=overwrite)
    write_manifest(records, manifest_dir / "dataset_manifest.csv")
    audit = audit_manifest(records)
    audit["source_records"] = pre_audit["records"]
    audit["source_cross_label_groups"] = pre_audit["cross_label_groups"]
    audit["excluded_conflicting_group_count"] = len(conflicting_groups)
    audit["excluded_conflicting_record_count"] = len(excluded_records)
    audit["excluded_conflicting_records"] = [record.relative_path for record in excluded_records]
    (manifest_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit
