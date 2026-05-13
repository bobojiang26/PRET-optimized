#!/usr/bin/env python3
"""Convert a COCO-like MIL manifest JSON to PRET-style data_info JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def _stringify_classes(classes: Optional[Sequence[Any]]) -> Sequence[str]:
    if classes is None:
        return []
    return [str(class_name) for class_name in classes]


def _parse_int_label(raw_label: Any) -> Optional[int]:
    if isinstance(raw_label, bool):
        return None
    if isinstance(raw_label, int):
        return raw_label
    if isinstance(raw_label, float) and raw_label.is_integer():
        return int(raw_label)
    if isinstance(raw_label, str):
        text = raw_label.strip()
        if text == "":
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def normalize_label(
    raw_label: Any,
    classes: Optional[Sequence[Any]] = None,
    label_id_plus_one: bool = False,
) -> int:
    """Return a numeric label id.

    Numeric labels are used directly. Non-numeric labels are mapped to their
    index in ``classes``.
    """
    class_names = _stringify_classes(classes)
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}

    label_id = _parse_int_label(raw_label)
    if label_id is None:
        label_name = str(raw_label)
        if label_name not in class_to_index:
            raise ValueError(
                "Non-numeric label {!r} is not present in classes: {}".format(
                    raw_label, class_names
                )
            )
        label_id = class_to_index[label_name]

    if class_names:
        raw_label_name = str(raw_label)
        label_matches_class_name = raw_label_name in class_to_index
        label_matches_class_index = 0 <= label_id < len(class_names)
        if not label_matches_class_name and not label_matches_class_index:
            raise ValueError(
                "Label {!r} resolved to {}, but it is not a class name or a "
                "valid class index for classes: {}".format(raw_label, label_id, class_names)
            )

    if label_id_plus_one:
        label_id += 1
    return label_id


def pick_sample_key(
    sample: Mapping[str, Any],
    key_field: str,
    fallback_key_fields: Iterable[str],
) -> str:
    for field in (key_field, *fallback_key_fields):
        value = sample.get(field)
        if value is not None and str(value) != "":
            return str(value)
    raise ValueError(
        "Sample has no usable key. Tried fields: {}".format(
            ", ".join((key_field, *fallback_key_fields))
        )
    )


def convert_manifest_dict(
    manifest: Mapping[str, Any],
    key_field: str = "id",
    label_field: str = "label",
    label_id_plus_one: bool = False,
    fixed_test_set: bool = False,
    allow_duplicate_keys: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Counter]:
    """Convert manifest content to PRET-style data_info content."""
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Input JSON must contain a top-level list field named 'samples'.")

    classes = manifest.get("classes")
    if classes is not None and not isinstance(classes, list):
        raise ValueError("If present, top-level 'classes' must be a list.")

    output: Dict[str, Dict[str, Any]] = {}
    label_counts: Counter = Counter()
    fallback_key_fields = ("id", "slide_id", "case_id")

    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError("samples[{}] must be an object.".format(index))
        if label_field not in sample:
            raise ValueError("samples[{}] is missing label field {!r}.".format(index, label_field))

        sample_key = pick_sample_key(sample, key_field, fallback_key_fields)
        if sample_key in output and not allow_duplicate_keys:
            raise ValueError(
                "Duplicate output key {!r}. Use --allow-duplicate-keys if the "
                "last duplicate should overwrite earlier entries.".format(sample_key)
            )

        label_id = normalize_label(
            sample[label_field],
            classes=classes,
            label_id_plus_one=label_id_plus_one,
        )
        output[sample_key] = {
            "wsi_label": label_id,
            "fixed_test_set": fixed_test_set,
        }
        label_counts[label_id] += 1

    return output, label_counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a COCO-like MIL manifest JSON with info/classes/samples "
            "into PRET data_info JSON."
        )
    )
    parser.add_argument("input_json", type=Path, help="Input manifest JSON path.")
    parser.add_argument("output_json", type=Path, help="Output data_info JSON path.")
    parser.add_argument(
        "--key-field",
        default="id",
        help="Sample field used as the output JSON key. Default: id.",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Sample field containing the class label. Default: label.",
    )
    parser.add_argument(
        "--label-id-plus-one",
        action="store_true",
        help="Add 1 to every resolved label id during conversion.",
    )
    parser.add_argument(
        "--fixed-test-set",
        action="store_true",
        help="Set fixed_test_set=true for every output sample. Default is false.",
    )
    parser.add_argument(
        "--allow-duplicate-keys",
        action="store_true",
        help="Allow duplicate output keys; later samples overwrite earlier samples.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = load_json(args.input_json)
    output, label_counts = convert_manifest_dict(
        manifest,
        key_field=args.key_field,
        label_field=args.label_field,
        label_id_plus_one=args.label_id_plus_one,
        fixed_test_set=args.fixed_test_set,
        allow_duplicate_keys=args.allow_duplicate_keys,
    )
    dump_json(output, args.output_json)

    label_summary = ", ".join(
        "label {}: {}".format(label_id, count)
        for label_id, count in sorted(label_counts.items())
    )
    print(
        "Converted {} samples to {}. {}".format(
            len(output),
            args.output_json,
            label_summary,
        )
    )


if __name__ == "__main__":
    main()
