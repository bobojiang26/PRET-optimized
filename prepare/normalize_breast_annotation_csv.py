#!/usr/bin/env python3
"""Convert RuiJin breast annotation CSVs to PRET's simple CSV format.

Input CSV columns can include many metadata fields, but must contain:
    wsi_path, labels, coordinates

Output CSV columns are exactly:
    wsi_path, labels, coordinates

The label merge file is expected to contain a Python dict like:
    breast_subtype_dict = {
        "merged label": ["raw label 1", "raw label 2"],
        ...
    }
Extra text before/after the dict is ignored.
"""

import argparse
import ast
import csv
import json
import os
from collections import Counter, OrderedDict
from urllib.parse import unquote


def normalize_header(value):
    return str(value).strip().lower()


def strip_wrapping_quotes(value):
    label = str(value).strip()
    while len(label) >= 2 and label[0] == label[-1] and label[0] in ['"', "'"]:
        label = label[1:-1].strip()
    return label


def normalize_label_key(value):
    label = strip_wrapping_quotes(value)
    label = label.replace('（', '(').replace('）', ')')
    label = ''.join(label.split())
    return label.casefold()


def flatten_label_values(value):
    if isinstance(value, (list, tuple, set)):
        labels = []
        for item in value:
            labels.extend(flatten_label_values(item))
        return labels
    label = strip_wrapping_quotes(value)
    return [label] if label else []


def parse_label_values(label_text):
    raw = str(label_text).strip()
    if not raw:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            return flatten_label_values(loader(raw))
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass

    return flatten_label_values(raw)


def extract_first_dict(text, path):
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end < start:
        raise ValueError(f'{path}: cannot find a Python dict in merge map text')
    return text[start:end + 1]


def load_merge_map(path):
    if not path:
        return OrderedDict(), {}

    with open(path, 'r', encoding='utf-8-sig') as f:
        text = f.read()

    raw_map = ast.literal_eval(extract_first_dict(text, path))
    if not isinstance(raw_map, dict):
        raise ValueError(f'{path}: merge map must be a dict')

    canonical_order = OrderedDict()
    alias_to_canonical = {}
    for canonical, aliases in raw_map.items():
        canonical = strip_wrapping_quotes(canonical)
        canonical_order[canonical] = None
        all_aliases = [canonical]
        if isinstance(aliases, (list, tuple, set)):
            all_aliases.extend(aliases)
        else:
            all_aliases.append(aliases)

        for alias in all_aliases:
            key = normalize_label_key(alias)
            if key in alias_to_canonical and alias_to_canonical[key] != canonical:
                raise ValueError(
                    f'{path}: alias {alias!r} maps to both '
                    f'{alias_to_canonical[key]!r} and {canonical!r}'
                )
            alias_to_canonical[key] = canonical

    return canonical_order, alias_to_canonical


def merge_labels(labels, alias_to_canonical, unknown_policy):
    merged = []
    unknown = []
    seen = set()

    for label in labels:
        key = normalize_label_key(label)
        canonical = alias_to_canonical.get(key)
        if canonical is None:
            if unknown_policy in ['error', 'skip-row', 'skip-label']:
                unknown.append(label)
                continue
            canonical = strip_wrapping_quotes(label)

        if canonical not in seen:
            merged.append(canonical)
            seen.add(canonical)

    if unknown and unknown_policy == 'error':
        raise KeyError('unknown label(s): ' + ', '.join(repr(label) for label in unknown))
    if unknown and unknown_policy == 'skip-row':
        return [], unknown
    return merged, unknown


def column_lookup(fieldnames, required):
    normalized = {normalize_header(name): name for name in fieldnames}
    missing = [name for name in required if normalize_header(name) not in normalized]
    if missing:
        raise ValueError(f'input CSV missing columns {missing}; got {fieldnames}')
    return {name: normalized[normalize_header(name)] for name in required}


def convert_csv(args):
    _, alias_to_canonical = load_merge_map(args.merge_map)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    input_label_counts = Counter()
    output_label_counts = Counter()
    output_label_order = OrderedDict()
    unknown_counts = Counter()

    with open(args.csv, 'r', encoding='utf-8-sig', newline='') as src, \
            open(args.out, 'w', encoding='utf-8', newline='') as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError('input CSV has no header row')
        lookup = column_lookup(
            reader.fieldnames,
            [args.wsi_path_column, args.labels_column, args.coordinates_column],
        )
        writer = csv.DictWriter(dst, fieldnames=['wsi_path', 'labels', 'coordinates'])
        writer.writeheader()

        for row_idx, row in enumerate(reader, start=2):
            total += 1
            wsi_path = str(row[lookup[args.wsi_path_column]]).strip()
            coordinates = str(row[lookup[args.coordinates_column]]).strip()
            labels = parse_label_values(row[lookup[args.labels_column]])

            if args.decode_wsi_path:
                wsi_path = unquote(wsi_path) if '%' in wsi_path else wsi_path

            if not wsi_path or not coordinates or not labels:
                skipped += 1
                continue

            for label in labels:
                input_label_counts[label] += 1

            try:
                merged_labels, unknown = merge_labels(labels, alias_to_canonical, args.unknown_label)
            except KeyError as exc:
                raise KeyError(f'row {row_idx}: {exc}') from exc

            for label in unknown:
                unknown_counts[label] += 1
            if not merged_labels:
                skipped += 1
                continue

            for label in merged_labels:
                output_label_counts[label] += 1
                output_label_order.setdefault(label, None)

            writer.writerow({
                'wsi_path': wsi_path,
                'labels': json.dumps(merged_labels, ensure_ascii=False),
                'coordinates': coordinates,
            })
            written += 1

    if args.label_map_out:
        label_to_id = {label: idx for idx, label in enumerate(output_label_order, start=args.label_id_start)}
        with open(args.label_map_out, 'w', encoding='utf-8') as f:
            json.dump(label_to_id, f, ensure_ascii=False, indent=4)

    print(f'[done] input rows: {total}')
    print(f'[done] output rows: {written}')
    print(f'[done] skipped rows: {skipped}')
    if unknown_counts:
        preview = ', '.join(f'{k}:{v}' for k, v in unknown_counts.most_common(20))
        print(f'[warning] unknown labels handled by --unknown-label={args.unknown_label}: {preview}')
    print('[done] merged label counts:')
    for label, count in output_label_counts.most_common():
        print(f'  {label}: {count}')


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Convert breast annotation CSV to PRET CSV with wsi_path, labels, coordinates.'
    )
    parser.add_argument('--csv', required=True, help='input annotation CSV')
    parser.add_argument('--out', required=True, help='output PRET-style CSV')
    parser.add_argument('--merge-map', default='/Users/chenbozhou/Downloads/breast_cls.txt',
        help='label merge map text file')
    parser.add_argument('--unknown-label', default='skip-row',
        choices=['skip-label', 'error', 'keep', 'skip-row'],
        help='how to handle labels not found in --merge-map; default skip-row drops the whole annotation row')
    parser.add_argument('--label-map-out', default='',
        help='optional JSON label map generated from merged labels in output')
    parser.add_argument('--label-id-start', default=1, type=int,
        help='first id for --label-map-out')
    parser.add_argument('--wsi-path-column', default='wsi_path',
        help='input column name for WSI path')
    parser.add_argument('--labels-column', default='labels',
        help='input column name for labels')
    parser.add_argument('--coordinates-column', default='coordinates',
        help='input column name for normalized coordinates')
    parser.add_argument('--decode-wsi-path', action=argparse.BooleanOptionalAction, default=True,
        help='URL-decode WSI path, e.g. %%23 to #')
    return parser


def main():
    args = build_argparser().parse_args()
    if args.label_id_start < 0:
        raise ValueError('--label-id-start must be non-negative')
    convert_csv(args)


if __name__ == '__main__':
    main()
