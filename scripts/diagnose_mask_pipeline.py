#!/usr/bin/env python
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


H5_EXTENSIONS = ('.h5', '.hdf5')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose PRET mask data flow from data_info JSON and h5-aligned patch-label NPY files.'
    )
    parser.add_argument('--dataset_info', required=True)
    parser.add_argument('--h5_dir', default='')
    parser.add_argument('--class_num', type=int, default=0)
    parser.add_argument('--multilabel', action='store_true')
    parser.add_argument('--multilabel_mask_negative_source', default='other_positive',
        choices=['all_zero', 'other_positive', 'none'])
    parser.add_argument('--out_json', default='')
    parser.add_argument('--max_warnings', type=int, default=50)
    return parser.parse_args()


def resolve_path(path, dataset_info_path):
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(os.path.dirname(os.path.abspath(dataset_info_path)), path)
    if os.path.exists(candidate):
        return candidate
    return path


def slide_keys(value):
    raw = str(value)
    stem = os.path.splitext(os.path.basename(raw))[0]
    return {raw, os.path.basename(raw), stem}


def find_h5_files(path):
    if not path:
        return {}
    root = Path(path)
    if not root.exists():
        return {}
    if root.is_file():
        candidates = [root] if root.suffix.lower() in H5_EXTENSIONS else []
    else:
        candidates = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in H5_EXTENSIONS)
    out = {}
    for candidate in candidates:
        for key in slide_keys(candidate.name):
            out[key] = str(candidate)
    return out


def read_h5_lengths(path):
    try:
        import h5py
    except ImportError:
        return None
    with h5py.File(path, 'r') as f:
        features_len = int(f['features'].shape[0]) if 'features' in f else None
        coord_key = next((key for key in ['coords', 'coordinates'] if key in f), None)
        coords_len = int(f[coord_key].shape[0]) if coord_key is not None else None
    return {'features': features_len, 'coords': coords_len}


def get_wsi_label_ids(info):
    if 'wsi_labels' in info:
        labels = info['wsi_labels']
        if isinstance(labels, (list, tuple, set)):
            return sorted({int(_) for _ in labels})
        return [int(labels)]
    label = int(info.get('wsi_label', 0))
    return [label] if label > 0 else []


def is_multilabel_dataset(dataset_info, args):
    return args.multilabel or any('wsi_labels' in info for info in dataset_info.values())


def positive_patch_count(labels):
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return int((labels > 0).any(axis=1).sum())
    return int((labels > 0).sum())


def positive_classes(labels, class_num):
    labels = np.asarray(labels)
    if labels.ndim == 2:
        upto = min(class_num, labels.shape[1])
        return [idx + 1 for idx in range(upto) if bool((labels[:, idx] > 0).any())]
    values = np.unique(labels)
    return sorted(int(v) for v in values if 0 < int(v) < 254)


def infer_class_num(dataset_info, args):
    if args.class_num > 0:
        return args.class_num
    max_label = 0
    for info in dataset_info.values():
        labels = get_wsi_label_ids(info)
        if labels:
            max_label = max(max_label, max(labels))
    for info in dataset_info.values():
        path = info.get('h5_patch_labels')
        if not path:
            continue
        path = resolve_path(path, args.dataset_info)
        if not os.path.exists(path):
            continue
        labels = np.load(path, mmap_mode='r')
        if labels.ndim == 2:
            max_label = max(max_label, labels.shape[1])
        else:
            pos = positive_classes(labels, max_label or 1)
            if pos:
                max_label = max(max_label, max(pos))
    return max_label


def one_vs_rest_counts(raw_labels, wsi_labels, cls, class_num, multilabel, negative_source):
    raw_labels = np.asarray(raw_labels)
    n = int(raw_labels.shape[0])
    wsi_label = wsi_labels[0] if len(wsi_labels) == 1 else 0

    if multilabel:
        if raw_labels.ndim == 2:
            col = int(cls) - 1
            target = raw_labels[:, col] > 0 if 0 <= col < raw_labels.shape[1] else np.zeros(n, dtype=bool)
            if negative_source == 'all_zero':
                neg = ~target
            elif negative_source == 'other_positive' and raw_labels.shape[1] > 1:
                any_pos = (raw_labels > 0).any(axis=1)
                neg = any_pos & ~target
            else:
                neg = np.zeros(n, dtype=bool)
            pos = target
            ignored = ~(pos | neg)
            return {1: int(pos.sum()), 0: int(neg.sum()), 255: int(ignored.sum())}

        pos = raw_labels == int(cls)
        if negative_source == 'all_zero':
            neg = ~pos
        elif negative_source == 'other_positive':
            neg = (raw_labels > 0) & (raw_labels != int(cls)) & (raw_labels < 254)
        else:
            neg = np.zeros(n, dtype=bool)
        ignored = ~(pos | neg)
        return {1: int(pos.sum()), 0: int(neg.sum()), 255: int(ignored.sum())}

    if class_num > 1 and raw_labels.ndim == 1 and set(np.unique(raw_labels).astype(int).tolist()).issubset({0, 1}):
        pos = raw_labels == 1 if int(wsi_label) == int(cls) else np.zeros(n, dtype=bool)
        neg = raw_labels == 1 if int(wsi_label) != int(cls) else np.zeros(n, dtype=bool)
        ignored = raw_labels == 0
        return {1: int(pos.sum()), 0: int(neg.sum()), 255: int(ignored.sum())}

    if class_num > 1:
        pos = raw_labels == int(cls)
        neg = (raw_labels > 0) & (raw_labels != int(cls))
        ignored = ~(pos | neg)
        return {1: int(pos.sum()), 0: int(neg.sum()), 255: int(ignored.sum())}

    values, counts = np.unique(raw_labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def add_counts(dst, src):
    for key, value in src.items():
        dst[int(key)] += int(value)


def warn(warnings, message):
    warnings.append(message)


def main():
    args = parse_args()
    with open(args.dataset_info, 'r', encoding='utf8') as f:
        dataset_info = json.load(f)

    class_num = infer_class_num(dataset_info, args)
    multilabel = is_multilabel_dataset(dataset_info, args)
    h5_files = find_h5_files(args.h5_dir)

    warnings = []
    class_summary = {
        cls: {
            'wsi_positive_slides': 0,
            'patch_positive_slides': 0,
            'patch_positive_tokens': 0,
            'one_vs_rest_counts': defaultdict(int),
        }
        for cls in range(1, class_num + 1)
    }
    slides_checked = 0
    labels_checked = 0
    h5_length_checked = 0

    for slide_name, info in sorted(dataset_info.items()):
        slides_checked += 1
        wsi_labels = get_wsi_label_ids(info)
        for cls in wsi_labels:
            if 1 <= cls <= class_num:
                class_summary[cls]['wsi_positive_slides'] += 1

        label_path = info.get('h5_patch_labels')
        if not label_path:
            warn(warnings, f'{slide_name}: missing h5_patch_labels')
            continue
        label_path = resolve_path(label_path, args.dataset_info)
        if not os.path.exists(label_path):
            warn(warnings, f'{slide_name}: h5_patch_labels file not found: {label_path}')
            continue

        raw_labels = np.load(label_path, mmap_mode='r')
        labels_checked += 1
        if raw_labels.ndim not in [1, 2]:
            warn(warnings, f'{slide_name}: h5_patch_labels must be 1D or 2D, got {raw_labels.shape}')
            continue

        expected_pos = positive_patch_count(raw_labels)
        recorded_pos = info.get('pos_patch_num')
        if recorded_pos is not None and int(recorded_pos) != int(expected_pos):
            warn(warnings, f'{slide_name}: pos_patch_num={recorded_pos}, computed={expected_pos}')

        pos_classes = positive_classes(raw_labels, class_num)
        for cls in pos_classes:
            if 1 <= cls <= class_num:
                class_summary[cls]['patch_positive_slides'] += 1
                if raw_labels.ndim == 2:
                    class_summary[cls]['patch_positive_tokens'] += int((raw_labels[:, cls - 1] > 0).sum())
                else:
                    class_summary[cls]['patch_positive_tokens'] += int((raw_labels == cls).sum())

        for cls in wsi_labels:
            if 1 <= cls <= class_num and cls not in pos_classes:
                warn(warnings, f'{slide_name}: wsi label {cls} has zero positive h5 patch labels')
        for cls in pos_classes:
            if cls not in wsi_labels:
                warn(warnings, f'{slide_name}: h5 patch labels contain class {cls}, but wsi labels are {wsi_labels}')

        if h5_files:
            h5_path = None
            for key in slide_keys(slide_name):
                if key in h5_files:
                    h5_path = h5_files[key]
                    break
            if h5_path is None:
                warn(warnings, f'{slide_name}: no matching h5 file under {args.h5_dir}')
            else:
                lengths = read_h5_lengths(h5_path)
                if lengths is not None:
                    h5_length_checked += 1
                    for key, length in lengths.items():
                        if length is not None and int(raw_labels.shape[0]) != int(length):
                            warn(warnings, f'{slide_name}: label length {raw_labels.shape[0]} != h5 {key} length {length}')

        for cls in range(1, class_num + 1):
            counts = one_vs_rest_counts(
                raw_labels, wsi_labels, cls, class_num, multilabel,
                args.multilabel_mask_negative_source
            )
            add_counts(class_summary[cls]['one_vs_rest_counts'], counts)

    for cls, stats in class_summary.items():
        counts = stats['one_vs_rest_counts']
        if counts.get(1, 0) == 0:
            warn(warnings, f'class {cls}: one-vs-rest has zero positive reference tokens')
        if counts.get(0, 0) == 0:
            warn(warnings, f'class {cls}: one-vs-rest has zero negative reference tokens')

    serializable_classes = {}
    for cls, stats in class_summary.items():
        serializable_classes[str(cls)] = {
            'wsi_positive_slides': int(stats['wsi_positive_slides']),
            'patch_positive_slides': int(stats['patch_positive_slides']),
            'patch_positive_tokens': int(stats['patch_positive_tokens']),
            'one_vs_rest_counts': {str(k): int(v) for k, v in sorted(stats['one_vs_rest_counts'].items())},
        }

    summary = {
        'dataset_info': args.dataset_info,
        'slides_checked': slides_checked,
        'h5_patch_label_files_checked': labels_checked,
        'h5_length_files_checked': h5_length_checked,
        'class_num': class_num,
        'multilabel': multilabel,
        'multilabel_mask_negative_source': args.multilabel_mask_negative_source,
        'classes': serializable_classes,
        'warning_count': len(warnings),
        'warnings_preview': warnings[:args.max_warnings],
    }

    print('[diagnose] slides:', slides_checked)
    print('[diagnose] h5_patch_label_files:', labels_checked)
    print('[diagnose] class_num:', class_num)
    print('[diagnose] multilabel:', multilabel)
    print('[diagnose] warnings:', len(warnings))
    print('[diagnose] per-class one-vs-rest counts:')
    for cls in range(1, class_num + 1):
        stats = serializable_classes[str(cls)]
        print(
            '  class ' + str(cls) +
            ': wsi_pos=' + str(stats['wsi_positive_slides']) +
            ', patch_pos_slides=' + str(stats['patch_positive_slides']) +
            ', patch_pos_tokens=' + str(stats['patch_positive_tokens']) +
            ', ovr=' + json.dumps(stats['one_vs_rest_counts'], sort_keys=True)
        )
    for message in warnings[:args.max_warnings]:
        print('[warning]', message)
    if len(warnings) > args.max_warnings:
        print('[warning] ... ' + str(len(warnings) - args.max_warnings) + ' more omitted')

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, 'w', encoding='utf8') as f:
            json.dump(summary, f, indent=2)
        print('[diagnose] wrote:', args.out_json)


if __name__ == '__main__':
    main()
