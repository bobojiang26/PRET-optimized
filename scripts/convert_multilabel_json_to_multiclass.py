#!/usr/bin/env python
import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a PRET multi-label data_info JSON into single-label multiclass data_info.'
    )
    parser.add_argument('--input_json', required=True, help='source PRET data_info JSON with wsi_labels')
    parser.add_argument('--output_json', required=True, help='output single-label multiclass data_info JSON')
    parser.add_argument('--classes', type=int, nargs='+', required=True,
        help='original class ids to extract, e.g. --classes 5 7')
    parser.add_argument('--out_h5_label_dir', required=True,
        help='directory for converted 1D h5 patch label .npy files')
    parser.add_argument('--keep_original_class_ids', action='store_true',
        help='keep original class ids in wsi_label/patch labels instead of mapping to 1..K')
    parser.add_argument('--strict_no_other_labels', action='store_true',
        help='drop slides that contain labels outside --classes')
    parser.add_argument('--skip_missing_h5_patch_labels', action='store_true',
        help='skip slides whose source h5_patch_labels file is missing instead of failing')
    parser.add_argument('--skip_zero_pos', action='store_true',
        help='skip slides whose selected class has zero positive h5 patches')
    parser.add_argument('--copy_extra_fields', action='store_true',
        help='copy non-label metadata fields from the source JSON')
    return parser.parse_args()


def resolve_existing_path(path, input_json):
    if path is None or path == '':
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    json_relative = os.path.join(os.path.dirname(os.path.abspath(input_json)), path)
    if os.path.exists(json_relative):
        return json_relative
    return path


def slide_labels(info):
    if 'wsi_labels' in info:
        labels = info['wsi_labels']
        if isinstance(labels, (list, tuple, set)):
            return sorted({int(_) for _ in labels})
        return [int(labels)]
    if 'wsi_label' in info:
        label = int(info['wsi_label'])
        return [label] if label > 0 else []
    return []


def source_class_mask(source_labels, original_class_id):
    if source_labels.ndim == 2:
        col = int(original_class_id) - 1
        if col < 0 or col >= source_labels.shape[1]:
            return np.zeros(source_labels.shape[0], dtype=bool)
        return source_labels[:, col] > 0
    if source_labels.ndim == 1:
        return source_labels == int(original_class_id)
    raise ValueError(f'h5_patch_labels must be 1D or 2D, got shape {source_labels.shape}')


def output_label_for_class(original_class_id, class_to_output, keep_original):
    if keep_original:
        return int(original_class_id)
    return int(class_to_output[original_class_id])


def convert_patch_labels(source_path, output_path, selected_class, output_label):
    source_labels = np.load(source_path)
    out = np.zeros(source_labels.shape[0], dtype=np.uint16)
    mask = source_class_mask(source_labels, selected_class)
    out[mask] = int(output_label)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, out)
    return int(mask.sum()), source_labels.shape, out.dtype.name


def build_output_item(source_info, output_label, output_h5_patch_labels, pos_patch_num, copy_extra_fields):
    if copy_extra_fields:
        item = {
            key: value
            for key, value in source_info.items()
            if key not in {
                'wsi_labels', 'wsi_label', 'h5_patch_labels',
                'patch_labels', 'pos_patch_num', 'pseudo_label',
            }
        }
    else:
        item = {
            'fixed_test_set': bool(source_info.get('fixed_test_set', False)),
        }
    item['wsi_label'] = int(output_label)
    item['h5_patch_labels'] = output_h5_patch_labels
    item['pos_patch_num'] = int(pos_patch_num)
    if 'h5_input' in source_info:
        item['h5_input'] = bool(source_info['h5_input'])
    return item


def main():
    args = parse_args()
    selected_classes = [int(_) for _ in args.classes]
    if len(selected_classes) != len(set(selected_classes)):
        raise ValueError('--classes must not contain duplicates')
    class_to_output = {
        original_class_id: idx + 1
        for idx, original_class_id in enumerate(selected_classes)
    }

    with open(args.input_json, 'r', encoding='utf8') as f:
        source = json.load(f)

    os.makedirs(args.out_h5_label_dir, exist_ok=True)
    output = {}
    summary = {
        'input_slides': len(source),
        'output_slides': 0,
        'class_mapping': {
            str(original_class_id): output_label_for_class(
                original_class_id, class_to_output, args.keep_original_class_ids
            )
            for original_class_id in selected_classes
        },
        'skipped_no_selected_class': 0,
        'skipped_multiple_selected_classes': 0,
        'skipped_other_labels': 0,
        'skipped_missing_h5_patch_labels': 0,
        'skipped_zero_pos': 0,
        'class_counts': {},
        'pos_patch_counts': {},
    }

    for slide_name, info in sorted(source.items()):
        labels = slide_labels(info)
        selected_present = [label for label in labels if label in selected_classes]
        other_labels = [label for label in labels if label not in selected_classes]

        if len(selected_present) == 0:
            summary['skipped_no_selected_class'] += 1
            continue
        if len(selected_present) > 1:
            summary['skipped_multiple_selected_classes'] += 1
            continue
        if args.strict_no_other_labels and other_labels:
            summary['skipped_other_labels'] += 1
            continue
        if 'h5_patch_labels' not in info:
            if args.skip_missing_h5_patch_labels:
                summary['skipped_missing_h5_patch_labels'] += 1
                continue
            raise ValueError(f'{slide_name}: missing h5_patch_labels in source JSON')

        selected_class = int(selected_present[0])
        output_label = output_label_for_class(
            selected_class, class_to_output, args.keep_original_class_ids
        )
        source_label_path = resolve_existing_path(info['h5_patch_labels'], args.input_json)
        if not os.path.exists(source_label_path):
            if args.skip_missing_h5_patch_labels:
                summary['skipped_missing_h5_patch_labels'] += 1
                continue
            raise FileNotFoundError(f'{slide_name}: h5_patch_labels not found: {source_label_path}')

        output_label_path = os.path.join(args.out_h5_label_dir, slide_name + '.npy')
        pos_patch_num, source_shape, dtype_name = convert_patch_labels(
            source_label_path, output_label_path, selected_class, output_label
        )
        if pos_patch_num == 0 and args.skip_zero_pos:
            summary['skipped_zero_pos'] += 1
            if os.path.exists(output_label_path):
                os.remove(output_label_path)
            continue

        output[slide_name] = build_output_item(
            info, output_label, output_label_path, pos_patch_num, args.copy_extra_fields
        )
        output_key = str(output_label)
        summary['class_counts'][output_key] = summary['class_counts'].get(output_key, 0) + 1
        summary['pos_patch_counts'][output_key] = (
            summary['pos_patch_counts'].get(output_key, 0) + int(pos_patch_num)
        )
        summary['output_slides'] += 1
        if source_shape[0] == 0:
            print(f'[warning] {slide_name}: source h5 patch labels are empty')
        if pos_patch_num == 0:
            print(
                f'[warning] {slide_name}: selected source class {selected_class} '
                f'mapped to {output_label} has zero positive patches'
            )
        print(
            f'[convert] {slide_name}: source_class={selected_class} -> wsi_label={output_label}, '
            f'pos_patch_num={pos_patch_num}, source_shape={tuple(source_shape)}, dtype={dtype_name}'
        )

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    summary_path = os.path.splitext(args.output_json)[0] + '_summary.json'
    with open(summary_path, 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print('[summary] input slides:', summary['input_slides'])
    print('[summary] output slides:', summary['output_slides'])
    print('[summary] class counts:', summary['class_counts'])
    print('[summary] pos patch counts:', summary['pos_patch_counts'])
    print('[summary] wrote:', args.output_json)
    print('[summary] wrote:', summary_path)


if __name__ == '__main__':
    main()
