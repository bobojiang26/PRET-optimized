#!/usr/bin/env python
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np


H5_EXTENSIONS = ('.h5', '.hdf5')
COORD_KEYS = ('coords', 'coordinates')
LABEL_COLORS = {
    1: '#dc2626',     # target positive
    0: '#2563eb',     # other foreground / hard negative
    255: '#d4d4d4',   # ignored / unannotated / background
    254: '#f59e0b',   # uncertain
    -1: '#7c3aed',    # unknown
}
LABEL_NAMES = {
    1: 'target positive',
    0: 'other/hard negative',
    255: 'ignored/unannotated',
    254: 'uncertain',
    -1: 'unknown',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize mask labels before and after PRET mask-processing/tagger refinement.'
    )
    parser.add_argument('--h5_dir', '--h5-dir', required=True, help='directory containing h5 feature files')
    parser.add_argument('--dataset_info', '--dataset-info', required=True, help='PRET data_info JSON')
    parser.add_argument('--out_dir', '--out-dir', default='records/mask_processing_vis')
    parser.add_argument('--slides', nargs='*', default=None, help='slide names to visualize; default samples from dataset_info')
    parser.add_argument('--classes', nargs='*', type=int, default=None, help='classes to visualize; default all 1..class_num')
    parser.add_argument('--class_num', '--class-num', type=int, default=0, help='number of classes; inferred when possible')
    parser.add_argument('--max_slides', '--max-slides', type=int, default=8, help='maximum slides when --slides is omitted')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--multilabel', action='store_true', help='interpret h5 patch labels as multilabel/multi-hot')
    parser.add_argument('--multilabel_mask_negative_source', '--multilabel-mask-negative-source',
        default='other_positive', choices=['all_zero', 'other_positive', 'none'])
    parser.add_argument('--h5_coordinate_mode', '--h5-coordinate-mode',
        default='pixel', choices=['grid', 'pixel'])
    parser.add_argument('--h5_patch_size', '--h5-patch-size', type=int, default=512)
    parser.add_argument('--run_mask_tagger', '--run-mask-tagger', action='store_true',
        help='run execute_mask_subtyping_tagger on selected slides before plotting the after_tagger panel')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--ignore', type=float, default=0.0, help='uncertainty width passed to mask tagger')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--svg_size', '--svg-size', type=int, default=720)
    parser.add_argument('--point_radius', '--point-radius', type=float, default=2.2)
    parser.add_argument('--point_opacity', '--point-opacity', type=float, default=0.82)
    return parser.parse_args()


def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_.#-]+', '_', str(name))


def svg_escape(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def resolve_path(path, base_dir):
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(base_dir, path)
    return candidate if os.path.exists(candidate) else path


def slide_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def decode_slide_key(name):
    name = str(name).strip()
    return unquote(name) if '%' in name else name


def find_h5_files(h5_dir):
    root = Path(h5_dir)
    if not root.exists():
        raise FileNotFoundError(h5_dir)
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in H5_EXTENSIONS]
    out = {}
    for path in sorted(files):
        stem = slide_stem(str(path))
        out[stem] = str(path)
        out[decode_slide_key(stem)] = str(path)
    return out


def read_h5(path, need_features=False):
    import h5py

    with h5py.File(path, 'r') as f:
        coord_key = next((key for key in COORD_KEYS if key in f), None)
        if coord_key is None:
            raise KeyError(f'{path}: missing coords/coordinates key')
        coords = np.asarray(f[coord_key])[:, :2].astype(np.float64, copy=False)
        feats = np.asarray(f['features']).astype(np.float32, copy=False) if need_features else None
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f'{path}: coordinates must have shape (N, >=2), got {coords.shape}')
    if feats is not None:
        norms = np.linalg.norm(feats, ord=2, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-8)
        if feats.shape[0] != coords.shape[0]:
            raise ValueError(f'{path}: features/coords length mismatch: {feats.shape[0]} vs {coords.shape[0]}')
    return coords, feats


def load_patch_labels(slide_info, dataset_info_dir, coords):
    label_key = 'h5_patch_labels' if 'h5_patch_labels' in slide_info else 'patch_labels'
    if label_key not in slide_info:
        raise KeyError('slide has no h5_patch_labels/patch_labels entry')
    label_path = resolve_path(slide_info[label_key], dataset_info_dir)
    labels = np.load(label_path)
    if labels.shape[0] != coords.shape[0]:
        raise ValueError(f'{label_path}: expected {coords.shape[0]} patch labels, got {labels.shape[0]}')
    return labels


def get_wsi_label_ids(info):
    if 'wsi_labels' in info:
        labels = info.get('wsi_labels') or []
        if isinstance(labels, (list, tuple)):
            return [int(x) for x in labels]
        return [int(labels)]
    if 'wsi_label' in info:
        label = int(info.get('wsi_label'))
        return [] if label <= 0 else [label]
    return []


def has_wsi_label(info, cls):
    return int(cls) in set(get_wsi_label_ids(info))


def infer_class_num(dataset_info, dataset_info_dir, explicit_class_num=0):
    if explicit_class_num > 0:
        return explicit_class_num
    max_label = 0
    for info in dataset_info.values():
        labels = get_wsi_label_ids(info)
        if labels:
            max_label = max(max_label, max(labels))
    if max_label > 0:
        return max_label

    for info in dataset_info.values():
        label_path = info.get('h5_patch_labels') or info.get('patch_labels')
        if label_path:
            try:
                label_path = resolve_path(label_path, dataset_info_dir)
                labels = np.load(label_path, mmap_mode='r')
                if labels.ndim == 2:
                    return int(labels.shape[1])
                if labels.size:
                    max_label = max(max_label, int(np.max(labels)))
            except Exception:
                continue
    if max_label > 0:
        return max_label
    raise ValueError('cannot infer class_num; pass --class_num')


def choose_slides(dataset_info, requested, max_slides, seed):
    names = list(dataset_info.keys())
    if requested:
        missing = [name for name in requested if name not in dataset_info]
        if missing:
            raise KeyError(f'slides not found in dataset_info: {missing}')
        return requested
    rng = np.random.default_rng(seed)
    labeled = [name for name in names if get_wsi_label_ids(dataset_info[name])]
    pool = labeled if labeled else names
    if max_slides > 0 and len(pool) > max_slides:
        idx = rng.choice(len(pool), size=max_slides, replace=False)
        return [pool[i] for i in sorted(idx)]
    return pool


def patch_labels_for_class(patch_labels, cls, class_num, multilabel=False, multilabel_negative_source='all_zero'):
    patch_labels = np.asarray(patch_labels)
    if multilabel:
        if patch_labels.ndim == 2:
            col = int(cls) - 1
            source = (multilabel_negative_source or 'all_zero').lower()
            if source == 'all_zero':
                out = np.zeros(patch_labels.shape[0], dtype=np.int64)
            else:
                out = np.full(patch_labels.shape[0], 255, dtype=np.int64)
            if 0 <= col < patch_labels.shape[1]:
                target = patch_labels[:, col] > 0
                if source == 'other_positive' and patch_labels.shape[1] > 1:
                    other = np.delete(patch_labels, col, axis=1).max(1) > 0
                    out[other] = 0
                out[target] = 1
            return out
        source = (multilabel_negative_source or 'all_zero').lower()
        if source == 'all_zero':
            return (patch_labels == int(cls)).astype(np.int64)
        out = np.full(patch_labels.shape[0], 255, dtype=np.int64)
        out[patch_labels == int(cls)] = 1
        if source == 'other_positive':
            other = (patch_labels > 0) & (patch_labels != int(cls)) & (patch_labels < 254)
            out[other] = 0
        return out

    if class_num > 1:
        out = np.full(patch_labels.shape[0], 255, dtype=np.int64)
        out[patch_labels == int(cls)] = 1
        out[(patch_labels > 0) & (patch_labels != int(cls))] = 0
        if np.any(patch_labels == 1) and not np.any(patch_labels == int(cls)):
            out[patch_labels == 1] = 0
        return out
    return patch_labels.astype(np.int64, copy=True)


def raw_panel_labels(patch_labels, cls, multilabel):
    patch_labels = np.asarray(patch_labels)
    if multilabel and patch_labels.ndim == 2:
        out = np.full(patch_labels.shape[0], 255, dtype=np.int64)
        col = int(cls) - 1
        if 0 <= col < patch_labels.shape[1]:
            target = patch_labels[:, col] > 0
            other = np.delete(patch_labels, col, axis=1).max(1) > 0 if patch_labels.shape[1] > 1 else np.zeros_like(target)
            out[other] = 0
            out[target] = 1
        return out
    return patch_labels_for_class(patch_labels, cls, max(int(cls), 1), multilabel=False)


def label_counts(labels):
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def final_reference_panel(labels):
    labels = np.asarray(labels, dtype=np.int64)
    out = np.full(labels.shape[0], 255, dtype=np.int64)
    keep = (labels == 0) | (labels == 1)
    out[keep] = labels[keep]
    return out


def format_counts(counts):
    return ', '.join(f'{k}:{counts[k]}' for k in sorted(counts))


def coords_to_plot_xy(coords, coordinate_mode, h5_patch_size):
    coords = np.asarray(coords, dtype=np.float64)
    if coordinate_mode == 'pixel':
        if h5_patch_size <= 0:
            raise ValueError('--h5_patch_size must be > 0 for pixel coordinates')
        x = np.floor(coords[:, 0] / float(h5_patch_size))
        y = np.floor(coords[:, 1] / float(h5_patch_size))
    else:
        x = coords[:, 0]
        y = coords[:, 1]
    return x, y


def project_points(x, y, panel_x, panel_y, plot_w, plot_h):
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    sx = panel_x + (x - x_min) / x_span * plot_w
    sy = panel_y + (y - y_min) / y_span * plot_h
    return sx, sy, (x_min, x_max, y_min, y_max)


def write_mask_svg(out_path, slide, cls, coords, panels, args):
    x, y = coords_to_plot_xy(coords, args.h5_coordinate_mode, args.h5_patch_size)
    panel_count = len(panels)
    panel_size = int(args.svg_size)
    margin = 50
    plot_w = panel_size - margin * 2
    plot_h = panel_size - margin * 2
    legend_h = 82
    width = panel_size * panel_count
    height = panel_size + legend_h
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf7"/>',
        f'<text x="18" y="28" font-family="monospace" font-size="16" fill="#222">{svg_escape(slide)}  class={cls}</text>',
    ]

    for panel_idx, (title, labels) in enumerate(panels):
        x0 = panel_idx * panel_size + margin
        y0 = margin
        counts = label_counts(labels)
        sx, sy, bounds = project_points(x, y, x0, y0, plot_w, plot_h)
        lines.extend([
            f'<text x="{x0}" y="{y0 - 12}" font-family="monospace" font-size="13" fill="#222">{svg_escape(title)} | {svg_escape(format_counts(counts))}</text>',
            f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#222" stroke-width="1"/>',
        ])
        for draw_label in [255, 254, -1, 0, 1]:
            mask = np.asarray(labels) == draw_label
            if not np.any(mask):
                continue
            color = LABEL_COLORS.get(draw_label, '#111827')
            opacity = 0.36 if draw_label == 255 else float(args.point_opacity)
            radius = max(0.6, float(args.point_radius) * (0.55 if draw_label == 255 else 1.0))
            for px, py in zip(sx[mask], sy[mask]):
                lines.append(
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" '
                    f'fill="{color}" fill-opacity="{opacity:.2f}"/>'
                )
        x_min, x_max, y_min, y_max = bounds
        lines.append(
            f'<text x="{x0}" y="{y0 + plot_h + 18}" font-family="monospace" font-size="11" fill="#555">'
            f'x={x_min:.0f}..{x_max:.0f} y={y_min:.0f}..{y_max:.0f}</text>'
        )

    legend_x = 18
    legend_y = height - 56
    for idx, label in enumerate([1, 0, 255, 254, -1]):
        x0 = legend_x + idx * 210
        lines.append(f'<circle cx="{x0}" cy="{legend_y}" r="6" fill="{LABEL_COLORS[label]}"/>')
        lines.append(
            f'<text x="{x0 + 12}" y="{legend_y + 4}" font-family="monospace" font-size="12" fill="#333">'
            f'{label}: {svg_escape(LABEL_NAMES[label])}</text>'
        )

    lines.append('</svg>')
    with open(out_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines) + '\n')


def build_patch_names(slide, coords, args):
    x, y = coords_to_plot_xy(coords, args.h5_coordinate_mode, args.h5_patch_size)
    return [os.path.join('h5_features', slide, f'{int(px)}_{int(py)}.jpeg') for px, py in zip(x, y)]


def choose_device(requested):
    import torch

    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda requested, but CUDA is not available')
        return torch.device('cuda')
    if requested == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_mask_tagger_for_class(slides, slide_data, dataset_info, cls, args):
    import torch

    core_dir = Path(__file__).resolve().parents[1] / 'core'
    sys.path.insert(0, str(core_dir))
    from modules import execute_mask_subtyping_tagger

    device = choose_device(args.device)
    feats, labels, patch_names, wsi_names, wsi_binary_labels = [], [], [], [], []
    offsets = {}
    start = 0
    for slide in slides:
        item = slide_data[slide]
        if item['features'] is None:
            raise ValueError('--run_mask_tagger requires h5 features')
        cur_labels = item['initial_by_class'][cls]
        offsets[slide] = (start, start + cur_labels.shape[0])
        start += cur_labels.shape[0]
        feats.append(item['features'])
        labels.append(cur_labels)
        patch_names.extend(build_patch_names(slide, item['coords'], args))
        wsi_names.append(slide)
        wsi_binary_labels.append((slide, 1 if has_wsi_label(dataset_info[slide], cls) else 0))

    feats_t = torch.from_numpy(np.concatenate(feats, 0).astype(np.float32, copy=False)).to(device)
    labels_t = torch.from_numpy(np.concatenate(labels, 0).astype(np.int64, copy=False)).to(device)
    labels_t = execute_mask_subtyping_tagger(
        feats_t, labels_t, patch_names, wsi_names, wsi_binary_labels,
        vis_info=None, uncertain=args.ignore, topk=args.topk
    )
    labels_np = labels_t.detach().cpu().numpy()
    return {slide: labels_np[start:end] for slide, (start, end) in offsets.items()}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dataset_info = load_json(args.dataset_info)
    dataset_info_dir = os.path.dirname(os.path.abspath(args.dataset_info))
    h5_map = find_h5_files(args.h5_dir)
    class_num = infer_class_num(dataset_info, dataset_info_dir, args.class_num)
    classes = args.classes or list(range(1, class_num + 1))
    slides = choose_slides(dataset_info, args.slides, args.max_slides, args.seed)

    slide_data = {}
    summary = {'class_num': class_num, 'slides': {}, 'outputs': []}
    for slide in slides:
        h5_path = h5_map.get(slide)
        if h5_path is None:
            print(f'[warning] skip {slide}: no matching h5 file under {args.h5_dir}', flush=True)
            continue
        coords, features = read_h5(h5_path, need_features=args.run_mask_tagger)
        patch_labels = load_patch_labels(dataset_info[slide], dataset_info_dir, coords)
        raw_by_class = {}
        initial_by_class = {}
        for cls in classes:
            raw_by_class[cls] = raw_panel_labels(patch_labels, cls, args.multilabel)
            initial_by_class[cls] = patch_labels_for_class(
                patch_labels, cls, class_num,
                multilabel=args.multilabel,
                multilabel_negative_source=args.multilabel_mask_negative_source
            )
        slide_data[slide] = {
            'h5_path': h5_path,
            'coords': coords,
            'features': features,
            'patch_labels': patch_labels,
            'raw_by_class': raw_by_class,
            'initial_by_class': initial_by_class,
        }
        summary['slides'][slide] = {
            'h5_path': h5_path,
            'point_count': int(coords.shape[0]),
            'wsi_labels': get_wsi_label_ids(dataset_info[slide]),
        }

    if not slide_data:
        raise SystemExit('No slides were visualized.')

    after_tagger = {cls: {} for cls in classes}
    if args.run_mask_tagger:
        for cls in classes:
            print(f'[mask-vis] running mask tagger for class {cls} on {len(slide_data)} slide(s)', flush=True)
            after_tagger[cls] = run_mask_tagger_for_class(
                list(slide_data), slide_data, dataset_info, cls, args
            )

    for slide, item in slide_data.items():
        for cls in classes:
            panels = [
                ('raw target-vs-other', item['raw_by_class'][cls]),
                ('initial one-vs-rest', item['initial_by_class'][cls]),
            ]
            if args.run_mask_tagger:
                panels.append(('after mask tagger', after_tagger[cls][slide]))
                final_labels = final_reference_panel(after_tagger[cls][slide])
            else:
                final_labels = final_reference_panel(item['initial_by_class'][cls])
            panels.append(('final reference kept', final_labels))
            out_name = f'{safe_name(slide)}_class_{cls}_mask_processing.svg'
            out_path = os.path.join(args.out_dir, out_name)
            write_mask_svg(out_path, slide, cls, item['coords'], panels, args)
            row = {
                'slide': slide,
                'class': int(cls),
                'output': out_path,
                'raw_counts': label_counts(item['raw_by_class'][cls]),
                'initial_counts': label_counts(item['initial_by_class'][cls]),
                'final_reference_counts': label_counts(final_labels),
            }
            if args.run_mask_tagger:
                row['after_tagger_counts'] = label_counts(after_tagger[cls][slide])
            summary['outputs'].append(row)
            print(
                f"[mask-vis] {slide} class={cls} raw={format_counts(row['raw_counts'])} "
                f"initial={format_counts(row['initial_counts'])}" +
                (f" after={format_counts(row['after_tagger_counts'])}" if args.run_mask_tagger else '') +
                f" final={format_counts(row['final_reference_counts'])}" +
                f" -> {out_path}",
                flush=True,
            )

    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2)
    print(f'[mask-vis] wrote: {summary_path}')


if __name__ == '__main__':
    main()
