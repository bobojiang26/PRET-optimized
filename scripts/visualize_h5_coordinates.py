#!/usr/bin/env python
import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np


H5_EXTENSIONS = ('.h5', '.hdf5')
COORD_KEYS = ('coords', 'coordinates')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize h5 coordinate distributions and summarize whether they look like grid or pixel coordinates.'
    )
    parser.add_argument('--h5_dir', '--h5-dir', default='', help='directory containing h5 files')
    parser.add_argument('--h5_files', '--h5-files', nargs='*', default=None, help='explicit h5 files; overrides --h5_dir')
    parser.add_argument('--out_dir', '--out-dir', default='records/h5_coordinate_vis')
    parser.add_argument('--recursive', dest='recursive', action='store_true', default=True)
    parser.add_argument('--no_recursive', '--no-recursive', dest='recursive', action='store_false')
    parser.add_argument('--max_files', '--max-files', type=int, default=24, help='maximum files to plot; 0 means all')
    parser.add_argument('--max_points', '--max-points', type=int, default=8000, help='maximum coordinates plotted per slide')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--pixel_step_threshold', '--pixel-step-threshold', type=int, default=16,
        help='diagnostic threshold only: min step >= threshold is reported as pixel-like')
    parser.add_argument('--svg_size', '--svg-size', type=int, default=900)
    return parser.parse_args()


def find_h5_files(args):
    if args.h5_files:
        return sorted(args.h5_files)
    if not args.h5_dir:
        raise ValueError('provide --h5_dir or --h5_files')
    root = Path(args.h5_dir)
    if root.is_file():
        return [str(root)] if root.suffix.lower() in H5_EXTENSIONS else []
    if not root.exists():
        raise FileNotFoundError(args.h5_dir)
    iterator = root.rglob('*') if args.recursive else root.iterdir()
    return sorted(str(p) for p in iterator if p.is_file() and p.suffix.lower() in H5_EXTENSIONS)


def safe_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[^A-Za-z0-9_.#-]+', '_', stem)


def axis_step(values):
    values = sorted({int(round(float(v))) for v in values})
    diffs = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not diffs:
        return None
    step = diffs[0]
    for diff in diffs[1:]:
        step = math.gcd(step, diff)
    return int(step) if step > 0 else None


def read_h5_info(path):
    import h5py

    with h5py.File(path, 'r') as f:
        coord_key = next((key for key in COORD_KEYS if key in f), None)
        if coord_key is None:
            raise KeyError(f'{path}: missing coords/coordinates key')
        coords = np.asarray(f[coord_key])[:, :2]
        feature_count = int(f['features'].shape[0]) if 'features' in f else None
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f'{path}: coordinates must have shape (N, >=2), got {coords.shape}')
    return coords.astype(np.float64, copy=False), coord_key, feature_count


def summarize_coords(path, coords, coord_key, feature_count, pixel_step_threshold):
    x_step = axis_step(coords[:, 0])
    y_step = axis_step(coords[:, 1])
    steps = [s for s in [x_step, y_step] if s is not None]
    min_step = min(steps) if steps else None
    recommended = 'pixel' if min_step is not None and min_step >= pixel_step_threshold else 'grid'
    return {
        'path': path,
        'slide': safe_name(path),
        'coord_key': coord_key,
        'point_count': int(coords.shape[0]),
        'feature_count': feature_count,
        'x_min': float(coords[:, 0].min()),
        'x_max': float(coords[:, 0].max()),
        'y_min': float(coords[:, 1].min()),
        'y_max': float(coords[:, 1].max()),
        'x_unique': int(np.unique(coords[:, 0]).shape[0]),
        'y_unique': int(np.unique(coords[:, 1]).shape[0]),
        'x_step_gcd': x_step,
        'y_step_gcd': y_step,
        'min_step_gcd': min_step,
        'diagnostic_mode': recommended,
    }


def sample_coords(coords, max_points, rng):
    if max_points <= 0 or coords.shape[0] <= max_points:
        return coords
    idx = rng.choice(coords.shape[0], size=max_points, replace=False)
    return coords[np.sort(idx)]


def svg_escape(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def write_scatter_svg(path, coords, summary, out_path, svg_size=900, max_points=8000, rng=None):
    rng = rng or np.random.default_rng(1024)
    pts = sample_coords(coords, max_points, rng)
    width = int(svg_size)
    height = int(svg_size)
    margin = 56
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    x_min, x_max = summary['x_min'], summary['x_max']
    y_min, y_max = summary['y_min'], summary['y_max']
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)

    x_svg = margin + (pts[:, 0] - x_min) / x_span * plot_w
    y_svg = margin + (pts[:, 1] - y_min) / y_span * plot_h

    title = f"{summary['slide']}  n={summary['point_count']}  step={summary['min_step_gcd']}  looks={summary['diagnostic_mode']}"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf7"/>',
        f'<text x="{margin}" y="30" font-family="monospace" font-size="16" fill="#222">{svg_escape(title)}</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#222" stroke-width="1"/>',
        f'<text x="{margin}" y="{height - 18}" font-family="monospace" font-size="12" fill="#555">x: {x_min:.0f} .. {x_max:.0f}</text>',
        f'<text x="{width - margin}" y="{height - 18}" font-family="monospace" font-size="12" fill="#555" text-anchor="end">y: {y_min:.0f} .. {y_max:.0f}</text>',
    ]
    radius = 1.7 if pts.shape[0] <= 3000 else 1.1
    opacity = 0.62 if pts.shape[0] <= 3000 else 0.34
    for x, y in zip(x_svg, y_svg):
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="#2563eb" fill-opacity="{opacity}"/>')
    lines.append('</svg>')
    with open(out_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines) + '\n')


def write_overview_svg(summaries, out_path):
    width, height = 1100, max(360, 24 * len(summaries) + 100)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf7"/>',
        '<text x="24" y="34" font-family="monospace" font-size="18" fill="#222">H5 coordinate summary</text>',
        '<text x="24" y="64" font-family="monospace" font-size="12" fill="#555">slide | n | x range | y range | gcd step | diagnostic mode</text>',
    ]
    y = 92
    for item in summaries:
        text = (
            f"{item['slide']} | n={item['point_count']} | "
            f"x={item['x_min']:.0f}..{item['x_max']:.0f} | "
            f"y={item['y_min']:.0f}..{item['y_max']:.0f} | "
            f"step={item['min_step_gcd']} | {item['diagnostic_mode']}"
        )
        color = '#166534' if item['diagnostic_mode'] == 'grid' else '#9a3412'
        lines.append(f'<text x="24" y="{y}" font-family="monospace" font-size="13" fill="{color}">{svg_escape(text)}</text>')
        y += 24
    lines.append('</svg>')
    with open(out_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    h5_files = find_h5_files(args)
    if args.max_files > 0:
        h5_files = h5_files[:args.max_files]
    if not h5_files:
        raise SystemExit('No h5 files found.')

    rng = np.random.default_rng(args.seed)
    summaries = []
    for idx, h5_path in enumerate(h5_files, start=1):
        coords, coord_key, feature_count = read_h5_info(h5_path)
        summary = summarize_coords(h5_path, coords, coord_key, feature_count, args.pixel_step_threshold)
        summaries.append(summary)
        out_svg = os.path.join(args.out_dir, safe_name(h5_path) + '_coords.svg')
        write_scatter_svg(h5_path, coords, summary, out_svg, args.svg_size, args.max_points, rng)
        print(
            f"[coords] {idx}/{len(h5_files)} {summary['slide']}: "
            f"n={summary['point_count']} x={summary['x_min']:.0f}..{summary['x_max']:.0f} "
            f"y={summary['y_min']:.0f}..{summary['y_max']:.0f} "
            f"step={summary['min_step_gcd']} looks={summary['diagnostic_mode']} -> {out_svg}",
            flush=True,
        )

    overview_path = os.path.join(args.out_dir, 'overview.svg')
    summary_path = os.path.join(args.out_dir, 'summary.json')
    write_overview_svg(summaries, overview_path)
    with open(summary_path, 'w', encoding='utf8') as f:
        json.dump({'files': summaries}, f, indent=2)

    modes = {}
    for item in summaries:
        modes[item['diagnostic_mode']] = modes.get(item['diagnostic_mode'], 0) + 1
    print(f'[coords] diagnostic modes: {modes}')
    print(f'[coords] wrote: {overview_path}')
    print(f'[coords] wrote: {summary_path}')


if __name__ == '__main__':
    main()
