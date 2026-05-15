import argparse
import ast
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote
from xml.sax.saxutils import escape


NUMBER_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
PRET_XML_IDS = {
    'mask': 1,
    'box': 2,
    'roughMask': 3,
}


def load_label_map(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf8') as f:
        raw = json.load(f)
    out = {}
    for label, value in raw.items():
        out[str(label)] = int(value)
    return out


def write_label_map(label_map, path):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)


def load_size_map(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf8') as f:
        raw = json.load(f)
    out = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            width = value.get('width', value.get('w'))
            height = value.get('height', value.get('h'))
        else:
            width, height = value[:2]
        out[str(key)] = (int(width), int(height))
    return out


def normalize_header(name):
    return name.strip().strip('"').strip("'").lower()


def decode_wsi_path(path):
    path = str(path).strip()
    return unquote(path) if '%' in path else path


def slide_name_from_path(wsi_path, name_mode):
    base = os.path.basename(decode_wsi_path(wsi_path))
    if name_mode == 'basename':
        return base
    return os.path.splitext(base)[0]


def resolve_wsi_path(raw_path, csv_path):
    raw_path = decode_wsi_path(raw_path)
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(os.path.dirname(csv_path), raw_path))


def lookup_slide_size(wsi_path, slide_name, size_map, h5_size_map=None, use_openslide=True):
    h5_size_map = h5_size_map or {}
    decoded_wsi_path = decode_wsi_path(wsi_path)
    candidates = [
        wsi_path,
        decoded_wsi_path,
        os.path.abspath(wsi_path),
        os.path.abspath(decoded_wsi_path),
        os.path.basename(wsi_path),
        os.path.basename(decoded_wsi_path),
        slide_name,
    ]
    for key in candidates:
        if key in size_map:
            return size_map[key]
    for key in candidates:
        if key in h5_size_map:
            return h5_size_map[key]

    if not use_openslide:
        raise ValueError(
            f'No size entry found for {wsi_path}. Add it to --size-json, '
            'provide a matching h5 file via --h5-dir, or allow OpenSlide.'
        )

    try:
        import openslide
        slide = openslide.OpenSlide(wsi_path)
        width, height = slide.level_dimensions[0]
        slide.close()
        return int(width), int(height)
    except Exception as exc:
        raise ValueError(
            f'Cannot read WSI size for {wsi_path}: {exc}. '
            'For sdpc files not supported by OpenSlide, pass --size-json with '
            '{"slide_name": {"width": W, "height": H}}.'
        )


def parse_points(coord_text, width, height, clip=True):
    nums = [float(x) for x in NUMBER_RE.findall(coord_text)]
    if len(nums) < 4 or len(nums) % 2 != 0:
        raise ValueError(f'coordinates must contain 2 or more x/y pairs, got: {coord_text!r}')

    points = []
    for i in range(0, len(nums), 2):
        x_rel, y_rel = nums[i], nums[i + 1]
        if clip:
            x_rel = min(1.0, max(0.0, x_rel))
            y_rel = min(1.0, max(0.0, y_rel))
        elif not (0 <= x_rel <= 1 and 0 <= y_rel <= 1):
            raise ValueError(f'normalized coordinate out of [0, 1]: ({x_rel}, {y_rel})')
        points.append((x_rel * width, y_rel * height))

    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        points = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]
    return points


def normalize_label_value(value):
    label = str(value).strip()
    while len(label) >= 2 and label[0] == label[-1] and label[0] in ['"', "'"]:
        label = label[1:-1].strip()
    return label


def flatten_label_values(value):
    if isinstance(value, (list, tuple, set)):
        labels = []
        for item in value:
            labels.extend(flatten_label_values(item))
        return labels
    label = normalize_label_value(value)
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


def label_to_id(label, label_map, auto=False):
    label = str(label).strip()
    if label in label_map:
        return label_map[label]
    if auto:
        next_id = max(label_map.values(), default=0) + 1
        label_map[label] = next_id
        return next_id
    try:
        return int(label)
    except ValueError as exc:
        raise KeyError(f'label {label!r} not found in label map') from exc


def should_keep_label(label, label_id, positive_labels, include_zero_labels):
    if positive_labels is not None:
        return label in positive_labels or str(label_id) in positive_labels
    if include_zero_labels:
        return True
    return label_id != 0


def pret_annotation_id(label_id, prompt_type):
    if prompt_type == 'label-id':
        return label_id
    return PRET_XML_IDS[prompt_type]


def read_regions(args, label_map, size_map, h5_size_map=None):
    regions_by_slide = defaultdict(list)
    skipped = defaultdict(int)
    csv_path = os.path.abspath(args.csv)
    positive_labels = set(args.positive_labels) if args.positive_labels else None

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV file has no header row')
        field_lookup = {normalize_header(name): name for name in reader.fieldnames}
        for required in ['wsi_path', 'labels', 'coordinates']:
            if required not in field_lookup:
                raise ValueError(f'CSV must contain column {required!r}; got {reader.fieldnames}')

        for row_idx, row in enumerate(reader, start=2):
            raw_wsi_path = row[field_lookup['wsi_path']]
            labels = parse_label_values(row[field_lookup['labels']])
            coords = row[field_lookup['coordinates']]
            wsi_path = resolve_wsi_path(raw_wsi_path, csv_path)
            slide_name = slide_name_from_path(wsi_path, args.name_mode)
            width, height = lookup_slide_size(
                wsi_path, slide_name, size_map, h5_size_map, use_openslide=not args.no_openslide
            )
            points = parse_points(coords, width, height, clip=not args.no_clip)

            if not labels:
                skipped['<empty>'] += 1
                continue

            for label in labels:
                label_id = label_to_id(label, label_map, auto=args.auto_label_map)
                if not should_keep_label(label, label_id, positive_labels, args.include_zero_labels):
                    skipped[label] += 1
                    continue

                regions_by_slide[slide_name].append({
                    'row': row_idx,
                    'wsi_path': wsi_path,
                    'label': label,
                    'label_id': label_id,
                    'annotation_id': pret_annotation_id(label_id, args.prompt_type),
                    'points': points,
                    'size': (width, height),
                })
    return regions_by_slide, skipped


def write_xml_files(regions_by_slide, xml_out):
    if not xml_out:
        return 0
    os.makedirs(xml_out, exist_ok=True)
    written = 0
    for slide_name, regions in sorted(regions_by_slide.items()):
        annotations = defaultdict(list)
        for region in regions:
            annotations[region['annotation_id']].append(region)

        lines = ['<?xml version="1.0"?>', '<Annotations>']
        for annotation_id, annotation_regions in sorted(annotations.items()):
            names = sorted({r['label'] for r in annotation_regions})
            name_attr = escape('+'.join(names))
            lines.append(f'  <Annotation Id="{annotation_id}" Name="{name_attr}" Type="4" Visible="1">')
            lines.append('    <Regions>')
            for region_id, region in enumerate(annotation_regions):
                lines.append(f'      <Region Id="{region_id}" Type="0">')
                lines.append('        <Vertices>')
                for x, y in region['points']:
                    lines.append(f'          <Vertex X="{x:.3f}" Y="{y:.3f}" />')
                lines.append('        </Vertices>')
                lines.append('      </Region>')
            lines.append('    </Regions>')
            lines.append('  </Annotation>')
        lines.append('</Annotations>')

        out_path = os.path.join(xml_out, slide_name + '.xml')
        with open(out_path, 'w', encoding='utf8') as f:
            f.write('\n'.join(lines) + '\n')
        written += 1
    return written


def rasterize_patch_mask(regions, patch_scale):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing mask PNGs requires cv2 and numpy') from exc

    width, height = regions[0]['size']
    grid_w = int(math.ceil(width / patch_scale))
    grid_h = int(math.ceil(height / patch_scale))
    pad_w = grid_w * patch_scale
    pad_h = grid_h * patch_scale

    mid_scale = max(1, patch_scale // 32)
    while patch_scale % mid_scale != 0:
        mid_scale -= 1
    resize_scale = patch_scale // mid_scale
    mid_h = pad_h // mid_scale
    mid_w = pad_w // mid_scale
    mid = np.zeros((mid_h, mid_w), dtype=np.uint8)

    for region in regions:
        contour = np.array(
            [[int(x / mid_scale + 0.5), int(y / mid_scale + 0.5)] for x, y in region['points']],
            dtype=np.int32,
        )
        if contour.shape[0] >= 3:
            cv2.fillPoly(mid, [contour], 1)

    mask = mid.reshape(grid_h, resize_scale, grid_w, resize_scale).max(axis=(1, 3))
    return mask.astype(np.uint8)


def write_mask_files(regions_by_slide, mask_out, patch_scale):
    if not mask_out:
        return {}, 0
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Writing mask PNGs requires cv2') from exc

    os.makedirs(mask_out, exist_ok=True)
    pos_counts = {}
    written = 0
    for slide_name, regions in sorted(regions_by_slide.items()):
        mask = rasterize_patch_mask(regions, patch_scale)
        out_path = os.path.join(mask_out, slide_name + '.png')
        cv2.imwrite(out_path, mask)
        pos_counts[slide_name] = int((mask == 1).sum())
        written += 1
    return pos_counts, written


def find_h5_files(path):
    if not path:
        return {}
    h5_files = {}
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(('.h5', '.hdf5')):
            stem = os.path.splitext(name)[0]
            h5_path = os.path.join(path, name)
            h5_files[stem] = h5_path
            h5_files[decode_wsi_path(stem)] = h5_path
    return h5_files


def infer_axis_step(values):
    values = sorted({int(round(float(value))) for value in values})
    diffs = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not diffs:
        return None
    step = diffs[0]
    for diff in diffs[1:]:
        step = math.gcd(step, diff)
    if step <= 0:
        return None
    return step


def infer_patch_scale_from_coordinates(coords):
    steps = []
    x_step = infer_axis_step(coords[:, 0])
    y_step = infer_axis_step(coords[:, 1])
    if x_step is not None:
        steps.append(x_step)
    if y_step is not None:
        steps.append(y_step)
    if not steps:
        return None
    return min(steps)


def read_h5_coordinates(h5_path):
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Reading h5 coordinates requires h5py and numpy') from exc

    with h5py.File(h5_path, 'r') as f:
        if 'coordinates' not in f:
            raise KeyError(f'{h5_path} must contain h5 key: coordinates')
        coords = np.asarray(f['coordinates'])
        if 'features' in f and f['features'].shape[0] != coords.shape[0]:
            raise ValueError(
                f'{h5_path}: features and coordinates must have the same first dimension, '
                f'got {f["features"].shape[0]} and {coords.shape[0]}'
            )

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f'{h5_path}: coordinates must have shape (N, >=2), got {coords.shape}')
    return coords


def infer_h5_metadata(h5_files, explicit_patch_scale):
    metadata = {}
    inferred_patch_scales = {}
    for slide_name, h5_path in sorted(h5_files.items()):
        coords = read_h5_coordinates(h5_path)
        inferred_patch_scale = infer_patch_scale_from_coordinates(coords)
        if explicit_patch_scale > 0:
            patch_scale = explicit_patch_scale
        elif inferred_patch_scale is not None:
            patch_scale = inferred_patch_scale
        else:
            patch_scale = 512
            print(
                f'[warning] Could not infer patch scale from {h5_path}; '
                f'falling back to {patch_scale}. Pass --patch-scale to override.'
            )

        width = int(math.ceil(float(coords[:, 0].max()) + patch_scale))
        height = int(math.ceil(float(coords[:, 1].max()) + patch_scale))
        metadata[slide_name] = {
            'path': h5_path,
            'coordinates': coords,
            'patch_scale': int(patch_scale),
            'width': width,
            'height': height,
            'inferred_patch_scale': inferred_patch_scale,
        }
        if inferred_patch_scale is not None:
            inferred_patch_scales[int(inferred_patch_scale)] = inferred_patch_scales.get(int(inferred_patch_scale), 0) + 1

    return metadata, inferred_patch_scales


def choose_patch_scale(args, h5_metadata, inferred_patch_scales):
    if args.patch_scale > 0:
        if inferred_patch_scales:
            inferred = max(inferred_patch_scales, key=inferred_patch_scales.get)
            if inferred != args.patch_scale:
                print(
                    f'[warning] Explicit --patch-scale={args.patch_scale}, but the most common '
                    f'h5 coordinate step is {inferred}. Using the explicit value.'
                )
        return args.patch_scale

    if inferred_patch_scales:
        patch_scale = max(inferred_patch_scales, key=inferred_patch_scales.get)
        print(f'[info] Inferred patch scale from h5 coordinates: {patch_scale}')
        return patch_scale

    if h5_metadata:
        patch_scale = next(iter(h5_metadata.values()))['patch_scale']
        print(f'[info] Using fallback patch scale from h5 metadata: {patch_scale}')
        return patch_scale

    print('[info] No --h5-dir metadata available; using default patch scale 512.')
    return 512


def h5_size_map_from_metadata(h5_metadata, patch_scale):
    out = {}
    for slide_name, info in h5_metadata.items():
        width = int(math.ceil(float(info['coordinates'][:, 0].max()) + patch_scale))
        height = int(math.ceil(float(info['coordinates'][:, 1].max()) + patch_scale))
        out[slide_name] = (width, height)
    return out


def rasterize_mid_mask(regions, patch_scale):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc

    try:
        import cv2
    except ImportError:
        cv2 = None

    width, height = regions[0]['size']
    mid_scale = max(1, patch_scale // 32)
    while patch_scale % mid_scale != 0:
        mid_scale -= 1
    mid_h = int(math.ceil(height / mid_scale))
    mid_w = int(math.ceil(width / mid_scale))
    mid = np.zeros((mid_h, mid_w), dtype=np.uint8)

    for region in regions:
        contour = np.array(
            [[int(x / mid_scale + 0.5), int(y / mid_scale + 0.5)] for x, y in region['points']],
            dtype=np.int32,
        )
        if contour.shape[0] >= 3:
            if cv2 is not None:
                cv2.fillPoly(mid, [contour], 1)
            else:
                fill_polygon_numpy(mid, contour)
    return mid, mid_scale


def fill_polygon_numpy(mask, contour):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc

    min_x = max(0, int(np.floor(contour[:, 0].min())))
    max_x = min(mask.shape[1], int(np.ceil(contour[:, 0].max())) + 1)
    min_y = max(0, int(np.floor(contour[:, 1].min())))
    max_y = min(mask.shape[0], int(np.ceil(contour[:, 1].max())) + 1)
    if max_x <= min_x or max_y <= min_y:
        return

    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]
    x = xx + 0.5
    y = yy + 0.5
    inside = np.zeros(x.shape, dtype=bool)

    points = contour.astype(float)
    xj, yj = points[-1]
    for xi, yi in points:
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= crosses
        xj, yj = xi, yi

    mask[min_y:max_y, min_x:max_x][inside] = 1


def labels_from_mid_mask(coords, mid, mid_scale, patch_scale):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc

    labels = np.zeros(coords.shape[0], dtype=np.uint8)
    mid_h, mid_w = mid.shape
    for idx, coord in enumerate(coords):
        x0 = float(coord[0])
        y0 = float(coord[1])
        x1 = x0 + patch_scale
        y1 = y0 + patch_scale
        mx0 = max(0, int(math.floor(x0 / mid_scale)))
        my0 = max(0, int(math.floor(y0 / mid_scale)))
        mx1 = min(mid_w, int(math.ceil(x1 / mid_scale)))
        my1 = min(mid_h, int(math.ceil(y1 / mid_scale)))
        if mx1 <= mx0 or my1 <= my0:
            continue
        labels[idx] = 1 if mid[my0:my1, mx0:mx1].max() > 0 else 0
    return labels


def labels_for_h5_coordinates(regions, h5_path, patch_scale, multi_label=False, class_num=0, binary_labels=True):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc

    coords = read_h5_coordinates(h5_path)

    if multi_label:
        if class_num <= 0:
            raise ValueError('--multi-label h5 label writing requires at least one class')
        labels = np.zeros((coords.shape[0], class_num), dtype=np.uint8)
        for cls in range(1, class_num + 1):
            cls_regions = [r for r in regions if int(r['label_id']) == cls]
            if not cls_regions:
                continue
            mid, mid_scale = rasterize_mid_mask(cls_regions, patch_scale)
            labels[:, cls - 1] = labels_from_mid_mask(coords, mid, mid_scale, patch_scale)
        return labels

    if binary_labels:
        mid, mid_scale = rasterize_mid_mask(regions, patch_scale)
        return labels_from_mid_mask(coords, mid, mid_scale, patch_scale)

    labels = np.zeros(coords.shape[0], dtype=np.uint16)
    for cls in sorted({int(r['label_id']) for r in regions}):
        cls_regions = [r for r in regions if int(r['label_id']) == cls]
        mid, mid_scale = rasterize_mid_mask(cls_regions, patch_scale)
        cls_hits = labels_from_mid_mask(coords, mid, mid_scale, patch_scale)
        labels[cls_hits == 1] = cls
    return labels


def write_h5_label_files(
    regions_by_slide,
    h5_dir,
    h5_label_out,
    patch_scale,
    multi_label=False,
    class_num=0,
    binary_labels=True,
):
    if not h5_label_out:
        return {}, 0
    if not h5_dir:
        raise ValueError('--h5-dir is required when --h5-label-out is set')

    h5_files = find_h5_files(h5_dir)
    if not h5_files:
        raise ValueError(f'No .h5/.hdf5 files found in {h5_dir}')

    os.makedirs(h5_label_out, exist_ok=True)
    pos_counts = {}
    written = 0
    missing = []
    for slide_name, regions in sorted(regions_by_slide.items()):
        if slide_name not in h5_files:
            missing.append(slide_name)
            continue
        labels = labels_for_h5_coordinates(
            regions,
            h5_files[slide_name],
            patch_scale,
            multi_label=multi_label,
            class_num=class_num,
            binary_labels=binary_labels,
        )
        out_path = os.path.join(h5_label_out, slide_name + '.npy')
        np_save(out_path, labels)
        pos_counts[slide_name] = int((labels == 1).sum())
        written += 1

    if missing:
        preview = ', '.join(missing[:10])
        raise ValueError(
            f'Missing h5 files for {len(missing)} annotated slide(s): {preview}. '
            'H5 file stems must match CSV-derived slide names.'
        )
    return pos_counts, written


def np_save(path, value):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc
    np.save(path, value)


def write_data_info(
    regions_by_slide,
    data_info_out,
    mask_out,
    pos_counts,
    wsi_label_mode,
    h5_label_out='',
    h5_pos_counts=None,
):
    if not data_info_out:
        return
    h5_pos_counts = h5_pos_counts or {}
    out = {}
    for slide_name, regions in sorted(regions_by_slide.items()):
        label_ids = sorted({int(r['label_id']) for r in regions})
        if wsi_label_mode == 'binary':
            wsi_label = 1
        elif wsi_label_mode == 'single-label':
            if len(label_ids) != 1:
                raise ValueError(f'{slide_name} has multiple labels {label_ids}; cannot use single-label mode')
            wsi_label = label_ids[0]
        elif wsi_label_mode == 'multi-label':
            wsi_label = None
        else:
            wsi_label = max(label_ids)

        item = {
            'fixed_test_set': False,
        }
        if wsi_label_mode == 'multi-label':
            item['wsi_labels'] = [int(_) for _ in label_ids]
        else:
            item['wsi_label'] = int(wsi_label)
        if mask_out:
            item['patch_labels'] = os.path.join(mask_out, slide_name + '.png')
        if h5_label_out:
            item['h5_patch_labels'] = os.path.join(h5_label_out, slide_name + '.npy')
        if mask_out or h5_label_out:
            item['pos_patch_num'] = int(h5_pos_counts.get(slide_name, pos_counts.get(slide_name, 0)))
        out[slide_name] = item

    out_dir = os.path.dirname(data_info_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(data_info_out, 'w', encoding='utf8') as f:
        json.dump(out, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert normalized CSV annotations to PRET-compatible XML and optional patch-mask PNGs.'
    )
    parser.add_argument('--csv', required=True, help='CSV with wsi_path, labels, coordinates columns')
    parser.add_argument('--label-map', default='', help='optional JSON mapping label names to integer ids')
    parser.add_argument('--auto-label-map', action='store_true',
        help='assign missing labels ids in first-seen CSV order, starting at max(existing ids)+1')
    parser.add_argument('--label-map-out', default='',
        help='optional path to write the final label map, useful with --auto-label-map')
    parser.add_argument('--xml-out', default='', help='directory to write PRET-compatible XML files')
    parser.add_argument('--mask-out', default='', help='optional directory to write PRET patch-grid PNG masks')
    parser.add_argument('--h5-dir', default='', help='optional directory containing per-slide h5 feature files')
    parser.add_argument('--h5-label-out', default='',
        help='optional directory to write per-h5 patch label .npy arrays aligned to h5 feature order')
    parser.add_argument('--data-info-out', default='', help='optional data_info JSON for annotated slides')
    parser.add_argument('--size-json', default='',
        help='optional slide size JSON for formats OpenSlide cannot read; h5 coordinates can infer this when --h5-dir is set')
    parser.add_argument('--patch-scale', type=int, default=0,
        help='level-0 pixels per patch; 0 infers from h5 coordinates when --h5-dir is set, otherwise uses 512')
    parser.add_argument('--prompt-type', default='mask',
        choices=['mask', 'box', 'roughMask', 'label-id'],
        help='XML Annotation Id convention: mask=1, box=2, roughMask=3, label-id preserves the label map id')
    parser.add_argument('--positive-labels', nargs='*',
        help='optional label names or ids to keep; by default all non-zero label ids are kept')
    parser.add_argument('--include-zero-labels', action='store_true',
        help='keep rows whose mapped label id is 0')
    parser.add_argument('--name-mode', default='stem', choices=['stem', 'basename'],
        help='output XML/PNG slide key from WSI stem or full basename')
    parser.add_argument('--wsi-label-mode', default='binary', choices=['binary', 'single-label', 'max-label', 'multi-label'],
        help='wsi_label policy when --data-info-out is used')
    parser.add_argument('--no-openslide', action='store_true',
        help='do not try to read WSI dimensions with OpenSlide; require --size-json or matching --h5-dir metadata')
    parser.add_argument('--no-clip', action='store_true',
        help='raise on normalized coordinates outside [0,1] instead of clipping')
    args = parser.parse_args()
    if not args.xml_out and not args.mask_out and not args.h5_label_out and not args.data_info_out:
        parser.error('provide at least one of --xml-out, --mask-out, --h5-label-out, or --data-info-out')
    if not args.label_map and not args.auto_label_map:
        parser.error('provide --label-map, or use --auto-label-map to create one from CSV labels')
    if args.h5_label_out and not args.h5_dir:
        parser.error('--h5-dir is required when --h5-label-out is set')
    if args.patch_scale < 0:
        parser.error('--patch-scale must be >= 0')
    return args


def main():
    args = parse_args()
    label_map = load_label_map(args.label_map)
    size_map = load_size_map(args.size_json)
    h5_files = find_h5_files(args.h5_dir)
    h5_metadata, inferred_patch_scales = infer_h5_metadata(h5_files, args.patch_scale) if h5_files else ({}, {})
    args.patch_scale = choose_patch_scale(args, h5_metadata, inferred_patch_scales)
    h5_size_map = h5_size_map_from_metadata(h5_metadata, args.patch_scale)
    regions_by_slide, skipped = read_regions(args, label_map, size_map, h5_size_map)
    if not regions_by_slide:
        raise SystemExit('No regions were converted. Check labels, label map, and --positive-labels.')
    write_label_map(label_map, args.label_map_out)

    xml_count = write_xml_files(regions_by_slide, args.xml_out)
    pos_counts, mask_count = write_mask_files(regions_by_slide, args.mask_out, args.patch_scale)
    h5_multi_label = args.wsi_label_mode == 'multi-label'
    h5_binary_labels = args.wsi_label_mode == 'binary'
    class_num = max(label_map.values(), default=0)
    h5_pos_counts, h5_label_count = write_h5_label_files(
        regions_by_slide,
        args.h5_dir,
        args.h5_label_out,
        args.patch_scale,
        multi_label=h5_multi_label,
        class_num=class_num,
        binary_labels=h5_binary_labels,
    )
    write_data_info(
        regions_by_slide,
        args.data_info_out,
        args.mask_out,
        pos_counts,
        args.wsi_label_mode,
        h5_label_out=args.h5_label_out,
        h5_pos_counts=h5_pos_counts,
    )

    region_count = sum(len(v) for v in regions_by_slide.values())
    print(f'converted slides: {len(regions_by_slide)}')
    print(f'converted regions: {region_count}')
    if xml_count:
        print(f'xml files written: {xml_count} -> {args.xml_out}')
    if mask_count:
        print(f'mask png files written: {mask_count} -> {args.mask_out}')
    if h5_label_count:
        print(f'h5 patch label arrays written: {h5_label_count} -> {args.h5_label_out}')
    if args.label_map_out:
        print(f'label_map written: {args.label_map_out}')
    if args.data_info_out:
        print(f'data_info written: {args.data_info_out}')
    if skipped:
        skipped_text = ', '.join(f'{label}:{count}' for label, count in sorted(skipped.items()))
        print(f'skipped labels: {skipped_text}')


if __name__ == '__main__':
    main()
