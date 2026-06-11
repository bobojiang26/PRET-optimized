import argparse
import ast
import csv
import gc
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote
from xml.sax.saxutils import escape


NUMBER_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
PRET_XML_IDS = {
    'mask': 1,
    'box': 2,
    'roughMask': 3,
}
DEFAULT_WSI_EXTENSIONS = (
    '.sdpc',
    '.svs',
    '.tif',
    '.tiff',
    '.mrxs',
    '.ndpi',
    '.scn',
    '.vms',
    '.vmu',
    '.bif',
)
DEFAULT_SLIDE_READERS = ('openslide', 'opensdpc')
DEFAULT_PATCH_SCALE = 512
PROGRESS_INTERVAL = 10
H5_EXTENSIONS = ('.h5', '.hdf5')
H5_COORDINATE_KEYS = ('coords', 'coordinates')


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


def write_size_json(size_map, path):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    serializable = {}
    for key, value in sorted(size_map.items()):
        if isinstance(value, dict):
            width = value.get('width', value.get('w'))
            height = value.get('height', value.get('h'))
        else:
            width, height = value[:2]
        serializable[str(key)] = {'width': int(width), 'height': int(height)}
    with open(path, 'w', encoding='utf8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=4)


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


def normalize_extensions(extensions):
    if not extensions:
        return DEFAULT_WSI_EXTENSIONS
    out = []
    for ext in extensions:
        ext = str(ext).strip().lower()
        if not ext:
            continue
        out.append(ext if ext.startswith('.') else f'.{ext}')
    return tuple(out) if out else DEFAULT_WSI_EXTENSIONS


def strip_known_slide_suffixes(value):
    value = str(value).strip()
    known_exts = set(DEFAULT_WSI_EXTENSIONS) | set(H5_EXTENSIONS)
    changed = True
    while changed:
        changed = False
        root, ext = os.path.splitext(value)
        if ext.lower() in known_exts:
            value = root
            changed = True
    return value


def slide_lookup_keys(value):
    raw = str(value).strip()
    decoded = decode_wsi_path(raw)
    variants = {
        raw,
        decoded,
        os.path.basename(raw),
        os.path.basename(decoded),
        os.path.abspath(raw),
        os.path.abspath(decoded),
    }
    expanded = set()
    for key in variants:
        if not key:
            continue
        decoded_key = decode_wsi_path(key)
        base = os.path.basename(decoded_key)
        expanded.update({
            key,
            decoded_key,
            base,
            os.path.splitext(key)[0],
            os.path.splitext(decoded_key)[0],
            os.path.splitext(base)[0],
            strip_known_slide_suffixes(key),
            strip_known_slide_suffixes(decoded_key),
            strip_known_slide_suffixes(base),
        })
    return {key for key in expanded if key}


def lookup_by_slide_keys(mapping, *values):
    if not mapping:
        return None
    for value in values:
        for key in slide_lookup_keys(value):
            if key in mapping:
                return mapping[key]
    return None


def iter_wsi_paths(path, extensions=None, recursive=False):
    if not path:
        return []
    root = Path(path)
    extensions = normalize_extensions(extensions)
    if root.is_file():
        return [root] if root.suffix.lower() in extensions else []
    if not root.exists():
        raise FileNotFoundError(f'WSI directory does not exist: {path}')
    iterator = root.rglob('*') if recursive else root.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in extensions)


def wsi_lookup_keys(path):
    return slide_lookup_keys(path)


def find_wsi_files(path, extensions=None, recursive=False):
    wsi_files = {}
    for wsi_path in iter_wsi_paths(path, extensions=extensions, recursive=recursive):
        for key in wsi_lookup_keys(wsi_path):
            wsi_files[key] = str(wsi_path)
    return wsi_files


def close_slide(slide):
    close_fn = getattr(slide, 'close', None)
    if callable(close_fn):
        close_fn()


def size_from_slide_object(slide, slide_path):
    if hasattr(slide, 'level_dimensions'):
        width, height = slide.level_dimensions[0]
        return int(width), int(height)
    if hasattr(slide, 'dimensions'):
        width, height = slide.dimensions
        return int(width), int(height)
    get_level_dimensions = getattr(slide, 'get_level_dimensions', None)
    if callable(get_level_dimensions):
        width, height = get_level_dimensions(0)
        return int(width), int(height)
    width = getattr(slide, 'width', None)
    height = getattr(slide, 'height', None)
    if width is not None and height is not None:
        return int(width), int(height)
    raise AttributeError(f'{slide_path}: slide object does not expose level-0 dimensions')


def read_size_with_openslide(slide_path):
    import openslide
    slide = openslide.OpenSlide(slide_path)
    try:
        return size_from_slide_object(slide, slide_path)
    finally:
        close_slide(slide)


def read_size_with_opensdpc(slide_path):
    import opensdpc

    opener = (
        getattr(opensdpc, 'OpenSdpc', None)
        or getattr(opensdpc, 'OpenSDPC', None)
        or getattr(opensdpc, 'OpenSlide', None)
    )
    if opener is None:
        raise AttributeError('opensdpc does not expose OpenSdpc/OpenSDPC/OpenSlide')

    slide = opener(slide_path)
    try:
        return size_from_slide_object(slide, slide_path)
    finally:
        close_slide(slide)


def read_slide_size(slide_path, readers=DEFAULT_SLIDE_READERS):
    readers = tuple(readers or [])
    errors = []
    for reader in readers:
        try:
            if reader == 'openslide':
                return read_size_with_openslide(slide_path)
            if reader == 'opensdpc':
                return read_size_with_opensdpc(slide_path)
            raise ValueError(f'unknown slide reader: {reader}')
        except Exception as exc:
            errors.append(f'{reader}: {exc}')
    if not readers:
        errors.append('no slide readers enabled')
    raise ValueError(f'Cannot read WSI size for {slide_path}. Tried {", ".join(errors)}')


def build_wsi_size_maps(
    wsi_dir,
    extensions=None,
    recursive=False,
    readers=DEFAULT_SLIDE_READERS,
    name_mode='stem',
    strict=True,
    progress_interval=PROGRESS_INTERVAL,
):
    lookup_map = {}
    json_map = {}
    failures = []
    print(f'[info] scanning WSI directory: dir={wsi_dir} recursive={recursive}', flush=True)
    wsi_paths = iter_wsi_paths(wsi_dir, extensions=extensions, recursive=recursive)
    total = len(wsi_paths)
    print(f'[info] scanning WSI sizes: files={total} dir={wsi_dir}', flush=True)
    start = time.perf_counter()
    for idx, wsi_path in enumerate(wsi_paths, start=1):
        width = None
        height = None
        try:
            width, height = read_slide_size(str(wsi_path), readers=readers)
        except Exception as exc:
            if strict:
                raise
            failures.append((str(wsi_path), str(exc)))
        if width is not None and height is not None:
            slide_name = slide_name_from_path(str(wsi_path), name_mode)
            json_map[slide_name] = {'width': width, 'height': height}
            for key in wsi_lookup_keys(wsi_path):
                lookup_map[key] = (width, height)
        if progress_interval > 0 and (idx == 1 or idx % progress_interval == 0 or idx == total):
            elapsed = time.perf_counter() - start
            print(
                f'[info] WSI size progress {idx}/{total}: readable={len(json_map)} '
                f'failed={len(failures)} elapsed={elapsed:.1f}s',
                flush=True,
            )
    if failures:
        preview = '; '.join(f'{os.path.basename(path)}: {err}' for path, err in failures[:5])
        print(f'[warning] skipped {len(failures)} WSI file(s) whose sizes could not be read: {preview}')
        if len(failures) > 5:
            print(f'[warning] ... {len(failures) - 5} more unreadable WSI file(s) omitted.')
    return lookup_map, json_map


def requested_slide_readers(args):
    if args.slide_reader == 'auto':
        readers = list(DEFAULT_SLIDE_READERS)
    else:
        readers = [args.slide_reader]
    if args.no_openslide:
        readers = [reader for reader in readers if reader != 'openslide']
    return readers


def lookup_slide_size(
    wsi_path,
    slide_name,
    size_map,
    h5_size_map=None,
    wsi_size_map=None,
    slide_readers=DEFAULT_SLIDE_READERS,
):
    h5_size_map = h5_size_map or {}
    wsi_size_map = wsi_size_map or {}
    matched = lookup_by_slide_keys(size_map, wsi_path, slide_name)
    if matched is not None:
        return matched
    matched = lookup_by_slide_keys(wsi_size_map, wsi_path, slide_name)
    if matched is not None:
        return matched
    matched = lookup_by_slide_keys(h5_size_map, wsi_path, slide_name)
    if matched is not None:
        return matched

    if not slide_readers:
        raise ValueError(
            f'No size entry found for {wsi_path}. Add it to --size-json, '
            'provide a matching h5 file via --h5-dir, pass --wsi-dir, or enable a slide reader.'
        )

    try:
        return read_slide_size(wsi_path, readers=slide_readers)
    except Exception as exc:
        raise ValueError(
            f'Cannot read WSI size for {wsi_path}: {exc}. '
            'For sdpc files, install/use opensdpc, pass --wsi-dir, or pass --size-json with '
            '{"slide_name": {"width": W, "height": H}}.'
        )


def parse_points(coord_text, width, height, clip=True):
    nums = [float(x) for x in NUMBER_RE.findall(coord_text)]
    if len(nums) < 4 or len(nums) % 2 != 0:
        raise ValueError(f'coordinates must contain 2 or more x/y pairs, got: {coord_text!r}')

    relative_points = []
    for i in range(0, len(nums), 2):
        x_rel, y_rel = nums[i], nums[i + 1]
        if clip:
            x_rel = min(1.0, max(0.0, x_rel))
            y_rel = min(1.0, max(0.0, y_rel))
        elif not (0 <= x_rel <= 1 and 0 <= y_rel <= 1):
            raise ValueError(f'normalized coordinate out of [0, 1]: ({x_rel}, {y_rel})')
        relative_points.append((x_rel, y_rel))

    if len(relative_points) == 2:
        (x1, y1), (x2, y2) = relative_points
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        relative_points = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]

    points = [(x * width, y * height) for x, y in relative_points]
    return points, relative_points


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


def read_regions(args, label_map, size_map, h5_size_map=None, wsi_size_map=None, slide_readers=None, h5_files=None):
    regions_by_slide = defaultdict(list)
    skipped = defaultdict(int)
    csv_path = os.path.abspath(args.csv)
    positive_labels = set(args.positive_labels) if args.positive_labels else None
    missing_h5 = defaultdict(int)
    h5_files = h5_files or {}
    progress_interval = PROGRESS_INTERVAL
    print(f'[info] reading annotation CSV: {csv_path}', flush=True)
    start = time.perf_counter()
    processed_rows = 0

    def report_csv_progress(force=False):
        should_report = (
            force or processed_rows == 1 or
            (progress_interval > 0 and processed_rows % progress_interval == 0)
        )
        if not should_report:
            return
        elapsed = time.perf_counter() - start
        region_count = sum(len(v) for v in regions_by_slide.values())
        print(
            f'[info] CSV progress rows={processed_rows} slides={len(regions_by_slide)} '
            f'regions={region_count} skipped_labels={sum(skipped.values())} '
            f'missing_h5_slides={len(missing_h5)} elapsed={elapsed:.1f}s',
            flush=True,
        )

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV file has no header row')
        field_lookup = {normalize_header(name): name for name in reader.fieldnames}
        for required in ['wsi_path', 'labels', 'coordinates']:
            if required not in field_lookup:
                raise ValueError(f'CSV must contain column {required!r}; got {reader.fieldnames}')

        for row_idx, row in enumerate(reader, start=2):
            processed_rows += 1
            raw_wsi_path = row[field_lookup['wsi_path']]
            labels = parse_label_values(row[field_lookup['labels']])
            coords = row[field_lookup['coordinates']]
            wsi_path = resolve_wsi_path(raw_wsi_path, csv_path)
            slide_name = slide_name_from_path(wsi_path, args.name_mode)
            if args.skip_missing_h5 and h5_files and lookup_by_slide_keys(h5_files, slide_name, wsi_path) is None:
                missing_h5[slide_name] += 1
                report_csv_progress()
                continue
            width, height = lookup_slide_size(
                wsi_path,
                slide_name,
                size_map,
                h5_size_map,
                wsi_size_map,
                slide_readers=slide_readers,
            )
            points, relative_points = parse_points(coords, width, height, clip=not args.no_clip)

            if not labels:
                skipped['<empty>'] += 1
                report_csv_progress()
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
                    'relative_points': relative_points,
                    'size': (width, height),
                })
            report_csv_progress()
    elapsed = time.perf_counter() - start
    region_count = sum(len(v) for v in regions_by_slide.values())
    print(
        f'[info] CSV read complete rows={processed_rows} slides={len(regions_by_slide)} '
        f'regions={region_count} skipped_labels={sum(skipped.values())} '
        f'missing_h5_slides={len(missing_h5)} elapsed={elapsed:.1f}s',
        flush=True,
    )
    if missing_h5:
        preview = ', '.join(f'{name}:{count}' for name, count in list(sorted(missing_h5.items()))[:20])
        print(f'[warning] skipped CSV rows for {len(missing_h5)} slide(s) without matching h5: {preview}')
        if len(missing_h5) > 20:
            print(f'[warning] ... {len(missing_h5) - 20} more missing-h5 slide(s) omitted.')
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

    mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

    # Adaptive chunk height in mid-scale: target ~50 MB per chunk
    target_chunk_bytes = 50 * 1024 * 1024
    chunk_mid_h = max(resize_scale, min(mid_h, max(1, target_chunk_bytes // max(1, mid_w))))
    chunk_mid_h = (chunk_mid_h // resize_scale) * resize_scale
    if chunk_mid_h < resize_scale:
        chunk_mid_h = resize_scale

    for chunk_mid_y0 in range(0, mid_h, chunk_mid_h):
        chunk_mid_y1 = min(mid_h, chunk_mid_y0 + chunk_mid_h)
        chunk_h = chunk_mid_y1 - chunk_mid_y0
        chunk_grid_h = chunk_h // resize_scale
        effective_chunk_h = chunk_grid_h * resize_scale

        chunk = np.zeros((effective_chunk_h, mid_w), dtype=np.uint8)

        for region in regions:
            pts = np.array(region['points'])
            py_min = pts[:, 1].min()
            py_max = pts[:, 1].max()
            if int(math.ceil(py_max / mid_scale)) <= chunk_mid_y0 or int(py_min / mid_scale) >= chunk_mid_y1:
                continue

            contour = np.array(
                [[int(x / mid_scale + 0.5), int(y / mid_scale + 0.5) - chunk_mid_y0]
                 for x, y in pts],
                dtype=np.int32,
            )
            if contour.shape[0] >= 3:
                cv2.fillPoly(chunk, [contour], 1)

        if chunk_grid_h > 0:
            grid_y0 = chunk_mid_y0 // resize_scale
            chunk_mask = chunk.reshape(
                chunk_grid_h, resize_scale, grid_w, resize_scale
            ).max(axis=(1, 3))
            mask[grid_y0:grid_y0 + chunk_grid_h, :] = chunk_mask

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


def find_h5_files(path, recursive=True):
    if not path:
        return {}
    print(f'[info] scanning h5 files: path={path} recursive={recursive}', flush=True)
    h5_files = {}
    root = Path(path)
    if root.is_file():
        candidates = [root] if root.suffix.lower() in H5_EXTENSIONS else []
    elif root.exists():
        iterator = root.rglob('*') if recursive else root.iterdir()
        candidates = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in H5_EXTENSIONS)
    else:
        candidates = []
    for h5_path in candidates:
        stem = h5_path.stem
        for key in slide_lookup_keys(stem):
            h5_files[key] = str(h5_path)
    print(f'[info] h5 files found: {len(candidates)}', flush=True)
    return h5_files


def infer_axis_step(values):
    values = sorted({int(round(float(value))) for value in values})
    diffs = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not diffs:
        return None
    counts = Counter(diffs)
    step, _ = min(counts.items(), key=lambda item: (-item[1], item[0]))
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


def infer_h5_coordinate_mode(coords, requested_mode='grid'):
    if requested_mode not in ['grid', 'pixel']:
        raise ValueError(
            'Set --h5-coordinate-mode to grid or pixel. Auto coordinate-mode detection has been removed; '
            'inspect coordinates first with scripts/visualize_h5_coordinates.py.'
        )
    return requested_mode


def h5_grid_shape_from_coordinates(coords):
    grid_w = int(math.floor(float(coords[:, 0].max()))) + 1
    grid_h = int(math.floor(float(coords[:, 1].max()))) + 1
    return max(grid_w, 1), max(grid_h, 1)


def h5_slide_size_from_coordinates(coords, coordinate_mode, patch_scale):
    if coordinate_mode == 'pixel':
        width = int(math.ceil(float(coords[:, 0].max()) + patch_scale))
        height = int(math.ceil(float(coords[:, 1].max()) + patch_scale))
        return width, height

    grid_w, grid_h = h5_grid_shape_from_coordinates(coords)
    return int(grid_w * patch_scale), int(grid_h * patch_scale)


def infer_grid_patch_scale_from_slide_size(coords, slide_size):
    if not slide_size:
        return None
    width, height = slide_size
    grid_w, grid_h = h5_grid_shape_from_coordinates(coords)
    candidates = []
    if grid_w > 0 and width > 0:
        candidates.append(float(width) / grid_w)
    if grid_h > 0 and height > 0:
        candidates.append(float(height) / grid_h)
    candidates = [value for value in candidates if value > 0]
    if not candidates:
        return None
    if len(candidates) > 1:
        small = min(candidates)
        large = max(candidates)
        if small <= 0 or large / small > 1.25:
            return None
    return max(1, int(round(sum(candidates) / len(candidates))))


def lookup_known_slide_size(slide_name, *size_maps):
    for size_map in size_maps:
        value = lookup_by_slide_keys(size_map, slide_name)
        if value is None:
            continue
        if isinstance(value, dict):
            width = value.get('width', value.get('w'))
            height = value.get('height', value.get('h'))
            return int(width), int(height)
        return int(value[0]), int(value[1])
    return None


def h5_coords_to_pixel_coords(coords, coordinate_mode, patch_scale):
    if coordinate_mode == 'pixel':
        return coords
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc
    pixel_coords = np.array(coords, copy=True)
    pixel_coords[:, 0] = pixel_coords[:, 0].astype(float) * patch_scale
    pixel_coords[:, 1] = pixel_coords[:, 1].astype(float) * patch_scale
    return pixel_coords


def read_h5_coordinates(h5_path):
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Reading h5 coordinates requires h5py and numpy') from exc

    with h5py.File(h5_path, 'r') as f:
        coord_key = next((key for key in H5_COORDINATE_KEYS if key in f), None)
        if coord_key is None:
            keys = ', '.join(H5_COORDINATE_KEYS)
            raise KeyError(f'{h5_path} must contain one h5 coordinate key: {keys}')
        coords = np.asarray(f[coord_key])
        if 'features' in f and f['features'].shape[0] != coords.shape[0]:
            raise ValueError(
                f'{h5_path}: features and {coord_key} must have the same first dimension, '
                f'got {f["features"].shape[0]} and {coords.shape[0]}'
            )

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f'{h5_path}: h5 coordinates must have shape (N, >=2), got {coords.shape}')
    return coords


def infer_h5_metadata(h5_files, args, known_size_map=None):
    metadata = {}
    patch_scales = {}
    coordinate_modes = defaultdict(int)
    h5_paths = sorted(set(h5_files.values()))
    total = len(h5_paths)
    progress_interval = PROGRESS_INTERVAL
    print(f'[info] reading h5 metadata: files={total}', flush=True)
    start = time.perf_counter()
    for idx, h5_path in enumerate(h5_paths, start=1):
        slide_name = os.path.splitext(os.path.basename(h5_path))[0]
        coords = read_h5_coordinates(h5_path)
        coordinate_step = infer_patch_scale_from_coordinates(coords)
        coordinate_mode = infer_h5_coordinate_mode(coords, requested_mode=args.h5_coordinate_mode)
        known_size = lookup_known_slide_size(slide_name, known_size_map)

        if args.patch_scale > 0:
            patch_scale = args.patch_scale
        elif coordinate_mode == 'pixel' and coordinate_step is not None:
            patch_scale = coordinate_step
        elif coordinate_mode == 'grid':
            patch_scale = infer_grid_patch_scale_from_slide_size(coords, known_size)
            if patch_scale is None:
                patch_scale = DEFAULT_PATCH_SCALE
                print(
                    f'[warning] {h5_path}: h5 coordinates look like patch-grid indices '
                    f'(step={coordinate_step}); slide size is unavailable or does not match the grid. '
                    f'Falling back to --patch-scale={patch_scale}. Pass --patch-scale or --size-json/--wsi-dir to override.'
                )
        else:
            patch_scale = DEFAULT_PATCH_SCALE
            print(
                f'[warning] Could not infer h5 patch scale from {h5_path}; '
                f'falling back to {patch_scale}. Pass --patch-scale to override.'
            )

        if known_size is not None:
            width, height = known_size
        else:
            width, height = h5_slide_size_from_coordinates(coords, coordinate_mode, patch_scale)

        info = {
            'path': h5_path,
            'coordinate_mode': coordinate_mode,
            'patch_scale': int(patch_scale),
            'width': width,
            'height': height,
            'coordinate_step': coordinate_step,
        }
        for key in slide_lookup_keys(slide_name):
            metadata[key] = info
        patch_scales[int(patch_scale)] = patch_scales.get(int(patch_scale), 0) + 1
        coordinate_modes[coordinate_mode] += 1
        if progress_interval > 0 and (idx == 1 or idx % progress_interval == 0 or idx == total):
            elapsed = time.perf_counter() - start
            print(
                f'[info] h5 metadata progress {idx}/{total}: {slide_name} '
                f'mode={coordinate_mode} step={coordinate_step} patch_scale={patch_scale} '
                f'patches={coords.shape[0]} elapsed={elapsed:.1f}s',
                flush=True,
            )
        del coords

    if coordinate_modes:
        mode_text = ', '.join(f'{mode}:{count}' for mode, count in sorted(coordinate_modes.items()))
        elapsed = time.perf_counter() - start
        print(f'[info] h5 coordinate modes: {mode_text}', flush=True)
        print(f'[info] h5 metadata complete files={total} elapsed={elapsed:.1f}s', flush=True)
    return metadata, patch_scales


def choose_patch_scale(args, h5_metadata, patch_scales):
    if args.patch_scale > 0:
        if patch_scales:
            inferred = max(patch_scales, key=patch_scales.get)
            if inferred != args.patch_scale:
                print(
                    f'[warning] Explicit --patch-scale={args.patch_scale}, but the most common h5 '
                    f'patch scale estimate is {inferred}. Using the explicit value.'
                )
        return args.patch_scale

    if patch_scales:
        patch_scale = max(patch_scales, key=patch_scales.get)
        print(f'[info] Using h5 patch scale: {patch_scale}')
        return patch_scale

    if h5_metadata:
        patch_scale = next(iter(h5_metadata.values()))['patch_scale']
        print(f'[info] Using fallback patch scale from h5 metadata: {patch_scale}')
        return patch_scale

    print(f'[info] No --h5-dir metadata available; using default patch scale {DEFAULT_PATCH_SCALE}.')
    return DEFAULT_PATCH_SCALE


def h5_size_map_from_metadata(h5_metadata):
    out = {}
    for slide_name, info in h5_metadata.items():
        size = (int(info['width']), int(info['height']))
        for key in slide_lookup_keys(slide_name):
            out[key] = size
    return out


def h5_size_json_from_metadata(h5_metadata):
    out = {}
    for info in h5_metadata.values():
        h5_path = info.get('path', '')
        if not h5_path:
            continue
        slide_name = strip_known_slide_suffixes(os.path.basename(h5_path))
        out[slide_name] = {
            'width': int(info['width']),
            'height': int(info['height']),
        }
    return out


def _compute_patch_labels_chunked(pixel_coords, regions, patch_scale, mid_scale, full_mid_h, full_mid_w):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc
    try:
        import cv2
    except ImportError:
        cv2 = None

    n_patches = pixel_coords.shape[0]
    labels = np.zeros(n_patches, dtype=np.uint8)

    # Pre-compute mid-scale coordinates for all patches so we can filter per chunk.
    mx0 = np.floor(pixel_coords[:, 0] / mid_scale).astype(np.int32)
    my0 = np.floor(pixel_coords[:, 1] / mid_scale).astype(np.int32)
    mx1 = np.ceil((pixel_coords[:, 0] + patch_scale) / mid_scale).astype(np.int32)
    my1 = np.ceil((pixel_coords[:, 1] + patch_scale) / mid_scale).astype(np.int32)
    np.clip(mx0, 0, full_mid_w, out=mx0)
    np.clip(my0, 0, full_mid_h, out=my0)
    np.clip(mx1, 0, full_mid_w, out=mx1)
    np.clip(my1, 0, full_mid_h, out=my1)

    prepared_regions = []
    for region in regions:
        pts = np.asarray(region['points'], dtype=np.float64)
        if pts.shape[0] < 3:
            continue
        contour = np.floor(pts / mid_scale + 0.5).astype(np.int32)
        prepared_regions.append((float(pts[:, 1].min()), float(pts[:, 1].max()), contour))

    if not prepared_regions:
        return labels

    # Adaptive chunk height: target ~50 MB per chunk (uint8 row = full_mid_w bytes).
    target_chunk_bytes = 50 * 1024 * 1024
    chunk_size = max(512, min(full_mid_h, max(1, target_chunk_bytes // max(1, full_mid_w))))

    for chunk_y0 in range(0, full_mid_h, chunk_size):
        chunk_y1 = min(full_mid_h, chunk_y0 + chunk_size)
        chunk_h = chunk_y1 - chunk_y0
        chunk = np.zeros((chunk_h, full_mid_w), dtype=np.uint8)
        chunk_has_regions = False

        # Rasterize annotations that overlap this vertical slab
        for py_min, py_max, contour in prepared_regions:
            if int(math.ceil(py_max / mid_scale)) <= chunk_y0 or int(py_min / mid_scale) >= chunk_y1:
                continue

            chunk_contour = contour.copy()
            chunk_contour[:, 1] -= chunk_y0
            if cv2 is not None:
                cv2.fillPoly(chunk, [chunk_contour], 1)
            else:
                _fill_polygon_numpy_local(chunk, chunk_contour)
            chunk_has_regions = True

        if not chunk_has_regions:
            continue

        # Use an integral image so all patch-rectangle hit tests in this chunk are vectorized.
        overlapping = np.where(
            (my1 > chunk_y0) &
            (my0 < chunk_y1) &
            (mx1 > mx0) &
            (labels == 0)
        )[0]
        if overlapping.size == 0:
            continue

        cy0 = np.maximum(0, my0[overlapping] - chunk_y0)
        cy1 = np.minimum(chunk_h, my1[overlapping] - chunk_y0)
        cx0 = mx0[overlapping]
        cx1 = mx1[overlapping]
        valid = (cy1 > cy0) & (cx1 > cx0)
        if not np.any(valid):
            continue

        overlapping = overlapping[valid]
        cy0 = cy0[valid]
        cy1 = cy1[valid]
        cx0 = cx0[valid]
        cx1 = cx1[valid]

        integral = np.zeros((chunk_h + 1, full_mid_w + 1), dtype=np.uint32)
        integral[1:, 1:] = chunk
        np.cumsum(integral, axis=0, dtype=np.uint32, out=integral)
        np.cumsum(integral, axis=1, dtype=np.uint32, out=integral)

        hits = (
            integral[cy1, cx1].astype(np.int64) -
            integral[cy0, cx1].astype(np.int64) -
            integral[cy1, cx0].astype(np.int64) +
            integral[cy0, cx0].astype(np.int64)
        ) > 0
        labels[overlapping[hits]] = 1

    return labels


def _fill_polygon_numpy_local(mask, contour):
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


def h5_relative_canvas_from_coords(coords, oversample=4, origin='zero'):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] < 2:
        raise ValueError(f'h5 coordinates must have shape (N, >=2), got {coords.shape}')

    oversample = max(1, int(oversample))
    x_step = infer_axis_step(coords[:, 0]) or 1
    y_step = infer_axis_step(coords[:, 1]) or 1
    if x_step <= 0 or y_step <= 0:
        raise ValueError(f'cannot infer positive h5 coordinate step: x={x_step}, y={y_step}')

    min_x = float(coords[:, 0].min())
    min_y = float(coords[:, 1].min())
    if origin == 'bbox':
        origin_x, origin_y = min_x, min_y
    else:
        origin_x, origin_y = min(0.0, min_x), min(0.0, min_y)

    grid_w = (float(coords[:, 0].max()) - origin_x) / float(x_step) + 1.0
    grid_h = (float(coords[:, 1].max()) - origin_y) / float(y_step) + 1.0
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError(f'invalid relative h5 grid size: {grid_w}x{grid_h}')

    synthetic_coords = np.empty((coords.shape[0], 2), dtype=np.float64)
    synthetic_coords[:, 0] = ((coords[:, 0] - origin_x) / float(x_step)) * oversample
    synthetic_coords[:, 1] = ((coords[:, 1] - origin_y) / float(y_step)) * oversample
    canvas_w = grid_w * oversample
    canvas_h = grid_h * oversample
    full_mid_w = max(1, int(math.ceil(canvas_w)))
    full_mid_h = max(1, int(math.ceil(canvas_h)))
    meta = {
        'x_step': int(x_step),
        'y_step': int(y_step),
        'origin_x': origin_x,
        'origin_y': origin_y,
        'grid_w': grid_w,
        'grid_h': grid_h,
        'canvas_w': canvas_w,
        'canvas_h': canvas_h,
        'oversample': oversample,
    }
    return synthetic_coords, oversample, full_mid_h, full_mid_w, meta


def regions_to_relative_canvas(regions, canvas_w, canvas_h):
    out = []
    for region in regions:
        rel_points = region.get('relative_points')
        if rel_points is None:
            width, height = region['size']
            rel_points = [(x / width, y / height) for x, y in region['points']]

        converted = dict(region)
        converted['points'] = [
            (
                min(1.0, max(0.0, float(x))) * canvas_w,
                min(1.0, max(0.0, float(y))) * canvas_h,
            )
            for x, y in rel_points
        ]
        out.append(converted)
    return out


def labels_for_h5_relative_coordinates(
    regions,
    h5_path,
    multi_label=False,
    class_num=0,
    binary_labels=True,
    coords=None,
    oversample=4,
    origin='zero',
):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc

    if coords is None:
        coords = read_h5_coordinates(h5_path)
    synthetic_coords, synthetic_patch_scale, full_mid_h, full_mid_w, meta = h5_relative_canvas_from_coords(
        coords,
        oversample=oversample,
        origin=origin,
    )
    relative_regions = regions_to_relative_canvas(regions, meta['canvas_w'], meta['canvas_h'])

    def _compute(regs):
        return _compute_patch_labels_chunked(
            synthetic_coords,
            regs,
            synthetic_patch_scale,
            1,
            full_mid_h,
            full_mid_w,
        )

    if multi_label:
        if class_num <= 0:
            raise ValueError('--multi-label h5 label writing requires at least one class')
        labels = np.zeros((coords.shape[0], class_num), dtype=np.uint8)
        for cls in range(1, class_num + 1):
            cls_regions = [r for r in relative_regions if int(r['label_id']) == cls]
            if not cls_regions:
                continue
            labels[:, cls - 1] = _compute(cls_regions)
        return labels, meta

    if binary_labels:
        return _compute(relative_regions), meta

    labels = np.zeros(coords.shape[0], dtype=np.uint16)
    for cls in sorted({int(r['label_id']) for r in relative_regions}):
        cls_regions = [r for r in relative_regions if int(r['label_id']) == cls]
        cls_hits = _compute(cls_regions)
        labels[cls_hits == 1] = cls
    return labels, meta


def labels_for_h5_coordinates(
    regions,
    h5_path,
    patch_scale,
    coordinate_mode='pixel',
    multi_label=False,
    class_num=0,
    binary_labels=True,
    coords=None,
):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Writing h5 patch labels requires numpy') from exc

    if coords is None:
        coords = read_h5_coordinates(h5_path)
    pixel_coords = h5_coords_to_pixel_coords(coords, coordinate_mode, patch_scale)

    width, height = regions[0]['size']
    mid_scale = max(1, patch_scale // 32)
    while patch_scale % mid_scale != 0:
        mid_scale -= 1
    full_mid_h = int(math.ceil(height / mid_scale))
    full_mid_w = int(math.ceil(width / mid_scale))

    def _compute(regs):
        return _compute_patch_labels_chunked(
            pixel_coords, regs, patch_scale, mid_scale, full_mid_h, full_mid_w)

    if multi_label:
        if class_num <= 0:
            raise ValueError('--multi-label h5 label writing requires at least one class')
        labels = np.zeros((coords.shape[0], class_num), dtype=np.uint8)
        for cls in range(1, class_num + 1):
            cls_regions = [r for r in regions if int(r['label_id']) == cls]
            if not cls_regions:
                continue
            labels[:, cls - 1] = _compute(cls_regions)
        return labels

    if binary_labels:
        return _compute(regions)

    labels = np.zeros(coords.shape[0], dtype=np.uint16)
    for cls in sorted({int(r['label_id']) for r in regions}):
        cls_regions = [r for r in regions if int(r['label_id']) == cls]
        cls_hits = _compute(cls_regions)
        labels[cls_hits == 1] = cls
    return labels


def count_positive_h5_patches(labels):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Counting h5 patch labels requires numpy') from exc

    labels = np.asarray(labels)
    if labels.ndim == 2:
        return int((labels > 0).any(axis=1).sum())
    return int((labels > 0).sum())


def write_h5_label_files(
    regions_by_slide,
    h5_dir,
    h5_label_out,
    patch_scale,
    multi_label=False,
    class_num=0,
    binary_labels=True,
    h5_metadata=None,
    h5_files=None,
    label_coordinate_space='relative',
    relative_oversample=4,
    relative_origin='zero',
):
    if not h5_label_out:
        return {}, 0
    if not h5_dir:
        raise ValueError('--h5-dir is required when --h5-label-out is set')

    h5_files = h5_files or find_h5_files(h5_dir)
    if not h5_files:
        raise ValueError(f'No .h5/.hdf5 files found in {h5_dir}')

    os.makedirs(h5_label_out, exist_ok=True)
    pos_counts = {}
    written = 0
    missing = []
    slide_items = sorted(regions_by_slide.items())
    total = len(slide_items)
    for idx, (slide_name, regions) in enumerate(slide_items, start=1):
        h5_path = lookup_by_slide_keys(h5_files, slide_name)
        if h5_path is None:
            missing.append(slide_name)
            continue
        slide_h5_metadata = lookup_by_slide_keys(h5_metadata or {}, slide_name) or {}
        slide_patch_scale = int(slide_h5_metadata.get('patch_scale', patch_scale))
        slide_coordinate_mode = slide_h5_metadata.get('coordinate_mode', 'pixel')
        label_ids = sorted({int(r['label_id']) for r in regions})
        print(
            f'[info] writing h5 labels {idx}/{total}: {slide_name} '
            f'coordinate_space={label_coordinate_space} mode={slide_coordinate_mode} patch_scale={slide_patch_scale} '
            f'regions={len(regions)} labels={label_ids}',
            flush=True,
        )
        start = time.perf_counter()
        if label_coordinate_space == 'relative':
            coords = read_h5_coordinates(h5_path)
            labels, relative_meta = labels_for_h5_relative_coordinates(
                regions,
                h5_path,
                multi_label=multi_label,
                class_num=class_num,
                binary_labels=binary_labels,
                coords=coords,
                oversample=relative_oversample,
                origin=relative_origin,
            )
            print(
                f'[info] relative h5 layout {slide_name}: '
                f'x_step={relative_meta["x_step"]} y_step={relative_meta["y_step"]} '
                f'grid={relative_meta["grid_w"]:.1f}x{relative_meta["grid_h"]:.1f} '
                f'origin={relative_origin} oversample={relative_meta["oversample"]}',
                flush=True,
            )
        else:
            labels = labels_for_h5_coordinates(
                regions,
                h5_path,
                slide_patch_scale,
                coordinate_mode=slide_coordinate_mode,
                multi_label=multi_label,
                class_num=class_num,
                binary_labels=binary_labels,
                coords=None,
            )
        out_path = os.path.join(h5_label_out, slide_name + '.npy')
        np_save(out_path, labels)
        pos_counts[slide_name] = count_positive_h5_patches(labels)
        written += 1
        del labels
        elapsed = time.perf_counter() - start
        print(
            f'[info] wrote h5 labels {idx}/{total}: {slide_name} '
            f'positive={pos_counts[slide_name]} elapsed={elapsed:.2f}s -> {out_path}',
            flush=True,
        )
        gc.collect()

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
    parser.add_argument('--csv', default='', help='CSV with wsi_path, labels, coordinates columns')
    parser.add_argument('--label-map', default='', help='optional JSON mapping label names to integer ids')
    parser.add_argument('--auto-label-map', action='store_true',
        help='assign missing labels ids in first-seen CSV order, starting at max(existing ids)+1')
    parser.add_argument('--label-map-out', default='',
        help='optional path to write the final label map, useful with --auto-label-map')
    parser.add_argument('--xml-out', default='', help='directory to write PRET-compatible XML files')
    parser.add_argument('--mask-out', default='', help='optional directory to write PRET patch-grid PNG masks')
    parser.add_argument('--h5-dir', default='', help='optional directory containing per-slide h5 feature files')
    parser.add_argument('--h5-recursive', dest='h5_recursive', action='store_true', default=True,
        help='search --h5-dir recursively; enabled by default')
    parser.add_argument('--no-h5-recursive', dest='h5_recursive', action='store_false',
        help='only search h5 files directly under --h5-dir')
    parser.add_argument('--h5-label-out', default='',
        help='optional directory to write per-h5 patch label .npy arrays aligned to h5 feature order')
    parser.add_argument('--skip-missing-h5', dest='skip_missing_h5', action='store_true', default=None,
        help='skip CSV rows whose slide has no matching h5 file; defaults to on when --h5-label-out is used')
    parser.add_argument('--no-skip-missing-h5', dest='skip_missing_h5', action='store_false',
        help='raise instead of skipping when annotated slides have no matching h5 file')
    parser.add_argument('--data-info-out', default='', help='optional data_info JSON for annotated slides')
    parser.add_argument('--size-json', default='',
        help='optional slide size JSON for formats OpenSlide cannot read; h5 coordinates can infer this when --h5-dir is set')
    parser.add_argument('--size-json-out', default='',
        help='optional path to write slide sizes read from --wsi-dir and/or inferred from --h5-dir')
    parser.add_argument('--write-size-json-only', action='store_true',
        help='only scan --wsi-dir and write --size-json-out; no CSV conversion is performed')
    parser.add_argument('--wsi-dir', default='',
        help='optional directory containing WSI files, including .sdpc, used to read slide sizes')
    parser.add_argument('--wsi-recursive', action='store_true',
        help='search --wsi-dir recursively')
    parser.add_argument('--strict-wsi-size-errors', action='store_true',
        help='raise if any --wsi-dir slide size cannot be read; by default unreadable WSI files are skipped during h5 label conversion')
    parser.add_argument('--wsi-extensions', nargs='*', default=list(DEFAULT_WSI_EXTENSIONS),
        help='WSI file extensions to scan under --wsi-dir; default includes sdpc, svs, tif, tiff, mrxs, ndpi, scn')
    parser.add_argument('--slide-reader', default='auto', choices=['auto', 'openslide', 'opensdpc'],
        help='slide reader used for WSI sizes; auto tries OpenSlide first, then opensdpc for sdpc-like files')
    parser.add_argument('--h5-coordinate-mode', default='grid', choices=['pixel', 'grid'],
        help='legacy absolute h5 label mode and evaluation visualization coordinate interpretation: pixel=level-0 top-left pixels, grid=patch indices')
    parser.add_argument('--h5-label-coordinate-space', default='relative', choices=['relative', 'absolute'],
        help='how to align CSV annotations to h5 patches when writing --h5-label-out; relative uses normalized CSV coordinates and the h5 patch layout, absolute uses legacy grid/pixel + patch-scale coordinates')
    parser.add_argument('--h5-relative-origin', default='zero', choices=['zero', 'bbox'],
        help='origin for relative h5 label alignment: zero preserves coordinate origin 0, bbox normalizes the observed h5 patch bounding box to [0,1]')
    parser.add_argument('--h5-relative-oversample', type=int, default=4,
        help='sub-patch rasterization factor for relative h5 label alignment; larger is more boundary-sensitive but slower')
    parser.add_argument('--patch-scale', type=int, default=0,
        help='level-0 pixels per patch; 0 infers from h5 pixel coordinates or WSI size for h5 grid coordinates, otherwise uses 512')
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
        help='do not try OpenSlide; if --slide-reader=auto, opensdpc may still be used for .sdpc files')
    parser.add_argument('--no-clip', action='store_true',
        help='raise on normalized coordinates outside [0,1] instead of clipping')
    args = parser.parse_args()
    if args.write_size_json_only:
        if not args.wsi_dir:
            parser.error('--write-size-json-only requires --wsi-dir')
        if not args.size_json_out:
            parser.error('--write-size-json-only requires --size-json-out')
        return args
    if not args.csv:
        parser.error('--csv is required unless --write-size-json-only is used')
    if not args.xml_out and not args.mask_out and not args.h5_label_out and not args.data_info_out and not args.size_json_out:
        parser.error(
            'provide at least one of --xml-out, --mask-out, --h5-label-out, '
            '--data-info-out, or --size-json-out'
        )
    if not args.label_map and not args.auto_label_map:
        parser.error('provide --label-map, or use --auto-label-map to create one from CSV labels')
    if args.h5_label_out and not args.h5_dir:
        parser.error('--h5-dir is required when --h5-label-out is set')
    if args.skip_missing_h5 is None:
        args.skip_missing_h5 = bool(args.h5_label_out)
    if args.patch_scale < 0:
        parser.error('--patch-scale must be >= 0')
    if args.h5_relative_oversample <= 0:
        parser.error('--h5-relative-oversample must be > 0')
    return args


def main():
    args = parse_args()
    print('[info] csv_to_pret_annotations started', flush=True)
    slide_readers = requested_slide_readers(args)

    wsi_size_map = {}
    wsi_size_json = {}
    if args.wsi_dir:
        wsi_size_map, wsi_size_json = build_wsi_size_maps(
            args.wsi_dir,
            extensions=args.wsi_extensions,
            recursive=args.wsi_recursive,
            readers=slide_readers,
            name_mode=args.name_mode,
            strict=args.write_size_json_only or args.strict_wsi_size_errors,
        )

    if args.write_size_json_only:
        write_size_json(wsi_size_json, args.size_json_out)
        print(f'wsi files scanned: {len(wsi_size_json)}')
        print(f'size_json written: {args.size_json_out}')
        return

    print('[info] loading label map and size map', flush=True)
    label_map = load_label_map(args.label_map)
    if not args.include_zero_labels and 0 in set(label_map.values()):
        print(
            '[warning] label map contains id 0, but --include-zero-labels is not set; '
            'rows mapped to 0 will be skipped. Use 1-based class ids for PRET class labels.',
            flush=True,
        )
    size_map = load_size_map(args.size_json)
    h5_files = find_h5_files(args.h5_dir, recursive=args.h5_recursive)
    if args.h5_label_out and not h5_files:
        raise SystemExit(
            f'No .h5/.hdf5 files found in --h5-dir={args.h5_dir}. '
            'Check the path, or use --h5-recursive for nested h5 files.'
        )
    known_size_map = {}
    known_size_map.update(wsi_size_map)
    known_size_map.update(size_map)
    h5_metadata, patch_scales = infer_h5_metadata(h5_files, args, known_size_map) if h5_files else ({}, {})
    gc.collect()
    args.patch_scale = choose_patch_scale(args, h5_metadata, patch_scales)
    h5_size_map = h5_size_map_from_metadata(h5_metadata)
    if args.size_json_out:
        combined_size_json = {}
        combined_size_json.update(h5_size_json_from_metadata(h5_metadata))
        combined_size_json.update(wsi_size_json)
        combined_size_json.update({key: {'width': value[0], 'height': value[1]} for key, value in size_map.items()})
        write_size_json(combined_size_json, args.size_json_out)

    regions_by_slide, skipped = read_regions(
        args,
        label_map,
        size_map,
        h5_size_map,
        wsi_size_map,
        slide_readers=[] if args.h5_label_out else slide_readers,
        h5_files=h5_files,
    )
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
        h5_metadata=h5_metadata,
        h5_files=h5_files,
        label_coordinate_space=args.h5_label_coordinate_space,
        relative_oversample=args.h5_relative_oversample,
        relative_origin=args.h5_relative_origin,
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
    if args.size_json_out:
        print(f'size_json written: {args.size_json_out}')
    if args.data_info_out:
        print(f'data_info written: {args.data_info_out}')
    if skipped:
        skipped_text = ', '.join(f'{label}:{count}' for label, count in sorted(skipped.items()))
        print(f'skipped labels: {skipped_text}')


if __name__ == '__main__':
    main()
