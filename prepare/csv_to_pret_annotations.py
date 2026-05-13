import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


NUMBER_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
PRET_XML_IDS = {
    'mask': 1,
    'box': 2,
    'roughMask': 3,
}


def load_label_map(path):
    with open(path, 'r', encoding='utf8') as f:
        raw = json.load(f)
    out = {}
    for label, value in raw.items():
        out[str(label)] = int(value)
    return out


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


def slide_name_from_path(wsi_path, name_mode):
    base = os.path.basename(wsi_path)
    if name_mode == 'basename':
        return base
    return os.path.splitext(base)[0]


def resolve_wsi_path(raw_path, csv_path):
    raw_path = raw_path.strip()
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(os.path.dirname(csv_path), raw_path))


def lookup_slide_size(wsi_path, slide_name, size_map, use_openslide=True):
    candidates = [
        wsi_path,
        os.path.abspath(wsi_path),
        os.path.basename(wsi_path),
        slide_name,
    ]
    for key in candidates:
        if key in size_map:
            return size_map[key]

    if not use_openslide:
        raise ValueError(
            f'No size entry found for {wsi_path}. Add it to --size-json or allow OpenSlide.'
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


def label_to_id(label, label_map):
    label = str(label).strip()
    if label in label_map:
        return label_map[label]
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


def read_regions(args, label_map, size_map):
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
            label = str(row[field_lookup['labels']]).strip()
            coords = row[field_lookup['coordinates']]
            label_id = label_to_id(label, label_map)
            if not should_keep_label(label, label_id, positive_labels, args.include_zero_labels):
                skipped[label] += 1
                continue

            wsi_path = resolve_wsi_path(raw_wsi_path, csv_path)
            slide_name = slide_name_from_path(wsi_path, args.name_mode)
            width, height = lookup_slide_size(
                wsi_path, slide_name, size_map, use_openslide=not args.no_openslide
            )
            points = parse_points(coords, width, height, clip=not args.no_clip)
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


def write_data_info(regions_by_slide, data_info_out, mask_out, pos_counts, wsi_label_mode):
    if not data_info_out:
        return
    out = {}
    for slide_name, regions in sorted(regions_by_slide.items()):
        label_ids = sorted({int(r['label_id']) for r in regions})
        if wsi_label_mode == 'binary':
            wsi_label = 1
        elif wsi_label_mode == 'single-label':
            if len(label_ids) != 1:
                raise ValueError(f'{slide_name} has multiple labels {label_ids}; cannot use single-label mode')
            wsi_label = label_ids[0]
        else:
            wsi_label = max(label_ids)

        item = {
            'wsi_label': int(wsi_label),
            'fixed_test_set': False,
        }
        if mask_out:
            item['patch_labels'] = os.path.join(mask_out, slide_name + '.png')
            item['pos_patch_num'] = int(pos_counts.get(slide_name, 0))
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
    parser.add_argument('--label-map', required=True, help='JSON mapping label names to integer ids')
    parser.add_argument('--xml-out', default='', help='directory to write PRET-compatible XML files')
    parser.add_argument('--mask-out', default='', help='optional directory to write PRET patch-grid PNG masks')
    parser.add_argument('--data-info-out', default='', help='optional data_info JSON for annotated slides')
    parser.add_argument('--size-json', default='', help='optional slide size JSON for formats OpenSlide cannot read')
    parser.add_argument('--patch-scale', type=int, default=512, help='level-0 pixels per PRET patch; usually 512')
    parser.add_argument('--prompt-type', default='mask',
        choices=['mask', 'box', 'roughMask', 'label-id'],
        help='XML Annotation Id convention: mask=1, box=2, roughMask=3, label-id preserves the label map id')
    parser.add_argument('--positive-labels', nargs='*',
        help='optional label names or ids to keep; by default all non-zero label ids are kept')
    parser.add_argument('--include-zero-labels', action='store_true',
        help='keep rows whose mapped label id is 0')
    parser.add_argument('--name-mode', default='stem', choices=['stem', 'basename'],
        help='output XML/PNG slide key from WSI stem or full basename')
    parser.add_argument('--wsi-label-mode', default='binary', choices=['binary', 'single-label', 'max-label'],
        help='wsi_label policy when --data-info-out is used')
    parser.add_argument('--no-openslide', action='store_true',
        help='do not try to read WSI dimensions with OpenSlide; require --size-json')
    parser.add_argument('--no-clip', action='store_true',
        help='raise on normalized coordinates outside [0,1] instead of clipping')
    args = parser.parse_args()
    if not args.xml_out and not args.mask_out and not args.data_info_out:
        parser.error('provide at least one of --xml-out, --mask-out, or --data-info-out')
    if args.patch_scale <= 0:
        parser.error('--patch-scale must be positive')
    return args


def main():
    args = parse_args()
    label_map = load_label_map(args.label_map)
    size_map = load_size_map(args.size_json)
    regions_by_slide, skipped = read_regions(args, label_map, size_map)
    if not regions_by_slide:
        raise SystemExit('No regions were converted. Check labels, label map, and --positive-labels.')

    xml_count = write_xml_files(regions_by_slide, args.xml_out)
    pos_counts, mask_count = write_mask_files(regions_by_slide, args.mask_out, args.patch_scale)
    write_data_info(regions_by_slide, args.data_info_out, args.mask_out, pos_counts, args.wsi_label_mode)

    region_count = sum(len(v) for v in regions_by_slide.values())
    print(f'converted slides: {len(regions_by_slide)}')
    print(f'converted regions: {region_count}')
    if xml_count:
        print(f'xml files written: {xml_count} -> {args.xml_out}')
    if mask_count:
        print(f'mask png files written: {mask_count} -> {args.mask_out}')
    if args.data_info_out:
        print(f'data_info written: {args.data_info_out}')
    if skipped:
        skipped_text = ', '.join(f'{label}:{count}' for label, count in sorted(skipped.items()))
        print(f'skipped labels: {skipped_text}')


if __name__ == '__main__':
    main()
