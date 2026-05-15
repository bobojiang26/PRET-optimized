import argparse
import os

from csv_to_pret_annotations import (
    DEFAULT_WSI_EXTENSIONS,
    build_wsi_size_maps,
    normalize_extensions,
    write_size_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Scan a WSI directory, read level-0 dimensions, and write a PRET size JSON file.'
    )
    parser.add_argument('--wsi-dir', required=True, help='directory containing WSI files, for example .sdpc files')
    parser.add_argument('--out', required=True, help='output size JSON path')
    parser.add_argument('--recursive', action='store_true', help='search --wsi-dir recursively')
    parser.add_argument('--extensions', nargs='*', default=list(DEFAULT_WSI_EXTENSIONS),
        help='extensions to scan; default includes sdpc, svs, tif, tiff, mrxs, ndpi, scn')
    parser.add_argument('--slide-reader', default='auto', choices=['auto', 'openslide', 'opensdpc'],
        help='slide reader to use; auto tries OpenSlide first, then opensdpc')
    parser.add_argument('--name-mode', default='stem', choices=['stem', 'basename'],
        help='JSON key from WSI stem or full basename')
    args = parser.parse_args()
    if not os.path.isdir(args.wsi_dir):
        parser.error(f'--wsi-dir does not exist or is not a directory: {args.wsi_dir}')
    return args


def requested_readers(slide_reader):
    if slide_reader == 'auto':
        return ('openslide', 'opensdpc')
    return (slide_reader,)


def main():
    args = parse_args()
    _, size_json = build_wsi_size_maps(
        args.wsi_dir,
        extensions=normalize_extensions(args.extensions),
        recursive=args.recursive,
        readers=requested_readers(args.slide_reader),
        name_mode=args.name_mode,
    )
    write_size_json(size_json, args.out)
    print(f'wsi files scanned: {len(size_json)}')
    print(f'size_json written: {args.out}')


if __name__ == '__main__':
    main()
