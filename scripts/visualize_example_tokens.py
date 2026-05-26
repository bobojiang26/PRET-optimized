#!/usr/bin/env python
import argparse
import csv
import inspect
import json
import os
import random
import sys
from html import escape

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CORE = os.path.join(ROOT, 'core')
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from main import (  # noqa: E402
    align_numpy_feature_dim,
    apply_context_feature_centering,
    dataset_is_multilabel,
    get_min_feature_dim,
    has_wsi_label,
    label_counts,
    load_dataset_info,
    load_weak_prompts,
    patch_labels_for_class,
    reference_mask_for_task,
    select_example_names,
    sparsify_reference_tokens,
)
from modules import execute_subtyping_tagger, execute_tagger  # noqa: E402

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


TOKEN_LABEL_NAMES = {
    -1: 'unknown',
    0: 'other_class',
    1: 'target_class',
    254: 'uncertain',
    255: 'background',
}

TOKEN_LABEL_COLORS = {
    -1: '#9ca3af',
    0: '#2563eb',
    1: '#dc2626',
    254: '#f59e0b',
    255: '#16a34a',
}

CLASS_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
    '#3182bd', '#31a354', '#756bb1', '#636363', '#e6550d',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize PRET example/reference token pools with t-SNE.'
    )
    parser.add_argument('--raw_feature_path', default='', help='h5 folder/path; used only to fill h5 dataset_info entries when needed')
    parser.add_argument('--wsi_path', required=True)
    parser.add_argument('--dump_features', required=True)
    parser.add_argument('--dataset_info', required=True)
    parser.add_argument('--prompt_type', default='slideLabel')
    parser.add_argument('--prompt_path', default='')
    parser.add_argument('--class_num', '--c', dest='c', type=int, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='target class ids to visualize')
    parser.add_argument('--all_classes', action='store_true', help='visualize class ids 1..class_num')
    parser.add_argument('--plot_mode', default='combined', choices=['combined', 'per_class', 'both'])
    parser.add_argument('--out_dir', default='records/example_token_vis')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--repeat', type=int, default=0, help='repeat index; repeated shuffles before selecting examples')
    parser.add_argument('--example_num', type=int, default=4)
    parser.add_argument('--example_ratio', type=float, default=0.0)
    parser.add_argument('--example_ratio_max_per_class', type=int, default=0)
    parser.add_argument('--require_label', action='store_true')
    parser.add_argument('--multilabel', action='store_true')
    parser.add_argument('--patch_scale', type=int, default=512)
    parser.add_argument('--topk', type=int, default=40)
    parser.add_argument('--ignore', type=float, default=0.0)
    parser.add_argument('--context_centering', default='none', choices=['none', 'example'])
    parser.add_argument('--reference_token_budget', type=int, default=0)
    parser.add_argument('--reference_sparsify_strategy', default='auto', choices=['auto', 'quality', 'legacy', 'hierarchical'])
    parser.add_argument('--reference_anchor_ratio', type=float, default=0.25)
    parser.add_argument('--reference_random_ratio', type=float, default=0.1)
    parser.add_argument('--max_tokens_per_class', type=int, default=10000, help='max tokens sampled for t-SNE plotting per class; 0 keeps all')
    parser.add_argument('--max_tokens_total', type=int, default=30000, help='max tokens in the combined t-SNE plot; 0 keeps all')
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--tsne_iter', type=int, default=1000)
    args = parser.parse_args()
    if args.all_classes:
        args.classes = list(range(1, args.c + 1)) if args.c > 1 else [1]
    elif not args.classes:
        parser.error('provide --classes or use --all_classes')
    return args


def load_labeled_names(dataset_info, args):
    labeled_names = []
    for n in dataset_info:
        if dataset_info[n].get('fixed_test_set', False):
            continue
        if 'pos_patch_num' in dataset_info[n]:
            pn = dataset_info[n]['pos_patch_num']
            if args.c == 1 and 'CAMELYON' in args.wsi_path:
                if pn >= 1000 and pn < 3000:
                    labeled_names.append(n)
            else:
                labeled_names.append(n)

        if args.prompt_type == 'slideLabel':
            if args.c > 1 and n not in labeled_names:
                labeled_names.append(n)
            if args.c == 1 and (
                dataset_info[n].get('wsi_label') == 0 or
                (dataset_info[n].get('h5_input', False) and 'pos_patch_num' not in dataset_info[n])
            ):
                labeled_names.append(n)
    return labeled_names


def select_examples(dataset_info, args):
    labeled_names = load_labeled_names(dataset_info, args)
    selected = None
    target_counts = None
    seen_examples = []
    for repeat_idx in range(args.repeat + 1):
        attempts = 0
        while True:
            attempts += 1
            random.shuffle(labeled_names)
            selected, target_counts = select_example_names(labeled_names, dataset_info, args)
            selected = sorted(selected)
            if selected not in seen_examples or attempts >= 1000:
                if selected in seen_examples and attempts >= 1000:
                    print(
                        f'[split] repeat={repeat_idx} could not find a new unique example set '
                        'after 1000 attempts; reusing a previous set.'
                    )
                seen_examples.append(selected)
                break
    return selected, target_counts, labeled_names


def slide_wsi_label(dataset_info, slide_features, slide_name, cls, multilabel):
    if multilabel and slide_name in dataset_info:
        return 1 if has_wsi_label(dataset_info[slide_name], cls) else 0
    if slide_name in dataset_info:
        return int(dataset_info[slide_name].get('wsi_label', slide_features.get('wsi_label', 0)))
    return int(slide_features.get('wsi_label', 0))


def class_patch_labels(raw_pl, example_wsi_label, cls, class_num, multilabel):
    raw_pl = np.asarray(raw_pl)
    if class_num > 1:
        if multilabel:
            return patch_labels_for_class(raw_pl, cls, class_num, multilabel=True)
        if raw_pl.ndim == 1 and set(np.unique(raw_pl).tolist()).issubset({0, 1}):
            pl = raw_pl.astype(np.int64, copy=True)
            pl[pl == 0] = 255
            if example_wsi_label != cls:
                pl[pl == 1] = 0
            else:
                pl[pl == 1] = 1
            return pl
        return patch_labels_for_class(raw_pl, cls, class_num, multilabel=False)
    return raw_pl.astype(np.int64, copy=True)


def load_example_pool(example_names, dataset_info, cls, args, multilabel):
    feature_arrays, labels, patch_names, slide_names, feature_names = [], [], [], [], []
    for slide_name in example_names:
        path = os.path.join(args.dump_features, slide_name + '.npy')
        if not os.path.exists(path):
            print(f'[warning] missing collected feature file: {path}')
            continue
        slide_features = np.load(path, allow_pickle=True).item()
        feats = slide_features['features']
        feature_arrays.append(feats)
        feature_names.append(slide_name)
        patch_names.extend(slide_features['patch_names'])
        slide_names.extend([slide_name] * feats.shape[0])
        example_wsi_label = slide_wsi_label(dataset_info, slide_features, slide_name, cls, multilabel)

        if args.prompt_type == 'mask':
            pl = class_patch_labels(slide_features['patch_labels'], example_wsi_label, cls, args.c, multilabel)
        else:
            pl = np.zeros(feats.shape[0]) - 1

        if args.prompt_type == 'slideLabel' and args.c > 1:
            if multilabel:
                pl[:] = 1 if has_wsi_label(dataset_info[slide_name], cls) else 0
            elif example_wsi_label != cls:
                pl[:] = 0
            else:
                pl[:] = 1
        elif args.prompt_type != 'mask':
            pl = load_weak_prompts(
                slide_name, example_wsi_label, args.wsi_path, pl,
                slide_features['patch_names'], args.prompt_path, args.prompt_type, side=args.patch_scale
            )
            if args.c > 1:
                pl[pl == 0] = 255
                if multilabel:
                    pl[pl == -1] = 1 if has_wsi_label(dataset_info[slide_name], cls) else 0
                else:
                    pl[pl == -1] = 1 if example_wsi_label == cls else 0
        labels.append(pl)

    if not feature_arrays:
        raise ValueError('No example feature arrays were loaded.')

    feature_dim = get_min_feature_dim(feature_arrays, f'visualize class={cls} examples')
    feature_arrays = [
        align_numpy_feature_dim(feat, feature_dim, f'visualize example slide {name}')
        for feat, name in zip(feature_arrays, feature_names)
    ]
    feats = torch.from_numpy(np.concatenate(feature_arrays, 0).astype(np.float32, copy=False)).cuda()
    token_labels = torch.from_numpy(np.concatenate(labels, 0)).cuda().long()
    return feats, token_labels, np.asarray(slide_names), np.asarray(patch_names, dtype=object)


def refine_example_labels(example_feats, example_labels, patch_names, example_names, cls, args, multilabel):
    if args.prompt_type != 'mask' and args.c == 1:
        return execute_tagger(
            example_feats, example_labels, patch_names, example_names,
            vis_info=None, uncertain=args.ignore, topk=args.topk
        )
    if args.prompt_type == 'slideLabel' and args.c > 1 and not multilabel:
        return execute_subtyping_tagger(
            example_feats, example_labels, patch_names, example_names,
            vis_info=None, uncertain=args.ignore, topk=args.topk
        )
    return example_labels


def apply_reference_sparsity(example_feats, example_labels, slide_names, patch_names, args):
    strategy = args.reference_sparsify_strategy
    if strategy == 'auto':
        strategy = 'legacy' if args.c == 1 else 'quality'
    if 'return_indices' not in inspect.signature(sparsify_reference_tokens).parameters:
        raise RuntimeError(
            'The imported core/main.py is older than scripts/visualize_example_tokens.py: '
            'sparsify_reference_tokens() does not support return_indices. '
            'Please update/sync core/main.py from optimized/main, or copy the latest '
            'core/main.py together with this script before running the visualization.'
        )
    example_feats, example_labels, keep_idxs = sparsify_reference_tokens(
        example_feats,
        example_labels,
        args.reference_token_budget,
        args.reference_anchor_ratio,
        strategy=strategy,
        random_ratio=args.reference_random_ratio,
        return_indices=True,
    )
    keep_idxs_np = keep_idxs.detach().cpu().numpy()
    return example_feats, example_labels, slide_names[keep_idxs_np], patch_names[keep_idxs_np], strategy


def filter_visualized_reference_tokens(example_feats, example_labels, slide_names, patch_names, args, context):
    keep = reference_mask_for_task(example_labels, args)
    kept = int(keep.sum().item())
    total = int(example_labels.shape[0])
    positive_count = int((example_labels == 1).sum().item())
    if kept == 0:
        raise ValueError(f'{context}: no usable reference tokens after dropping uncertain/ignored labels.')
    if positive_count == 0:
        raise ValueError(f'{context}: no positive target tokens for this class.')
    if kept == total:
        return example_feats, example_labels, slide_names, patch_names
    keep_np = keep.detach().cpu().numpy()
    print(f'[reference] {context}: kept {kept}/{total} usable tokens; dropped {total - kept} ignored tokens.')
    return example_feats[keep], example_labels[keep], slide_names[keep_np], patch_names[keep_np]


def sample_tokens_for_tsne(feats, labels, slide_names, patch_names, max_tokens, seed):
    total = feats.shape[0]
    if max_tokens <= 0 or total <= max_tokens:
        idx = np.arange(total)
    else:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(total, size=max_tokens, replace=False))
    torch_idx = torch.as_tensor(idx, device=feats.device, dtype=torch.long)
    return feats[torch_idx], labels[idx], slide_names[idx], patch_names[idx]


def class_color(label):
    return CLASS_COLORS[int(label) % len(CLASS_COLORS)]


def embed_tokens(feats, args):
    feats_np = feats.detach().cpu().numpy().astype(np.float32, copy=False)
    if feats_np.shape[0] < 2:
        return np.zeros((feats_np.shape[0], 2), dtype=np.float32), feats_np

    reduced = feats_np
    if args.pca_dim > 0 and feats_np.shape[1] > args.pca_dim and feats_np.shape[0] > args.pca_dim:
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        reduced = pca.fit_transform(feats_np).astype(np.float32, copy=False)

    perplexity = min(float(args.perplexity), max(1.0, (reduced.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=args.tsne_iter,
        init='random',
        random_state=args.seed,
        learning_rate=200.0,
    )
    embedding = tsne.fit_transform(reduced).astype(np.float32, copy=False)
    return embedding, reduced


def save_csv(path, embedding, labels, slide_names, patch_names):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'token_label', 'token_label_name', 'slide', 'patch'])
        for xy, label, slide, patch in zip(embedding, labels, slide_names, patch_names):
            label_int = int(label)
            writer.writerow([
                float(xy[0]), float(xy[1]), label_int,
                TOKEN_LABEL_NAMES.get(label_int, str(label_int)),
                slide, patch,
            ])


def save_class_csv(path, embedding, class_labels, slide_names, patch_names):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'class', 'slide', 'patch'])
        for xy, label, slide, patch in zip(embedding, class_labels, slide_names, patch_names):
            writer.writerow([float(xy[0]), float(xy[1]), int(label), slide, patch])


def save_svg_plot(path, embedding, labels, slide_names, title):
    labels_np = np.asarray(labels).astype(int)
    width, height = 900, 760
    margin = 70
    if embedding.shape[0] == 0:
        xs = np.asarray([])
        ys = np.asarray([])
    else:
        x_min, x_max = float(embedding[:, 0].min()), float(embedding[:, 0].max())
        y_min, y_max = float(embedding[:, 1].min()), float(embedding[:, 1].max())
        x_span = max(x_max - x_min, 1e-8)
        y_span = max(y_max - y_min, 1e-8)
        xs = margin + (embedding[:, 0] - x_min) / x_span * (width - margin * 2)
        ys = height - margin - (embedding[:, 1] - y_min) / y_span * (height - margin * 2)

    unique_labels = sorted(set(labels_np.tolist()))
    legend_items = []
    y_legend = 44
    for label in unique_labels:
        color = TOKEN_LABEL_COLORS.get(label, '#7c3aed')
        name = TOKEN_LABEL_NAMES.get(label, str(label))
        legend_items.append(
            f'<circle cx="710" cy="{y_legend}" r="5" fill="{color}" opacity="0.85" />'
            f'<text x="724" y="{y_legend + 4}" font-size="12" fill="#111827">'
            f'{escape(str(label))}: {escape(name)}</text>'
        )
        y_legend += 20

    points = []
    for x, y, label, slide in zip(xs, ys, labels_np, slide_names):
        color = TOKEN_LABEL_COLORS.get(int(label), '#7c3aed')
        points.append(
            f'<circle cx="{float(x):.3f}" cy="{float(y):.3f}" r="2.8" '
            f'fill="{color}" opacity="0.68"><title>{escape(str(slide))} '
            f'label={int(label)}</title></circle>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff" />\n'
        f'<text x="{margin}" y="38" font-size="20" font-family="Arial, sans-serif" '
        f'fill="#111827">{escape(title)}</text>\n'
        f'<rect x="{margin}" y="{margin}" width="{width - margin * 2}" '
        f'height="{height - margin * 2}" fill="#f9fafb" stroke="#d1d5db" />\n'
        + '\n'.join(points) + '\n'
        + '\n'.join(legend_items) + '\n'
        f'<text x="{width / 2 - 35:.0f}" y="{height - 24}" font-size="13" '
        'font-family="Arial, sans-serif" fill="#374151">t-SNE 1</text>\n'
        f'<text x="18" y="{height / 2 + 35:.0f}" font-size="13" '
        'font-family="Arial, sans-serif" fill="#374151" '
        'transform="rotate(-90 18 '
        f'{height / 2 + 35:.0f})">t-SNE 2</text>\n'
        '</svg>\n'
    )
    with open(path, 'w') as f:
        f.write(svg)


def save_svg_class_plot(path, embedding, class_labels, slide_names, title):
    labels_np = np.asarray(class_labels).astype(int)
    width, height = 980, 820
    margin = 70
    if embedding.shape[0] == 0:
        xs = np.asarray([])
        ys = np.asarray([])
    else:
        x_min, x_max = float(embedding[:, 0].min()), float(embedding[:, 0].max())
        y_min, y_max = float(embedding[:, 1].min()), float(embedding[:, 1].max())
        x_span = max(x_max - x_min, 1e-8)
        y_span = max(y_max - y_min, 1e-8)
        xs = margin + (embedding[:, 0] - x_min) / x_span * (width - margin * 2)
        ys = height - margin - (embedding[:, 1] - y_min) / y_span * (height - margin * 2)

    unique_labels = sorted(set(labels_np.tolist()))
    legend_items = []
    y_legend = 44
    for label in unique_labels:
        color = class_color(label)
        legend_items.append(
            f'<circle cx="780" cy="{y_legend}" r="5" fill="{color}" opacity="0.85" />'
            f'<text x="794" y="{y_legend + 4}" font-size="12" fill="#111827">'
            f'class {escape(str(label))}</text>'
        )
        y_legend += 20

    points = []
    for x, y, label, slide in zip(xs, ys, labels_np, slide_names):
        points.append(
            f'<circle cx="{float(x):.3f}" cy="{float(y):.3f}" r="2.8" '
            f'fill="{class_color(label)}" opacity="0.68"><title>{escape(str(slide))} '
            f'class={int(label)}</title></circle>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff" />\n'
        f'<text x="{margin}" y="38" font-size="20" font-family="Arial, sans-serif" '
        f'fill="#111827">{escape(title)}</text>\n'
        f'<rect x="{margin}" y="{margin}" width="{width - margin * 2}" '
        f'height="{height - margin * 2}" fill="#f9fafb" stroke="#d1d5db" />\n'
        + '\n'.join(points) + '\n'
        + '\n'.join(legend_items) + '\n'
        f'<text x="{width / 2 - 35:.0f}" y="{height - 24}" font-size="13" '
        'font-family="Arial, sans-serif" fill="#374151">t-SNE 1</text>\n'
        f'<text x="18" y="{height / 2 + 35:.0f}" font-size="13" '
        'font-family="Arial, sans-serif" fill="#374151" '
        'transform="rotate(-90 18 '
        f'{height / 2 + 35:.0f})">t-SNE 2</text>\n'
        '</svg>\n'
    )
    with open(path, 'w') as f:
        f.write(svg)


def save_plot(path, embedding, labels, slide_names, title):
    if plt is None:
        print('[warning] matplotlib is not installed; skipped plot generation.')
        return
    labels_np = np.asarray(labels).astype(int)
    unique_labels = sorted(set(labels_np.tolist()))
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(8, 7))
    for idx, label in enumerate(unique_labels):
        mask = labels_np == label
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=7, alpha=0.7, color=cmap(idx % 10),
            label=f'{label}:{TOKEN_LABEL_NAMES.get(label, str(label))}'
        )
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_class_plot(path, embedding, class_labels, slide_names, title):
    if plt is None:
        print('[warning] matplotlib is not installed; skipped plot generation.')
        return
    labels_np = np.asarray(class_labels).astype(int)
    unique_labels = sorted(set(labels_np.tolist()))
    plt.figure(figsize=(9.5, 7.8))
    for label in unique_labels:
        mask = labels_np == label
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=7, alpha=0.7, color=class_color(label), label=f'class {label}'
        )
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_info = load_dataset_info(args, context='example token visualization')
    multilabel = dataset_is_multilabel(dataset_info, args)
    example_names, target_counts, labeled_names = select_examples(dataset_info, args)
    print('[split] example candidate label counts: ' + label_counts_text(label_counts(labeled_names, dataset_info)))
    if target_counts is not None:
        print('[split] example target label counts: ' + label_counts_text(target_counts))
    print('[split] selected example label counts: ' + label_counts_text(label_counts(example_names, dataset_info)))
    print('[split] selected examples: ' + str(len(example_names)))

    summary = {
        'classes': args.classes,
        'plot_mode': args.plot_mode,
        'example_names': example_names,
        'example_target_counts': target_counts,
        'selected_example_label_counts': label_counts(example_names, dataset_info),
        'outputs': {},
    }
    combined_feats = []
    combined_labels = []
    combined_slides = []
    combined_patches = []
    combined_class_counts = {}
    sparse_strategies = {}

    for cls in args.classes:
        print(f'[visualize] class={cls}: loading example token pool')
        example_feats, example_labels, slide_names, patch_names = load_example_pool(
            example_names, dataset_info, cls, args, multilabel
        )
        example_labels = refine_example_labels(
            example_feats, example_labels, list(patch_names), example_names, cls, args, multilabel
        )
        example_feats, example_labels, slide_names, patch_names = filter_visualized_reference_tokens(
            example_feats, example_labels, slide_names, patch_names, args, f'visualize class={cls}'
        )
        example_feats, example_labels, slide_names, patch_names, sparse_strategy = apply_reference_sparsity(
            example_feats, example_labels, slide_names, patch_names, args
        )
        if args.context_centering == 'example':
            example_feats, _ = apply_context_feature_centering(example_feats, example_feats[:1], mode='example')

        labels_np = example_labels.detach().cpu().numpy()
        pos_mask = labels_np == 1
        pos_count = int(pos_mask.sum())
        sparse_strategies[str(cls)] = sparse_strategy
        combined_class_counts[str(cls)] = pos_count
        if args.plot_mode in ['combined', 'both']:
            if pos_count == 0:
                print(f'[warning] class={cls}: no positive target tokens kept for combined plot; skipped.')
            else:
                pos_torch = torch.as_tensor(pos_mask, device=example_feats.device, dtype=torch.bool)
                pos_feats = example_feats[pos_torch]
                pos_labels = np.full(pos_count, int(cls), dtype=np.int64)
                sampled_feats, sampled_labels, sampled_slides, sampled_patches = sample_tokens_for_tsne(
                    pos_feats, pos_labels, slide_names[pos_mask], patch_names[pos_mask],
                    args.max_tokens_per_class, args.seed + int(cls)
                )
                combined_feats.append(sampled_feats.detach().cpu())
                combined_labels.append(sampled_labels)
                combined_slides.append(sampled_slides)
                combined_patches.append(sampled_patches)
                print(
                    f'[visualize] class={cls}: collected {sampled_feats.shape[0]} '
                    f'positive tokens for combined plot'
                )
                del pos_feats, sampled_feats

        class_output = {
            'positive_tokens_before_plot_sampling': pos_count,
            'reference_sparsify_strategy': sparse_strategy,
        }
        if args.plot_mode in ['per_class', 'both']:
            sampled_feats, sampled_labels, sampled_slides, sampled_patches = sample_tokens_for_tsne(
                example_feats, labels_np, slide_names, patch_names,
                args.max_tokens_per_class, args.seed + int(cls)
            )
            embedding, _ = embed_tokens(sampled_feats, args)
            class_prefix = os.path.join(args.out_dir, f'class_{cls}')
            csv_path = class_prefix + '_tokens_tsne.csv'
            png_path = class_prefix + '_tokens_tsne.png'
            svg_path = class_prefix + '_tokens_tsne.svg'
            save_csv(csv_path, embedding, sampled_labels, sampled_slides, sampled_patches)
            save_svg_plot(
                svg_path, embedding, sampled_labels, sampled_slides,
                f'class {cls} example tokens ({sampled_feats.shape[0]} sampled)'
            )
            save_plot(
                png_path, embedding, sampled_labels, sampled_slides,
                f'class {cls} example tokens ({sampled_feats.shape[0]} sampled)'
            )
            counts = {str(k): int(v) for k, v in zip(*np.unique(sampled_labels.astype(int), return_counts=True))}
            class_output.update({
                'csv': csv_path,
                'svg': svg_path,
                'plot': png_path if plt is not None else None,
                'tokens_before_plot_sampling': int(example_feats.shape[0]),
                'tokens_plotted': int(sampled_feats.shape[0]),
                'token_label_counts_plotted': counts,
            })
            print(f'[visualize] class={cls}: wrote {csv_path}')
            print(f'[visualize] class={cls}: wrote {svg_path}')
            if plt is not None:
                print(f'[visualize] class={cls}: wrote {png_path}')
            del sampled_feats
        summary['outputs'][str(cls)] = {
            **class_output,
        }

        del example_feats, example_labels
        torch.cuda.empty_cache()

    if args.plot_mode in ['combined', 'both']:
        if not combined_feats:
            raise ValueError('No positive target tokens were collected for the combined class plot.')
        all_feats = torch.cat(combined_feats, 0)
        all_labels = np.concatenate(combined_labels, 0)
        all_slides = np.concatenate(combined_slides, 0)
        all_patches = np.concatenate(combined_patches, 0)
        if args.max_tokens_total > 0 and all_feats.shape[0] > args.max_tokens_total:
            all_feats, all_labels, all_slides, all_patches = sample_tokens_for_tsne(
                all_feats, all_labels, all_slides, all_patches,
                args.max_tokens_total, args.seed + 7919
            )

        print(f'[visualize] combined: running t-SNE for {all_feats.shape[0]} tokens')
        embedding, _ = embed_tokens(all_feats, args)
        combined_prefix = os.path.join(args.out_dir, 'combined_classes')
        csv_path = combined_prefix + '_tokens_tsne.csv'
        png_path = combined_prefix + '_tokens_tsne.png'
        svg_path = combined_prefix + '_tokens_tsne.svg'
        save_class_csv(csv_path, embedding, all_labels, all_slides, all_patches)
        save_svg_class_plot(
            svg_path, embedding, all_labels, all_slides,
            f'combined class example tokens ({all_feats.shape[0]} sampled)'
        )
        save_class_plot(
            png_path, embedding, all_labels, all_slides,
            f'combined class example tokens ({all_feats.shape[0]} sampled)'
        )
        counts = {str(k): int(v) for k, v in zip(*np.unique(all_labels.astype(int), return_counts=True))}
        summary['combined_output'] = {
            'csv': csv_path,
            'svg': svg_path,
            'plot': png_path if plt is not None else None,
            'tokens_plotted': int(all_feats.shape[0]),
            'class_token_counts_before_total_sampling': combined_class_counts,
            'class_token_counts_plotted': counts,
            'reference_sparsify_strategy_by_class': sparse_strategies,
        }
        print(f'[visualize] combined: wrote {csv_path}')
        print(f'[visualize] combined: wrote {svg_path}')
        if plt is not None:
            print(f'[visualize] combined: wrote {png_path}')

    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[visualize] wrote {summary_path}')


def label_counts_text(counts):
    return ', '.join(f'{k}:{counts[k]}' for k in sorted(counts))


if __name__ == '__main__':
    main()
