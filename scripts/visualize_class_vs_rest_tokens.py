#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
from html import escape

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, silhouette_score

import visualize_example_tokens as vet

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


TOKEN_LABEL_NAMES = {
    1: 'target_class',
    0: 'other_foreground',
    255: 'background',
    254: 'uncertain_overlap',
    -1: 'unknown',
}

TOKEN_LABEL_COLORS = {
    1: '#dc2626',
    0: '#2563eb',
    255: '#16a34a',
    254: '#f59e0b',
    -1: '#9ca3af',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize one target class against other foreground classes and background.'
    )
    parser.add_argument('--raw_feature_path', default='', help='h5 folder/path; used only to fill h5 dataset_info entries when needed')
    parser.add_argument('--wsi_path', required=True)
    parser.add_argument('--dump_features', required=True)
    parser.add_argument('--dataset_info', required=True)
    parser.add_argument('--prompt_type', default='mask', choices=['mask', 'slideLabel', 'box', 'roughMask'])
    parser.add_argument('--prompt_path', default='')
    parser.add_argument('--class_num', '--c', dest='c', type=int, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='target class ids to visualize')
    parser.add_argument('--all_classes', action='store_true', help='visualize class ids 1..class_num')
    parser.add_argument('--out_dir', default='records/class_vs_rest_token_vis')
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
    parser.add_argument('--max_tokens_per_group', type=int, default=10000, help='max plotted tokens per label group; 0 keeps all')
    parser.add_argument('--max_metric_tokens', type=int, default=5000, help='max tokens for O(n^2) metrics such as silhouette; 0 disables sampling')
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--tsne_iter', type=int, default=1000)
    parser.add_argument('--overlap_label', default='target', choices=['target', 'other', 'uncertain'],
        help='how to label multilabel patches that contain both target and another foreground class')
    args = parser.parse_args()
    if args.all_classes:
        args.classes = list(range(1, args.c + 1)) if args.c > 1 else [1]
    elif not args.classes:
        parser.error('provide --classes or use --all_classes')
    if args.prompt_type != 'mask':
        print(
            '[warning] background/other_foreground separation is only explicit with --prompt_type mask. '
            'Non-mask prompts may not contain reliable patch-level background labels.'
        )
    return args


def one_vs_rest_patch_labels(raw_patch_labels, cls, class_num, slide_has_target, overlap_label):
    raw = np.asarray(raw_patch_labels)
    if raw.ndim == 2:
        if raw.shape[1] < class_num:
            print(
                f'[warning] patch label matrix has {raw.shape[1]} columns, '
                f'but class_num={class_num}. Using available columns.'
            )
        target_col = int(cls) - 1
        unique = set(np.unique(raw).astype(int).tolist())
        if unique.issubset({0, 1}):
            target = raw[:, target_col] > 0 if 0 <= target_col < raw.shape[1] else np.zeros(raw.shape[0], dtype=bool)
            other = np.zeros(raw.shape[0], dtype=bool)
            if 0 <= target_col < raw.shape[1] and raw.shape[1] > 1:
                other = np.delete(raw, target_col, axis=1).max(1) > 0

            labels = np.full(raw.shape[0], 255, dtype=np.int64)
            labels[other] = 0
            labels[target] = 1
            overlap = target & other
            if overlap_label == 'other':
                labels[overlap] = 0
            elif overlap_label == 'uncertain':
                labels[overlap] = 254
            return labels

        target = raw[:, target_col] == 1 if 0 <= target_col < raw.shape[1] else np.zeros(raw.shape[0], dtype=bool)
        explicit_background = (raw == 255).all(1)
        labels = np.full(raw.shape[0], 254, dtype=np.int64)
        labels[explicit_background] = 255
        labels[target] = 1
        if overlap_label == 'uncertain' and raw.shape[1] > 1:
            other_positive = np.delete(raw, target_col, axis=1).max(1) == 1 if 0 <= target_col < raw.shape[1] else np.zeros(raw.shape[0], dtype=bool)
            labels[target & other_positive] = 254
        return labels

    raw = raw.reshape(-1)
    labels = np.full(raw.shape[0], 254 if np.any(raw == 254) else 255, dtype=np.int64)
    labels[raw == 255] = 255
    labels[raw == 254] = 254

    unique = set(np.unique(raw).astype(int).tolist())
    if class_num > 1 and unique.issubset({0, 1}):
        if slide_has_target:
            labels[raw == 1] = 1
        else:
            labels[raw == 1] = 0
        return labels

    if class_num > 1:
        valid_foreground = (raw > 0) & (raw < 254)
        labels[(valid_foreground) & (raw != int(cls))] = 0
        labels[raw == int(cls)] = 1
        return labels

    valid_foreground = (raw > 0) & (raw < 254)
    labels[valid_foreground] = 1
    return labels


def load_class_pool(example_names, dataset_info, cls, args, multilabel):
    feature_arrays, labels, patch_names, slide_names, feature_names = [], [], [], [], []
    for slide_name in example_names:
        path = os.path.join(args.dump_features, slide_name + '.npy')
        if not os.path.exists(path):
            print(f'[warning] missing collected feature file: {path}')
            continue
        slide_features = np.load(path, allow_pickle=True).item()
        if 'patch_labels' not in slide_features:
            print(f'[warning] missing patch_labels in collected feature file: {path}; skipped.')
            continue

        feats = slide_features['features']
        slide_has_target = (
            vet.has_wsi_label(dataset_info[slide_name], cls)
            if multilabel and slide_name in dataset_info
            else vet.slide_wsi_label(dataset_info, slide_features, slide_name, cls, multilabel) == cls
        )
        pl = one_vs_rest_patch_labels(
            slide_features['patch_labels'],
            cls,
            args.c,
            slide_has_target,
            args.overlap_label,
        )
        if pl.shape[0] != feats.shape[0]:
            raise ValueError(
                f'{slide_name}: patch_labels length {pl.shape[0]} does not match features {feats.shape[0]}.'
            )

        feature_arrays.append(feats)
        feature_names.append(slide_name)
        labels.append(pl)
        patch_names.extend(slide_features['patch_names'])
        slide_names.extend([slide_name] * feats.shape[0])

    if not feature_arrays:
        raise ValueError('No example feature arrays with patch_labels were loaded.')

    feature_dim = vet.get_min_feature_dim(feature_arrays, f'class-vs-rest class={cls} examples')
    feature_arrays = [
        vet.align_numpy_feature_dim(feat, feature_dim, f'class-vs-rest example slide {name}')
        for feat, name in zip(feature_arrays, feature_names)
    ]
    feats = torch.from_numpy(np.concatenate(feature_arrays, 0).astype(np.float32, copy=False)).cuda()
    token_labels = torch.from_numpy(np.concatenate(labels, 0)).cuda().long()
    return feats, token_labels, np.asarray(slide_names), np.asarray(patch_names, dtype=object)


def sample_per_label(feats, labels, slide_names, patch_names, max_per_group, seed):
    labels = np.asarray(labels).astype(int)
    rng = np.random.RandomState(seed)
    selected = []
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        if max_per_group > 0 and idx.shape[0] > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        selected.append(idx)
    idx = np.sort(np.concatenate(selected, 0)) if selected else np.empty(0, dtype=np.int64)
    torch_idx = torch.as_tensor(idx, device=feats.device, dtype=torch.long)
    return feats[torch_idx], labels[idx], slide_names[idx], patch_names[idx]


def sample_for_metrics(feats_np, labels, max_tokens, seed):
    labels = np.asarray(labels).astype(int)
    valid = np.isin(labels, [1, 0, 255])
    idx = np.where(valid)[0]
    if max_tokens > 0 and idx.shape[0] > max_tokens:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(idx, size=max_tokens, replace=False))
    return feats_np[idx], labels[idx]


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def separability_metrics(feats, labels, max_metric_tokens, seed):
    feats_np = feats.detach().cpu().numpy().astype(np.float32, copy=False)
    feats_np, labels = sample_for_metrics(feats_np, labels, max_metric_tokens, seed)
    if feats_np.shape[0] == 0:
        return {}

    feats_t = torch.from_numpy(feats_np)
    feats_t = F.normalize(feats_t.float(), p=2, dim=1, eps=1e-8)
    labels = np.asarray(labels).astype(int)
    present = [label for label in [1, 0, 255] if np.any(labels == label)]
    centroids = {}
    for label in present:
        centroids[label] = F.normalize(feats_t[labels == label].mean(0, keepdim=True), p=2, dim=1, eps=1e-8)

    metrics = {
        'tokens_used': int(feats_np.shape[0]),
        'label_counts_used': {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }

    if len(present) >= 2:
        centroid_tensor = torch.cat([centroids[label] for label in present], 0)
        sims = (feats_t @ centroid_tensor.t()).cpu().numpy()
        pred = np.asarray(present, dtype=int)[sims.argmax(1)]
        metrics['nearest_centroid_acc'] = float((pred == labels).mean())
        per_label_acc = {}
        for label in present:
            mask = labels == label
            per_label_acc[str(label)] = float((pred[mask] == label).mean())
        metrics['nearest_centroid_acc_per_label'] = per_label_acc
        metrics['nearest_centroid_balanced_acc'] = float(np.mean(list(per_label_acc.values())))

    def centroid_margin(pos_label, neg_labels):
        if not neg_labels:
            return None, None
        if pos_label not in centroids or any(label not in centroids for label in neg_labels):
            return None, None
        pos_sim = (feats_t @ centroids[pos_label].t()).reshape(-1).cpu().numpy()
        neg_centroids = torch.cat([centroids[label] for label in neg_labels], 0)
        neg_sim = (feats_t @ neg_centroids.t()).max(1).values.cpu().numpy()
        score = pos_sim - neg_sim
        valid = (labels == pos_label) | np.isin(labels, neg_labels)
        return (labels[valid] == pos_label).astype(int), score[valid]

    y, score = centroid_margin(1, [label for label in [0, 255] if label in centroids])
    metrics['auc_target_vs_non_target'] = safe_auc(y, score) if y is not None else None
    y, score = centroid_margin(1, [0])
    metrics['auc_target_vs_other_foreground'] = safe_auc(y, score) if y is not None else None
    y, score = centroid_margin(1, [255])
    metrics['auc_target_vs_background'] = safe_auc(y, score) if y is not None else None

    if len(present) >= 2 and feats_np.shape[0] >= len(present) + 1:
        try:
            metrics['silhouette_cosine'] = float(silhouette_score(feats_np, labels, metric='cosine'))
        except ValueError:
            metrics['silhouette_cosine'] = None

    centroid_cosine = {}
    for left in present:
        centroid_cosine[str(left)] = {}
        for right in present:
            centroid_cosine[str(left)][str(right)] = float((centroids[left] @ centroids[right].t()).item())
    metrics['centroid_cosine'] = centroid_cosine
    return metrics


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


def save_svg(path, embedding, labels, slide_names, title):
    labels_np = np.asarray(labels).astype(int)
    width, height = 940, 780
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

    legend_items = []
    y_legend = 44
    for label in sorted(set(labels_np.tolist())):
        color = TOKEN_LABEL_COLORS.get(label, '#7c3aed')
        name = TOKEN_LABEL_NAMES.get(label, str(label))
        legend_items.append(
            f'<circle cx="720" cy="{y_legend}" r="5" fill="{color}" opacity="0.85" />'
            f'<text x="734" y="{y_legend + 4}" font-size="12" fill="#111827">'
            f'{escape(str(label))}: {escape(name)}</text>'
        )
        y_legend += 20

    points = []
    for x, y, label, slide in zip(xs, ys, labels_np, slide_names):
        points.append(
            f'<circle cx="{float(x):.3f}" cy="{float(y):.3f}" r="2.8" '
            f'fill="{TOKEN_LABEL_COLORS.get(int(label), "#7c3aed")}" opacity="0.68">'
            f'<title>{escape(str(slide))} label={int(label)}</title></circle>'
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


def save_png(path, embedding, labels, title):
    if plt is None:
        print('[warning] matplotlib is not installed; skipped plot generation.')
        return
    labels_np = np.asarray(labels).astype(int)
    plt.figure(figsize=(8.6, 7.2))
    for label in sorted(set(labels_np.tolist())):
        mask = labels_np == label
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=7, alpha=0.7,
            color=TOKEN_LABEL_COLORS.get(label, '#7c3aed'),
            label=f'{label}:{TOKEN_LABEL_NAMES.get(label, str(label))}'
        )
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def label_counts(labels):
    labels = np.asarray(labels).astype(int)
    return {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_info = vet.load_dataset_info(args, context='class-vs-rest token visualization')
    multilabel = vet.dataset_is_multilabel(dataset_info, args)
    example_names, target_counts, labeled_names = vet.select_examples(dataset_info, args)
    print('[split] example candidate label counts: ' + vet.label_counts_text(vet.label_counts(labeled_names, dataset_info)))
    if target_counts is not None:
        print('[split] example target label counts: ' + vet.label_counts_text(target_counts))
    print('[split] selected example label counts: ' + vet.label_counts_text(vet.label_counts(example_names, dataset_info)))
    print('[split] selected examples: ' + str(len(example_names)))

    summary = {
        'classes': args.classes,
        'example_names': example_names,
        'example_target_counts': target_counts,
        'selected_example_label_counts': vet.label_counts(example_names, dataset_info),
        'outputs': {},
    }

    for cls in args.classes:
        print(f'[visualize] class={cls}: loading target/other/background token pool')
        example_feats, example_labels, slide_names, patch_names = load_class_pool(
            example_names, dataset_info, cls, args, multilabel
        )
        example_feats, example_labels, slide_names, patch_names, sparse_strategy = vet.apply_reference_sparsity(
            example_feats, example_labels, slide_names, patch_names, args
        )
        if args.context_centering == 'example':
            example_feats, _ = vet.apply_context_feature_centering(example_feats, example_feats[:1], mode='example')

        labels_np = example_labels.detach().cpu().numpy()
        counts_before = label_counts(labels_np)
        sampled_feats, sampled_labels, sampled_slides, sampled_patches = sample_per_label(
            example_feats, labels_np, slide_names, patch_names,
            args.max_tokens_per_group, args.seed + int(cls)
        )
        embedding, _ = vet.embed_tokens(sampled_feats, args)
        metrics = separability_metrics(
            sampled_feats, sampled_labels, args.max_metric_tokens, args.seed + int(cls) + 1009
        )

        class_prefix = os.path.join(args.out_dir, f'class_{cls}_target_vs_rest')
        csv_path = class_prefix + '_tsne.csv'
        svg_path = class_prefix + '_tsne.svg'
        png_path = class_prefix + '_tsne.png'
        title = f'class {cls}: target vs other foreground vs background ({sampled_feats.shape[0]} sampled)'
        save_csv(csv_path, embedding, sampled_labels, sampled_slides, sampled_patches)
        save_svg(svg_path, embedding, sampled_labels, sampled_slides, title)
        save_png(png_path, embedding, sampled_labels, title)

        summary['outputs'][str(cls)] = {
            'csv': csv_path,
            'svg': svg_path,
            'plot': png_path if plt is not None else None,
            'tokens_before_plot_sampling': int(example_feats.shape[0]),
            'tokens_plotted': int(sampled_feats.shape[0]),
            'token_label_counts_before_plot_sampling': counts_before,
            'token_label_counts_plotted': label_counts(sampled_labels),
            'reference_sparsify_strategy': sparse_strategy,
            'metrics': metrics,
        }
        print(f'[visualize] class={cls}: wrote {csv_path}')
        print(f'[visualize] class={cls}: wrote {svg_path}')
        if plt is not None:
            print(f'[visualize] class={cls}: wrote {png_path}')
        print(f'[metrics] class={cls}: ' + json.dumps(metrics, sort_keys=True))

        del example_feats, example_labels, sampled_feats
        torch.cuda.empty_cache()

    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[visualize] wrote {summary_path}')


if __name__ == '__main__':
    main()
