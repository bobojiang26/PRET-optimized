#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import visualize_example_tokens as vet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Quantify distances between PRET example/reference token classes.'
    )
    parser.add_argument('--raw_feature_path', default='', help='h5 folder/path; used only to fill h5 dataset_info entries when needed')
    parser.add_argument('--wsi_path', required=True)
    parser.add_argument('--dump_features', required=True)
    parser.add_argument('--dataset_info', required=True)
    parser.add_argument('--prompt_type', default='slideLabel')
    parser.add_argument('--prompt_path', default='')
    parser.add_argument('--class_num', '--c', dest='c', type=int, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='target class ids to compare')
    parser.add_argument('--all_classes', action='store_true', help='compare class ids 1..class_num')
    parser.add_argument('--out_dir', default='records/example_token_distances')
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
    parser.add_argument('--multilabel_mask_negative_source', default='other_positive',
        choices=['all_zero', 'other_positive', 'none'])
    parser.add_argument('--max_tokens_per_class', type=int, default=5000,
        help='max positive/target tokens kept per class for distance metrics; 0 keeps all')
    parser.add_argument('--pair_sample_size', type=int, default=20000,
        help='number of random token pairs per class pair; 0 uses all pairs when feasible')
    parser.add_argument('--pair_chunk_size', type=int, default=20000,
        help='chunk size for sampled pair distance computation')
    parser.add_argument('--max_silhouette_tokens', type=int, default=5000,
        help='max tokens used for cosine silhouette; 0 disables silhouette')
    parser.add_argument('--no_plots', action='store_true',
        help='skip matplotlib visualizations and write only CSV/JSON outputs')
    args = parser.parse_args()
    if args.all_classes:
        args.classes = list(range(1, args.c + 1)) if args.c > 1 else [1]
    elif not args.classes:
        parser.error('provide --classes or use --all_classes')
    return args


def sample_torch_rows(feats, max_tokens, seed):
    total = feats.shape[0]
    if max_tokens <= 0 or total <= max_tokens:
        return feats, np.arange(total)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(total, size=max_tokens, replace=False))
    idx_t = torch.as_tensor(idx, device=feats.device, dtype=torch.long)
    return feats[idx_t], idx


def collect_class_tokens(args):
    dataset_info = vet.load_dataset_info(args, context='example token distance analysis')
    multilabel = vet.dataset_is_multilabel(dataset_info, args)
    example_names, target_counts, labeled_names = vet.select_examples(dataset_info, args)
    print('[split] example candidate label counts: ' + vet.label_counts_text(vet.label_counts(labeled_names, dataset_info)))
    if target_counts is not None:
        print('[split] example target label counts: ' + vet.label_counts_text(target_counts))
    print('[split] selected example label counts: ' + vet.label_counts_text(vet.label_counts(example_names, dataset_info)))
    print('[split] selected examples: ' + str(len(example_names)))

    class_feats = {}
    class_meta = {}
    sparse_strategies = {}

    for cls in args.classes:
        print(f'[distance] class={cls}: loading positive example tokens')
        example_feats, example_labels, slide_names, patch_names = vet.load_example_pool(
            example_names, dataset_info, cls, args, multilabel
        )
        example_labels = vet.refine_example_labels(
            example_feats, example_labels, list(patch_names), example_names, cls, args, multilabel
        )
        if multilabel:
            example_feats, example_labels, slide_names, patch_names = vet.filter_ignored_reference_tokens(
                example_feats, example_labels, slide_names, patch_names, args, f'distance class={cls}'
            )
        example_feats, example_labels, slide_names, patch_names, sparse_strategy = vet.apply_reference_sparsity(
            example_feats, example_labels, slide_names, patch_names, args
        )
        if args.context_centering == 'example':
            example_feats, _ = vet.apply_context_feature_centering(example_feats, example_feats[:1], mode='example')

        labels_np = example_labels.detach().cpu().numpy()
        pos_mask = labels_np == 1
        pos_count = int(pos_mask.sum())
        sparse_strategies[str(cls)] = sparse_strategy
        if pos_count == 0:
            print(f'[warning] class={cls}: no positive target tokens kept; skipped.')
            del example_feats, example_labels
            torch.cuda.empty_cache()
            continue

        pos_t = torch.as_tensor(pos_mask, device=example_feats.device, dtype=torch.bool)
        pos_feats = F.normalize(example_feats[pos_t].float(), p=2, dim=1, eps=1e-8)
        sampled_feats, sampled_idx = sample_torch_rows(pos_feats, args.max_tokens_per_class, args.seed + int(cls))
        sampled_feats = sampled_feats.detach().cpu()
        sampled_slides = slide_names[pos_mask][sampled_idx]
        sampled_patches = patch_names[pos_mask][sampled_idx]

        class_feats[int(cls)] = sampled_feats
        class_meta[int(cls)] = {
            'positive_tokens_before_sampling': pos_count,
            'tokens_used': int(sampled_feats.shape[0]),
            'slides_used': sorted({str(_) for _ in sampled_slides.tolist()}),
            'slide_token_counts_used': {
                str(k): int(v)
                for k, v in zip(*np.unique(sampled_slides.astype(str), return_counts=True))
            },
            'patch_examples': [str(_) for _ in sampled_patches[:20].tolist()],
        }
        print(f'[distance] class={cls}: using {sampled_feats.shape[0]}/{pos_count} positive tokens')

        del example_feats, example_labels, pos_feats, sampled_feats
        torch.cuda.empty_cache()

    if len(class_feats) < 2:
        raise ValueError('Need at least two classes with positive tokens to compute inter-class distances.')

    min_dim = min(feats.shape[1] for feats in class_feats.values())
    aligned = {}
    for cls, feats in class_feats.items():
        if feats.shape[1] != min_dim:
            print(f'[warning] class={cls}: feature dim {feats.shape[1]} truncated to common dim {min_dim}')
            feats = feats[:, :min_dim]
        aligned[cls] = F.normalize(feats.float(), p=2, dim=1, eps=1e-8)

    return aligned, class_meta, {
        'classes': args.classes,
        'example_names': example_names,
        'example_target_counts': target_counts,
        'selected_example_label_counts': vet.label_counts(example_names, dataset_info),
        'reference_sparsify_strategy_by_class': sparse_strategies,
        'common_feature_dim': int(min_dim),
    }


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        'min': float(np.min(values)),
        'p05': float(np.percentile(values, 5)),
        'p25': float(np.percentile(values, 25)),
        'median': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
    }


def class_centroids(class_feats):
    centroids = {}
    for cls, feats in class_feats.items():
        centroids[cls] = F.normalize(feats.mean(0, keepdim=True), p=2, dim=1, eps=1e-8)
    return centroids


def write_class_stats(path, rows):
    fieldnames = [
        'class', 'tokens_used', 'positive_tokens_before_sampling',
        'mean_similarity_to_own_centroid', 'mean_distance_to_own_centroid',
        'p95_distance_to_own_centroid', 'nearest_other_centroid_class',
        'nearest_other_centroid_distance', 'own_minus_nearest_other_margin_mean',
        'own_minus_nearest_other_margin_p05', 'nearest_token_competitor_class',
        'nearest_centroid_recall',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_centroid_distances(path, classes, centroids):
    fieldnames = [
        'class_i', 'class_j', 'centroid_cosine_similarity',
        'centroid_cosine_distance', 'centroid_angular_distance_norm',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls_i in classes:
            for cls_j in classes:
                sim = float((centroids[cls_i] @ centroids[cls_j].t()).item())
                sim_clip = max(-1.0, min(1.0, sim))
                writer.writerow({
                    'class_i': cls_i,
                    'class_j': cls_j,
                    'centroid_cosine_similarity': sim,
                    'centroid_cosine_distance': 1.0 - sim,
                    'centroid_angular_distance_norm': float(np.arccos(sim_clip) / np.pi),
                })


def sampled_pair_distances(feats_i, feats_j, same_class, sample_size, chunk_size, seed):
    n_i, n_j = feats_i.shape[0], feats_j.shape[0]
    if n_i == 0 or n_j == 0:
        return np.asarray([], dtype=np.float32)

    rng = np.random.RandomState(seed)
    if sample_size <= 0:
        total_pairs = n_i * n_j
        if same_class:
            total_pairs = max(0, n_i * (n_i - 1))
        sample_size = min(total_pairs, 20000)
    if sample_size <= 0:
        return np.asarray([], dtype=np.float32)

    idx_i = rng.randint(0, n_i, size=sample_size)
    idx_j = rng.randint(0, n_j, size=sample_size)
    if same_class and n_i > 1:
        same = idx_i == idx_j
        while np.any(same):
            idx_j[same] = rng.randint(0, n_j, size=int(same.sum()))
            same = idx_i == idx_j

    distances = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, sample_size, chunk_size):
        end = min(sample_size, start + chunk_size)
        left = feats_i[torch.as_tensor(idx_i[start:end], dtype=torch.long)]
        right = feats_j[torch.as_tensor(idx_j[start:end], dtype=torch.long)]
        sim = (left * right).sum(1).clamp(-1.0, 1.0)
        distances.append((1.0 - sim).cpu().numpy())
    return np.concatenate(distances, 0)


def write_pairwise_token_distances(path, classes, class_feats, args):
    fieldnames = [
        'class_i', 'class_j', 'tokens_i', 'tokens_j', 'pairs_sampled',
        'cosine_distance_min', 'cosine_distance_p05', 'cosine_distance_p25',
        'cosine_distance_median', 'cosine_distance_p75', 'cosine_distance_p95',
        'cosine_distance_max', 'cosine_distance_mean', 'cosine_distance_std',
    ]
    rows = []
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx_i, cls_i in enumerate(classes):
            for cls_j in classes[idx_i:]:
                distances = sampled_pair_distances(
                    class_feats[cls_i], class_feats[cls_j], cls_i == cls_j,
                    args.pair_sample_size, args.pair_chunk_size,
                    args.seed + cls_i * 1009 + cls_j * 9176,
                )
                stats = percentile_summary(distances)
                row = {
                    'class_i': cls_i,
                    'class_j': cls_j,
                    'tokens_i': int(class_feats[cls_i].shape[0]),
                    'tokens_j': int(class_feats[cls_j].shape[0]),
                    'pairs_sampled': int(distances.shape[0]),
                    'cosine_distance_min': stats.get('min'),
                    'cosine_distance_p05': stats.get('p05'),
                    'cosine_distance_p25': stats.get('p25'),
                    'cosine_distance_median': stats.get('median'),
                    'cosine_distance_p75': stats.get('p75'),
                    'cosine_distance_p95': stats.get('p95'),
                    'cosine_distance_max': stats.get('max'),
                    'cosine_distance_mean': stats.get('mean'),
                    'cosine_distance_std': stats.get('std'),
                }
                writer.writerow(row)
                rows.append(row)
    return rows


def nearest_centroid_outputs(class_feats, centroids, classes):
    centroid_tensor = torch.cat([centroids[cls] for cls in classes], 0)
    pred_rows = []
    confusion = {cls: {other: 0 for other in classes} for cls in classes}
    class_stats_extra = {}

    for cls in classes:
        feats = class_feats[cls]
        sims = feats @ centroid_tensor.t()
        pred_idx = sims.argmax(1).cpu().numpy()
        pred_classes = np.asarray(classes, dtype=int)[pred_idx]
        for pred_cls in pred_classes:
            confusion[cls][int(pred_cls)] += 1

        own_idx = classes.index(cls)
        own_sim = sims[:, own_idx]
        other_sims = sims.clone()
        other_sims[:, own_idx] = -1e9
        nearest_other_sim, nearest_other_idx = other_sims.max(1)
        margin = (own_sim - nearest_other_sim).cpu().numpy()
        nearest_token_competitor_class = int(classes[int(torch.mode(nearest_other_idx).values.item())])
        margin_stats = percentile_summary(margin)
        class_stats_extra[cls] = {
            'nearest_centroid_recall': float((pred_classes == cls).mean()),
            'nearest_token_competitor_class': nearest_token_competitor_class,
            'own_minus_nearest_other_margin_mean': margin_stats.get('mean'),
            'own_minus_nearest_other_margin_p05': margin_stats.get('p05'),
        }

    for cls in classes:
        total = max(1, sum(confusion[cls].values()))
        for pred_cls in classes:
            count = int(confusion[cls][pred_cls])
            pred_rows.append({
                'true_class': cls,
                'pred_class': pred_cls,
                'count': count,
                'row_rate': float(count / total),
            })

    total_correct = sum(confusion[cls][cls] for cls in classes)
    total = sum(sum(row.values()) for row in confusion.values())
    overall_acc = float(total_correct / max(total, 1))
    return pred_rows, class_stats_extra, overall_acc


def write_confusion(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['true_class', 'pred_class', 'count', 'row_rate'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def centroid_distance_matrix(classes, centroids):
    matrix = np.full((len(classes), len(classes)), np.nan, dtype=np.float64)
    for i, cls_i in enumerate(classes):
        for j, cls_j in enumerate(classes):
            sim = float((centroids[cls_i] @ centroids[cls_j].t()).item())
            matrix[i, j] = 1.0 - sim
    return matrix


def pairwise_metric_matrix(classes, pair_rows, metric_key):
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    matrix = np.full((len(classes), len(classes)), np.nan, dtype=np.float64)
    for row in pair_rows:
        cls_i = int(row['class_i'])
        cls_j = int(row['class_j'])
        if cls_i not in class_to_idx or cls_j not in class_to_idx:
            continue
        value = row.get(metric_key)
        if value is None:
            continue
        i = class_to_idx[cls_i]
        j = class_to_idx[cls_j]
        matrix[i, j] = float(value)
        matrix[j, i] = float(value)
    return matrix


def confusion_rate_matrix(classes, confusion_rows):
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for row in confusion_rows:
        true_cls = int(row['true_class'])
        pred_cls = int(row['pred_class'])
        if true_cls not in class_to_idx or pred_cls not in class_to_idx:
            continue
        matrix[class_to_idx[true_cls], class_to_idx[pred_cls]] = float(row['row_rate'])
    return matrix


def _finite_bounds(matrix):
    finite = np.asarray(matrix)[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        high = low + 1e-6
    return low, high


def save_heatmap(path_prefix, matrix, classes, title, cbar_label, cmap='magma',
                 vmin=None, vmax=None, annotate_fmt='.2f'):
    if plt is None:
        return {}

    n = len(classes)
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _finite_bounds(matrix)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    width = max(7.0, 0.48 * n + 3.0)
    height = max(6.5, 0.48 * n + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('#e5e7eb')
    im = ax.imshow(matrix, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([str(cls) for cls in classes], rotation=45, ha='right')
    ax.set_yticklabels([str(cls) for cls in classes])
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    ax.set_title(title)

    if n <= 20:
        threshold = (float(vmin) + float(vmax)) / 2.0
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                if not np.isfinite(value):
                    text = 'NA'
                    color = '#111827'
                else:
                    text = format(float(value), annotate_fmt)
                    color = 'white' if float(value) > threshold else '#111827'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)

    fig.tight_layout()
    outputs = {}
    png_path = path_prefix + '.png'
    svg_path = path_prefix + '.svg'
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    outputs['png'] = png_path
    outputs['svg'] = svg_path
    return outputs


def save_class_summary_plot(path_prefix, class_rows):
    if plt is None:
        return {}

    rows = sorted(class_rows, key=lambda row: int(row['class']))
    classes = [int(row['class']) for row in rows]
    nearest_dist = np.asarray([
        float(row['nearest_other_centroid_distance'])
        if row.get('nearest_other_centroid_distance') is not None else np.nan
        for row in rows
    ], dtype=np.float64)
    own_p95 = np.asarray([
        float(row['p95_distance_to_own_centroid'])
        if row.get('p95_distance_to_own_centroid') is not None else np.nan
        for row in rows
    ], dtype=np.float64)
    recall = np.asarray([
        float(row['nearest_centroid_recall'])
        if row.get('nearest_centroid_recall') is not None else np.nan
        for row in rows
    ], dtype=np.float64)
    competitor = [
        str(int(row['nearest_other_centroid_class']))
        if row.get('nearest_other_centroid_class') is not None else 'NA'
        for row in rows
    ]

    x = np.arange(len(classes))
    width = max(9.0, 0.52 * len(classes) + 4.0)
    fig, axes = plt.subplots(2, 1, figsize=(width, 8.0), sharex=True)

    axes[0].bar(x, nearest_dist, color='#2563eb', alpha=0.82, label='Nearest other centroid distance')
    axes[0].plot(x, own_p95, color='#dc2626', marker='o', linewidth=1.6, label='P95 own-centroid distance')
    axes[0].set_ylabel('Cosine distance')
    axes[0].set_title('Class Separation Summary')
    axes[0].legend(loc='best')
    axes[0].grid(axis='y', alpha=0.25)
    for idx, label in enumerate(competitor):
        if np.isfinite(nearest_dist[idx]):
            axes[0].text(idx, nearest_dist[idx], '->' + label, ha='center', va='bottom', fontsize=8)

    axes[1].bar(x, recall, color='#059669', alpha=0.82)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel('Nearest-centroid recall')
    axes[1].set_xlabel('Class')
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(cls) for cls in classes])

    fig.tight_layout()
    outputs = {}
    png_path = path_prefix + '.png'
    svg_path = path_prefix + '.svg'
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    outputs['png'] = png_path
    outputs['svg'] = svg_path
    return outputs


def write_visualizations(out_dir, classes, centroids, pair_rows, confusion_rows, class_rows):
    if plt is None:
        print('[warning] matplotlib is not installed; skipped distance plots.')
        return {}

    outputs = {}
    centroid_matrix = centroid_distance_matrix(classes, centroids)
    pair_median_matrix = pairwise_metric_matrix(classes, pair_rows, 'cosine_distance_median')
    confusion_matrix = confusion_rate_matrix(classes, confusion_rows)

    outputs['centroid_distance_heatmap'] = save_heatmap(
        os.path.join(out_dir, 'centroid_distance_heatmap'),
        centroid_matrix,
        classes,
        'Class Centroid Cosine Distance',
        '1 - cosine similarity (smaller = closer)',
        cmap='magma',
        annotate_fmt='.2f',
    )
    outputs['pairwise_token_median_distance_heatmap'] = save_heatmap(
        os.path.join(out_dir, 'pairwise_token_median_distance_heatmap'),
        pair_median_matrix,
        classes,
        'Median Token-Pair Cosine Distance',
        'Median sampled token distance (smaller = closer)',
        cmap='magma',
        annotate_fmt='.2f',
    )
    outputs['nearest_centroid_confusion_heatmap'] = save_heatmap(
        os.path.join(out_dir, 'nearest_centroid_confusion_heatmap'),
        confusion_matrix,
        classes,
        'Nearest-Centroid Token Assignment',
        'Row-normalized rate',
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        annotate_fmt='.2f',
    )
    outputs['class_distance_summary'] = save_class_summary_plot(
        os.path.join(out_dir, 'class_distance_summary'),
        class_rows,
    )
    return outputs


def silhouette_summary(class_feats, classes, max_tokens, seed):
    if max_tokens <= 0:
        return None
    feats = []
    labels = []
    rng = np.random.RandomState(seed)
    total = sum(class_feats[cls].shape[0] for cls in classes)
    for cls in classes:
        class_np = class_feats[cls].numpy()
        take = class_np.shape[0]
        if total > max_tokens:
            take = max(1, int(round(max_tokens * class_np.shape[0] / total)))
            take = min(take, class_np.shape[0])
        idx = np.arange(class_np.shape[0])
        if take < class_np.shape[0]:
            idx = np.sort(rng.choice(idx, size=take, replace=False))
        feats.append(class_np[idx])
        labels.extend([cls] * idx.shape[0])
    feats = np.concatenate(feats, 0)
    labels = np.asarray(labels)
    if feats.shape[0] <= len(set(labels.tolist())):
        return None
    try:
        score = float(silhouette_score(feats, labels, metric='cosine'))
    except ValueError:
        return None
    return {
        'silhouette_cosine': score,
        'tokens_used': int(feats.shape[0]),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    class_feats, class_meta, split_summary = collect_class_tokens(args)
    classes = sorted(class_feats)
    centroids = class_centroids(class_feats)
    confusion_rows, nearest_extra, nearest_acc = nearest_centroid_outputs(class_feats, centroids, classes)

    class_rows = []
    centroid_tensor = torch.cat([centroids[cls] for cls in classes], 0)
    for cls in classes:
        feats = class_feats[cls]
        own = centroids[cls]
        own_dist = (1.0 - (feats @ own.t()).reshape(-1)).cpu().numpy()
        own_stats = percentile_summary(own_dist)
        centroid_sims = (own @ centroid_tensor.t()).reshape(-1).cpu().numpy()
        own_idx = classes.index(cls)
        centroid_sims[own_idx] = -1e9
        nearest_idx = int(np.argmax(centroid_sims))
        nearest_cls = int(classes[nearest_idx])
        nearest_distance = float(1.0 - centroid_sims[nearest_idx])
        class_rows.append({
            'class': cls,
            'tokens_used': int(feats.shape[0]),
            'positive_tokens_before_sampling': class_meta[cls]['positive_tokens_before_sampling'],
            'mean_similarity_to_own_centroid': float(1.0 - own_stats.get('mean', float('nan'))),
            'mean_distance_to_own_centroid': own_stats.get('mean'),
            'p95_distance_to_own_centroid': own_stats.get('p95'),
            'nearest_other_centroid_class': nearest_cls,
            'nearest_other_centroid_distance': nearest_distance,
            **nearest_extra.get(cls, {}),
        })

    class_stats_path = os.path.join(args.out_dir, 'class_stats.csv')
    centroid_path = os.path.join(args.out_dir, 'centroid_distances.csv')
    pairwise_path = os.path.join(args.out_dir, 'pairwise_token_distances.csv')
    confusion_path = os.path.join(args.out_dir, 'nearest_centroid_confusion.csv')
    summary_path = os.path.join(args.out_dir, 'summary.json')

    write_class_stats(class_stats_path, class_rows)
    write_centroid_distances(centroid_path, classes, centroids)
    pair_rows = write_pairwise_token_distances(pairwise_path, classes, class_feats, args)
    write_confusion(confusion_path, confusion_rows)
    sil = silhouette_summary(class_feats, classes, args.max_silhouette_tokens, args.seed + 31337)
    plot_outputs = {}
    if not args.no_plots:
        plot_outputs = write_visualizations(
            args.out_dir, classes, centroids, pair_rows, confusion_rows, class_rows
        )

    summary = {
        **split_summary,
        'outputs': {
            'class_stats': class_stats_path,
            'centroid_distances': centroid_path,
            'pairwise_token_distances': pairwise_path,
            'nearest_centroid_confusion': confusion_path,
            'plots': plot_outputs,
        },
        'class_meta': class_meta,
        'nearest_centroid_overall_acc': nearest_acc,
        'silhouette': sil,
        'pair_sample_size': args.pair_sample_size,
        'max_tokens_per_class': args.max_tokens_per_class,
        'notes': {
            'cosine_distance': '1 - cosine_similarity; smaller means closer.',
            'centroid_distances': 'Distances between L2-normalized class centroids.',
            'pairwise_token_distances': 'Random sampled token-pair cosine distance quantiles per class pair.',
            'nearest_centroid_confusion': 'Each token assigned to the closest class centroid.',
            'plots': 'Heatmaps are written when matplotlib is available; darker/brighter colors should be read with the colorbar.',
        },
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'[distance] wrote {class_stats_path}')
    print(f'[distance] wrote {centroid_path}')
    print(f'[distance] wrote {pairwise_path}')
    print(f'[distance] wrote {confusion_path}')
    if plot_outputs:
        for plot_name, paths in plot_outputs.items():
            for path in paths.values():
                print(f'[distance] wrote {plot_name}: {path}')
    print(f'[distance] wrote {summary_path}')


if __name__ == '__main__':
    main()
