# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import sys
import argparse
import random
import json
import time
import math
from contextlib import contextmanager
from urllib.parse import unquote

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import numpy as np
import cv2
import openslide
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_curve, roc_auc_score

from modules import inference, load_weak_prompts, execute_tagger, \
        execute_subtyping_tagger, execute_miner

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None

DEFAULT_H5_PIXEL_STEP_THRESHOLD = 16
H5_COORDINATE_KEYS = ('coords', 'coordinates')

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.cuda.empty_cache = lambda: None


# ====================== runtime diagnostics ======================

def get_cpu_memory_mb():
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return rss / 1024 / 1024
        return rss / 1024
    return None


def format_memory_usage():
    cpu_mem = get_cpu_memory_mb()
    parts = []
    if cpu_mem is not None:
        parts.append(f'cpu_rss={cpu_mem:.1f}MB')
    else:
        parts.append('cpu_rss=unknown')

    if torch.cuda.is_available():
        parts.append(f'cuda_alloc={torch.cuda.memory_allocated() / 1024 / 1024:.1f}MB')
        parts.append(f'cuda_reserved={torch.cuda.memory_reserved() / 1024 / 1024:.1f}MB')
        parts.append(f'cuda_max_alloc={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}MB')
    return ', '.join(parts)


def print_memory_usage(label):
    print(f'[memory] {label}: {format_memory_usage()}')


class StageTimer:
    def __init__(self, label):
        self.label = label
        self.times = {}
        self.counts = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def add(self, name, elapsed):
        self.times[name] = self.times.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def merge(self, other):
        for name, elapsed in other.times.items():
            self.times[name] = self.times.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + other.counts.get(name, 0)

    def report(self, total_elapsed=None):
        parts = []
        if total_elapsed is not None:
            parts.append(f'total={total_elapsed:.3f}s')
        parts.extend(
            f'{name}={self.times[name]:.3f}s/{self.counts[name]}x'
            for name in self.times
        )
        timing = ', '.join(parts) if parts else 'total=0.000s'
        print(f'[timing] {self.label}: {timing} | memory: {format_memory_usage()}')


FEATURE_DIM_WARNING_KEYS = set()


def warn_feature_dim_change(context, original_dim, target_dim):
    if original_dim == target_dim:
        return
    key = (context, original_dim, target_dim)
    if key in FEATURE_DIM_WARNING_KEYS:
        return
    FEATURE_DIM_WARNING_KEYS.add(key)
    action = 'truncate' if original_dim > target_dim else 'zero-pad'
    print(f'[warning] {context}: feature dim {original_dim} -> {target_dim}; {action} and L2-normalize for this run.')


def normalize_numpy_features(feats):
    norms = np.linalg.norm(feats, ord=2, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-8)


def align_numpy_feature_dim(feats, target_dim, context):
    feats = feats.astype(np.float32, copy=False)
    if feats.ndim != 2:
        raise ValueError(f'{context}: features must be 2D, got shape {feats.shape}')
    original_dim = feats.shape[1]
    if original_dim == target_dim:
        return feats
    warn_feature_dim_change(context, original_dim, target_dim)
    if original_dim > target_dim:
        feats = feats[:, :target_dim]
    else:
        feats = np.pad(feats, ((0, 0), (0, target_dim - original_dim)), mode='constant')
    return normalize_numpy_features(feats).astype(np.float32, copy=False)


def align_numpy_vector_dim(vec, target_dim, context):
    vec = vec.astype(np.float32, copy=False)
    original_dim = vec.shape[0]
    if original_dim == target_dim:
        return vec
    warn_feature_dim_change(context, original_dim, target_dim)
    if original_dim > target_dim:
        return vec[:target_dim]
    return np.pad(vec, (0, target_dim - original_dim), mode='constant').astype(np.float32, copy=False)


def align_torch_feature_dim(feats, target_dim, context):
    original_dim = feats.shape[1]
    if original_dim == target_dim:
        return feats
    warn_feature_dim_change(context, original_dim, target_dim)
    if original_dim > target_dim:
        feats = feats[:, :target_dim]
    else:
        feats = F.pad(feats, (0, target_dim - original_dim))
    return F.normalize(feats.float(), p=2, dim=1, eps=1e-8)


def align_torch_feature_pair(example_feats, query_feats, context):
    target_dim = min(example_feats.shape[1], query_feats.shape[1])
    example_feats = align_torch_feature_dim(example_feats, target_dim, context + ' examples')
    query_feats = align_torch_feature_dim(query_feats, target_dim, context + ' query')
    return example_feats, query_feats


def get_min_feature_dim(feat_list, context):
    dims = [feat.shape[1] for feat in feat_list if feat.shape[0] > 0]
    if len(dims) == 0:
        raise ValueError(f'{context}: no features available')
    target_dim = min(dims)
    dim_counts = {dim: dims.count(dim) for dim in sorted(set(dims))}
    if len(dim_counts) > 1:
        summary = ', '.join(f'{dim}:{count}' for dim, count in dim_counts.items())
        print(f'[warning] {context}: mixed feature dims ({summary}); aligning all features to {target_dim}.')
    return target_dim


def normalize_score_range(scores):
    if scores.numel() == 0:
        return scores
    score_min = scores.min()
    score_max = scores.max()
    if (score_max - score_min).abs() < 1e-8:
        return torch.ones_like(scores)
    return (scores - score_min) / (score_max - score_min + 1e-8)


def normalize_torch_rows(feats):
    return F.normalize(feats.float(), p=2, dim=1, eps=1e-8)


def apply_context_feature_centering(example_feats, query_feats, mode='none'):
    mode = (mode or 'none').lower()
    if mode == 'none':
        return example_feats, query_feats
    if mode == 'example':
        center = example_feats.mean(0, keepdim=True)
    elif mode == 'query':
        center = query_feats.mean(0, keepdim=True)
    elif mode == 'joint':
        center = torch.cat([example_feats, query_feats], 0).mean(0, keepdim=True)
    else:
        raise ValueError(f'unsupported context centering mode: {mode}')
    return normalize_torch_rows(example_feats - center), normalize_torch_rows(query_feats - center)


def save_numpy_records(path, records):
    if path == '':
        return
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    np.save(path, records)


def binary_conformal_summary(calib_scores, calib_labels, eval_scores, eval_labels, threshold, alpha):
    if alpha <= 0 or alpha >= 1 or calib_scores.numel() == 0 or eval_scores.numel() == 0:
        return None

    calib_prob = torch.sigmoid((calib_scores.float() - float(threshold))).cpu().numpy()
    calib_labels_np = calib_labels.float().cpu().numpy()
    true_prob = np.where(calib_labels_np > 0.5, calib_prob, 1.0 - calib_prob)
    nonconformity = 1.0 - true_prob
    nonconformity = np.sort(nonconformity)
    rank = int(np.ceil((len(nonconformity) + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), len(nonconformity) - 1)
    qhat = float(nonconformity[rank])

    eval_prob = torch.sigmoid((eval_scores.float() - float(threshold))).cpu().numpy()
    eval_labels_np = eval_labels.float().cpu().numpy()
    include_pos = (1.0 - eval_prob) <= qhat
    include_neg = eval_prob <= qhat
    set_size = include_pos.astype(np.float32) + include_neg.astype(np.float32)
    covered = np.where(eval_labels_np > 0.5, include_pos, include_neg)

    return {
        'alpha': float(alpha),
        'qhat': round(qhat, 6),
        'coverage': round(float(covered.mean()), 4),
        'avg_set_size': round(float(set_size.mean()), 4),
        'singleton_rate': round(float((set_size == 1).mean()), 4),
        'empty_rate': round(float((set_size == 0).mean()), 4),
        'both_rate': round(float((set_size == 2).mean()), 4),
    }


def select_high_quality_tokens(feats, importance, budget, anchor_ratio=0.25):
    if feats.shape[0] <= budget:
        return torch.arange(feats.shape[0], device=feats.device)

    budget = min(budget, feats.shape[0])
    anchor_budget = min(max(1, int(round(budget * anchor_ratio))), budget)
    anchors = importance.topk(anchor_budget, largest=True).indices
    if anchor_budget == budget:
        return anchors.sort()[0]

    selected_mask = torch.zeros(feats.shape[0], dtype=torch.bool, device=feats.device)
    selected_mask[anchors] = True
    selected = anchors.tolist()

    anchor_feats = feats[anchors]
    min_distance = (1 - feats @ anchor_feats.t()).min(1).values
    min_distance[selected_mask] = -1
    importance_norm = normalize_score_range(importance)

    while len(selected) < budget:
        diversity = min_distance.clamp_min(0)
        score = importance_norm * (0.35 + diversity)
        score[selected_mask] = -1
        next_idx = int(score.argmax().item())
        if selected_mask[next_idx]:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True
        next_distance = 1 - feats @ feats[next_idx: next_idx + 1].t()
        min_distance = torch.minimum(min_distance, next_distance[:, 0])
        min_distance[selected_mask] = -1

    return torch.tensor(selected, device=feats.device, dtype=torch.long).sort()[0]


def select_centroid_rank_tokens(feats, budget, random_ratio=0.1):
    if feats.shape[0] <= budget:
        return torch.arange(feats.shape[0], device=feats.device)

    budget = min(budget, feats.shape[0])
    centroid = F.normalize(feats.mean(0, keepdim=True), p=2, dim=1, eps=1e-8)
    scores = (feats @ centroid.t()).reshape(-1)
    sorted_idx = scores.argsort(descending=True)

    if budget == 1:
        selected = sorted_idx[:1]
    else:
        positions = torch.linspace(0, sorted_idx.shape[0] - 1, budget, device=sorted_idx.device)
        position_idxs = positions.round().long().unique()
        selected = sorted_idx[position_idxs]
        if selected.shape[0] < budget:
            picked_positions = torch.zeros(sorted_idx.shape[0], dtype=torch.bool, device=sorted_idx.device)
            picked_positions[position_idxs] = True
            fill = sorted_idx[~picked_positions][:budget - selected.shape[0]]
            selected = torch.cat([selected, fill], 0)

    random_keep = int(budget * random_ratio)
    random_keep = min(random_keep, max(budget - 1, 0))
    if random_keep > 0:
        remaining = torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device)
        remaining[selected] = False
        remaining_idxs = remaining.nonzero(as_tuple=False).reshape(-1)
        if remaining_idxs.shape[0] > 0:
            take = min(random_keep, remaining_idxs.shape[0])
            perm = torch.randperm(remaining_idxs.shape[0], device=feats.device)[:take]
            selected = torch.cat([selected[:-take], remaining_idxs[perm]], 0)

    return selected.sort()[0]


def hierarchical_cluster_priority(feats, idxs):
    if idxs.shape[0] <= 1:
        return -1.0
    feats_c = feats[idxs]
    centroid = F.normalize(feats_c.mean(0, keepdim=True), p=2, dim=1, eps=1e-8)
    compactness = (feats_c @ centroid.t()).reshape(-1).mean()
    spread = (1 - compactness).clamp_min(0)
    return float((spread * idxs.shape[0]).item())


def split_hierarchical_cluster(feats, idxs, importance):
    if idxs.shape[0] <= 1:
        return None, None

    feats_c = feats[idxs]
    seed_a = int(idxs[int(importance[idxs].argmax().item())].item())
    sim_a = (feats_c @ feats[seed_a: seed_a + 1].t()).reshape(-1)
    seed_b = int(idxs[int(sim_a.argmin().item())].item())
    sim_b = (feats_c @ feats[seed_b: seed_b + 1].t()).reshape(-1)
    left_mask = sim_a >= sim_b

    if left_mask.all() or (~left_mask).all():
        centroid = F.normalize(feats_c.mean(0, keepdim=True), p=2, dim=1, eps=1e-8)
        scores = (feats_c @ centroid.t()).reshape(-1)
        order = scores.argsort(descending=True)
        half = max(1, idxs.shape[0] // 2)
        left_mask = torch.zeros(idxs.shape[0], dtype=torch.bool, device=idxs.device)
        left_mask[order[:half]] = True

    return idxs[left_mask], idxs[~left_mask]


def select_hierarchical_tokens(feats, importance, budget):
    if feats.shape[0] <= budget:
        return torch.arange(feats.shape[0], device=feats.device)

    budget = min(budget, feats.shape[0])
    importance_norm = normalize_score_range(importance)
    clusters = [
        (hierarchical_cluster_priority(feats, torch.arange(feats.shape[0], device=feats.device)),
         torch.arange(feats.shape[0], device=feats.device))
    ]

    while len(clusters) < budget:
        split_pos = max(
            range(len(clusters)),
            key=lambda pos: (clusters[pos][0], int(clusters[pos][1].shape[0]))
        )
        priority, idxs = clusters.pop(split_pos)
        if priority < 0 or idxs.shape[0] <= 1:
            clusters.append((priority, idxs))
            break

        left, right = split_hierarchical_cluster(feats, idxs, importance_norm)
        if left is None or right is None or left.shape[0] == 0 or right.shape[0] == 0:
            clusters.append((priority, idxs))
            break
        clusters.append((hierarchical_cluster_priority(feats, left), left))
        clusters.append((hierarchical_cluster_priority(feats, right), right))

    selected = []
    for _, idxs in clusters:
        feats_c = feats[idxs]
        centroid = F.normalize(feats_c.mean(0, keepdim=True), p=2, dim=1, eps=1e-8)
        centrality = normalize_score_range((feats_c @ centroid.t()).reshape(-1))
        score = importance_norm[idxs] + centrality
        selected.append(idxs[int(score.argmax().item())])

    selected = torch.stack(selected)
    if selected.shape[0] < budget:
        selected_mask = torch.zeros(feats.shape[0], dtype=torch.bool, device=feats.device)
        selected_mask[selected] = True
        fill = importance_norm.clone()
        fill[selected_mask] = -1
        fill_num = min(budget - selected.shape[0], feats.shape[0] - selected.shape[0])
        if fill_num > 0:
            selected = torch.cat([selected, fill.topk(fill_num).indices], 0)

    return selected[:budget].sort()[0]


def sparsify_reference_tokens(example_feats, example_labels, budget, anchor_ratio=0.25,
                              strategy='quality', random_ratio=0.1):
    if budget <= 0 or example_feats.shape[0] <= budget:
        return example_feats, example_labels
    strategy = strategy.lower()
    if strategy not in ['quality', 'legacy', 'hierarchical']:
        raise ValueError(f'unsupported reference sparsification strategy: {strategy}')

    labels = torch.unique(example_labels)
    valid_labels = [label for label in labels.tolist() if label in [0, 1, 255]]
    label_idxs = {
        label: (example_labels == label).nonzero(as_tuple=False).reshape(-1)
        for label in valid_labels
    }
    total_tokens = sum(idx.shape[0] for idx in label_idxs.values())
    if total_tokens == 0:
        return example_feats, example_labels

    quotas = {}
    selected_total = 0
    for label, idx in label_idxs.items():
        quota = int(round(budget * idx.shape[0] / total_tokens))
        if idx.shape[0] > 0:
            quota = max(1, min(quota, idx.shape[0]))
        quotas[label] = quota
        selected_total += quota

    while selected_total > budget:
        label = max(quotas, key=lambda _: quotas[_])
        if quotas[label] <= 1:
            break
        quotas[label] -= 1
        selected_total -= 1
    while selected_total < budget:
        room = {label: label_idxs[label].shape[0] - quotas[label] for label in quotas}
        label = max(room, key=room.get)
        if room[label] <= 0:
            break
        quotas[label] += 1
        selected_total += 1

    centroids = {}
    for label, idx in label_idxs.items():
        if idx.shape[0] == 0:
            continue
        centroids[label] = F.normalize(example_feats[idx].mean(0, keepdim=True), p=2, dim=1, eps=1e-8)

    keep_idxs = []
    for label in valid_labels:
        idx = label_idxs[label]
        per_label_budget = quotas[label]
        if idx.shape[0] <= per_label_budget:
            keep_idxs.append(idx)
            continue

        feats_l = example_feats[idx]
        if strategy == 'legacy':
            selected = select_centroid_rank_tokens(feats_l, per_label_budget, random_ratio=random_ratio)
        else:
            own_centroid = centroids[label]
            own_score = (feats_l @ own_centroid.t()).reshape(-1)
            other_centroids = [centroids[_] for _ in valid_labels if _ != label and _ in centroids]
            if len(other_centroids) > 0:
                other_centroids = torch.cat(other_centroids, 0)
                other_score = (feats_l @ other_centroids.t()).max(1).values
            else:
                other_score = torch.zeros_like(own_score)

            importance = normalize_score_range(own_score) + normalize_score_range(own_score - other_score)
            if strategy == 'hierarchical':
                selected = select_hierarchical_tokens(feats_l, importance, per_label_budget)
            else:
                selected = select_high_quality_tokens(feats_l, importance, per_label_budget, anchor_ratio=anchor_ratio)
        keep_idxs.append(idx[selected])

    keep_idxs = torch.cat(keep_idxs, 0)
    if keep_idxs.shape[0] > budget:
        keep_idxs = keep_idxs[torch.randperm(keep_idxs.shape[0], device=keep_idxs.device)[:budget]]
    keep_idxs = keep_idxs.sort()[0]
    print(f'[reference] sparsified tokens ({strategy}): {example_feats.shape[0]} -> {keep_idxs.shape[0]}')
    return example_feats[keep_idxs], example_labels[keep_idxs]


# ====================== h5 feature input helpers ======================

def find_h5_files(path):
    if path is None:
        return []
    if os.path.isfile(path) and path.lower().endswith(('.h5', '.hdf5')):
        return [path]
    if not os.path.isdir(path):
        return []
    files = []
    files.extend(glob.glob(os.path.join(path, '*.h5')))
    files.extend(glob.glob(os.path.join(path, '*.hdf5')))
    return sorted(files)


def decode_slide_key(name):
    name = str(name).strip()
    return unquote(name) if '%' in name else name


def h5_slide_stem(h5_path):
    return os.path.splitext(os.path.basename(h5_path))[0]


def h5_slide_key(h5_path):
    return decode_slide_key(h5_slide_stem(h5_path))


def h5_file_map(h5_files):
    out = {}
    for h5_path in h5_files:
        stem = h5_slide_stem(h5_path)
        out[stem] = h5_path
        out[decode_slide_key(stem)] = h5_path
    return out


def get_dataset_slide_entry(dataset_info, slide_name, raw_slide_name=None):
    if slide_name in dataset_info:
        return slide_name
    if raw_slide_name is not None and raw_slide_name in dataset_info:
        dataset_info[slide_name] = dataset_info.pop(raw_slide_name)
        return slide_name
    dataset_info[slide_name] = {}
    return slide_name


def get_pseudo_label(idx, cls_num):
    if cls_num <= 1:
        return idx % 2
    return idx % cls_num + 1


def get_wsi_label_ids(info):
    if 'wsi_labels' in info:
        labels = info['wsi_labels']
        if isinstance(labels, (list, tuple, set)):
            return sorted({int(_) for _ in labels})
        return [int(labels)]
    label = int(info.get('wsi_label', 0))
    return [label] if label > 0 else []


def has_wsi_label(info, cls):
    return int(cls) in set(get_wsi_label_ids(info))


def dataset_is_multilabel(dataset_info, args=None):
    if args is not None and getattr(args, 'multilabel', False):
        return True
    return any('wsi_labels' in info for info in dataset_info.values())


def infer_dataset_class_num(dataset_info):
    max_label = 0
    for info in dataset_info.values():
        labels = get_wsi_label_ids(info)
        if labels:
            max_label = max(max_label, max(labels))
    return max_label


def normalize_multiclass_wsi_labels(dataset_info, class_num):
    if class_num <= 1:
        return

    labels = []
    for info in dataset_info.values():
        if 'wsi_label' not in info:
            continue
        try:
            labels.append(int(info['wsi_label']))
        except (TypeError, ValueError):
            return

    unique_labels = set(labels)
    if unique_labels == set(range(class_num)):
        for info in dataset_info.values():
            info['wsi_label'] = int(info['wsi_label']) + 1
        print(f'[warning] Detected zero-based multiclass wsi labels 0..{class_num - 1}; remapped to 1..{class_num} for PRET.')


def load_dataset_info(args):
    dataset_info = {}
    if args.dataset_info != '' and os.path.exists(args.dataset_info):
        dataset_info = json.load(open(args.dataset_info))

    h5_files = find_h5_files(args.raw_feature_path)
    if h5_files:
        created_or_filled = False
        missing_label_slides = []
        for idx, h5_path in enumerate(h5_files):
            raw_slide_name = h5_slide_stem(h5_path)
            slide_name = get_dataset_slide_entry(dataset_info, h5_slide_key(h5_path), raw_slide_name)
            dataset_info[slide_name]['h5_input'] = True
            if 'fixed_test_set' not in dataset_info[slide_name]:
                dataset_info[slide_name]['fixed_test_set'] = False
            if 'wsi_label' not in dataset_info[slide_name] and 'wsi_labels' not in dataset_info[slide_name]:
                dataset_info[slide_name]['wsi_label'] = get_pseudo_label(idx, args.c)
                dataset_info[slide_name]['pseudo_label'] = True
                created_or_filled = True
                missing_label_slides.append(slide_name)

        if created_or_filled:
            print('[warning] Missing slide labels for h5 inputs. PRET assigned deterministic pseudo labels by file order; metrics are for pipeline smoke tests only.')
            preview = ', '.join(missing_label_slides[:20])
            print(f'[warning] Missing wsi_label/wsi_labels for {len(missing_label_slides)} h5 slide(s): {preview}')
            if len(missing_label_slides) > 20:
                print(f'[warning] ... {len(missing_label_slides) - 20} more h5 slide(s) omitted.')

    if dataset_is_multilabel(dataset_info, args):
        inferred_class_num = infer_dataset_class_num(dataset_info)
        if args.c < inferred_class_num:
            print(
                f'[warning] Inferred {inferred_class_num} classes from wsi_labels; '
                f'overriding --class_num={args.c}.'
            )
            args.c = inferred_class_num
    else:
        normalize_multiclass_wsi_labels(dataset_info, args.c)
    return dataset_info


def make_synthetic_coordinates(num_patches):
    if num_patches == 0:
        return np.empty((0, 2), dtype=np.int32)
    grid_w = int(np.ceil(np.sqrt(num_patches)))
    patch_idxs = np.arange(num_patches, dtype=np.int32)
    return np.stack((patch_idxs % grid_w, patch_idxs // grid_w), axis=1)


def load_h5_feature_file(h5_path):
    with h5py.File(h5_path, 'r') as f:
        if 'features' not in f:
            raise KeyError(f'{h5_path} must contain h5 key: features')
        feats = np.asarray(f['features'], dtype=np.float32)
        coord_key = next((key for key in H5_COORDINATE_KEYS if key in f), None)
        coords = np.asarray(f[coord_key]) if coord_key is not None else None

    if feats.ndim != 2:
        raise ValueError(f'{h5_path}: features must be a 2D array, got shape {feats.shape}')
    if coords is None:
        coords = make_synthetic_coordinates(feats.shape[0])
        keys = '/'.join(H5_COORDINATE_KEYS)
        print(f'[warning] {h5_path} has no {keys} key. PRET generated synthetic row-major patch coordinates.')
    if coords.ndim != 2 or coords.shape[0] != feats.shape[0] or coords.shape[1] < 2:
        raise ValueError(f'{h5_path}: h5 coordinates must have shape (N, >=2), got {coords.shape}')

    norms = np.linalg.norm(feats, ord=2, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-8)
    return feats.astype(np.float32, copy=False), coords


def infer_h5_axis_step(values):
    values = sorted({int(round(float(value))) for value in values})
    diffs = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not diffs:
        return None
    step = diffs[0]
    for diff in diffs[1:]:
        step = math.gcd(step, diff)
    return step if step > 0 else None


def infer_h5_patch_size(coords):
    steps = []
    x_step = infer_h5_axis_step(coords[:, 0])
    y_step = infer_h5_axis_step(coords[:, 1])
    if x_step is not None:
        steps.append(x_step)
    if y_step is not None:
        steps.append(y_step)
    if not steps:
        return None
    return min(steps)


def infer_h5_coordinate_mode(coords, args, context='h5 input'):
    if args.h5_coordinate_mode != 'auto':
        return args.h5_coordinate_mode
    inferred_step = infer_h5_patch_size(coords)
    if inferred_step is not None and inferred_step >= args.h5_pixel_step_threshold:
        mode = 'pixel'
    else:
        mode = 'grid'
    print(f'[info] {context}: inferred h5 coordinate mode {mode} from coordinate step {inferred_step}.')
    return mode


def get_h5_patch_size(args, coords=None, context='h5 input', coordinate_mode=None):
    if args.h5_patch_size > 0:
        return args.h5_patch_size
    coordinate_mode = coordinate_mode or args.h5_coordinate_mode
    if coordinate_mode == 'pixel' and coords is not None:
        inferred = infer_h5_patch_size(coords)
        if inferred is not None:
            print(f'[info] {context}: inferred h5 patch size {inferred} from coordinates.')
            return inferred
    return args.patch_scale


def h5_coord_to_patch_xy(coord, args, patch_size=None, coordinate_mode=None):
    coordinate_mode = coordinate_mode or args.h5_coordinate_mode
    if coordinate_mode == 'pixel':
        patch_size = patch_size if patch_size is not None else get_h5_patch_size(args)
        return int(float(coord[0]) // patch_size), int(float(coord[1]) // patch_size)
    return int(coord[0]), int(coord[1])


def infer_h5_grid_size(coords, args, patch_size=None, coordinate_mode=None):
    patch_xy = [h5_coord_to_patch_xy(coord, args, patch_size, coordinate_mode) for coord in coords]
    if not patch_xy:
        return None
    max_x = max(x for x, _ in patch_xy)
    max_y = max(y for _, y in patch_xy)
    return (int(max_y) + 1, int(max_x) + 1)


def load_h5_patch_labels(info, coords, args, patch_size=None, coordinate_mode=None):
    if 'h5_patch_labels' in info:
        labels = np.load(info['h5_patch_labels']).astype(np.uint8, copy=False)
        if labels.shape[0] != coords.shape[0]:
            raise ValueError(
                f"{info['h5_patch_labels']}: expected {coords.shape[0]} labels, got {labels.shape[0]}"
            )
        return labels

    if 'patch_labels' not in info:
        return np.array([], dtype=np.uint8)

    mask = cv2.imread(info['patch_labels'])
    if mask is None:
        raise ValueError(f"Cannot read patch label mask: {info['patch_labels']}")
    mask = mask[:, :, 0]

    labels = []
    for coord in coords:
        x, y = h5_coord_to_patch_xy(coord, args, patch_size, coordinate_mode)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            labels.append(mask[y, x])
        else:
            labels.append(0)
    return np.asarray(labels, dtype=np.uint8)


def get_wsi_suffix(wsi_path):
    if not os.path.isdir(wsi_path):
        return None
    files = [f for f in os.listdir(wsi_path) if not f.startswith('.')]
    if len(files) == 0:
        return None
    return files[0].split('.')[-1]


def get_wsi_size(wsi_path, slide_name, wsi_suffix, patch_scale):
    if wsi_suffix is None:
        return None
    slide_path = os.path.join(wsi_path, slide_name + '.' + wsi_suffix)
    if not os.path.exists(slide_path):
        return None
    wsi = openslide.OpenSlide(slide_path)
    return (wsi.level_dimensions[0][1] // patch_scale, wsi.level_dimensions[0][0] // patch_scale)


# ====================== collect features and information ======================

def feature_processor(args):
    print('start feature processing ...')
    timer = StageTimer('feature_processor')
    print_memory_usage('feature_processor start')
    io_start = time.perf_counter()
    dataset_info = load_dataset_info(args)
    h5_map = h5_file_map(find_h5_files(args.raw_feature_path))
    os.makedirs(args.dump_features, exist_ok=True)

    for k, v in dataset_info.items():
        if os.path.exists(os.path.join(args.dump_features, k + '.npy')):
            continue

        feats, names, patch_label, wsi_label = [], [], [], -1

        if k in h5_map:
            feats, coords = load_h5_feature_file(h5_map[k])
            h5_coordinate_mode = infer_h5_coordinate_mode(coords, args, context=h5_map[k])
            h5_patch_size = get_h5_patch_size(args, coords, context=h5_map[k], coordinate_mode=h5_coordinate_mode)
            patch_label = load_h5_patch_labels(v, coords, args, h5_patch_size, h5_coordinate_mode)
            h5_grid_size = infer_h5_grid_size(coords, args, h5_patch_size, h5_coordinate_mode)
            names = []
            for i in range(coords.shape[0]):
                x, y = h5_coord_to_patch_xy(coords[i], args, h5_patch_size, h5_coordinate_mode)
                names.append(os.path.join('h5_features', k, f'{x}_{y}.jpeg'))
            info = {'features': feats, 'patch_names': names, 'patch_labels': np.array(patch_label)}
            if 'wsi_label' in v:
                info['wsi_label'] = v['wsi_label']
            if 'wsi_labels' in v:
                info['wsi_labels'] = v['wsi_labels']
            if h5_grid_size is not None:
                info['h5_grid_size'] = h5_grid_size
            info['h5_coordinate_mode'] = h5_coordinate_mode
            info['h5_patch_size'] = h5_patch_size
            np.save(os.path.join(args.dump_features, k + '.npy'), info)
            continue
        
        wsi_label = v.get('wsi_label', 1 if get_wsi_label_ids(v) else 0)

        # patch label as segmentation gt, if any
        if 'patch_labels' in v:
            mask = cv2.imread(v['patch_labels'])[:, :, 0]

        in_dir = os.path.join(args.raw_feature_path, k + '_files')
        in_dir = in_dir if in_dir[-1] != '/' else in_dir[:-1]
        patch_path = in_dir.replace(in_dir.split('/')[-2], 'images')
        ori_dir = sorted([int(_) for _ in os.listdir(patch_path)])[-1]
        patch_path = os.path.join(patch_path, str(ori_dir))

        for f in os.listdir(os.path.join(in_dir, 'x20')):
            name = os.path.join(patch_path, f.replace('.npy', '.jpeg'))
            if os.path.getsize(name) < args.file_min_size:
                continue
            
            # load feature, L2 norm
            feat = np.load(os.path.join(in_dir, 'x20', f)).astype(np.float32, copy=False)
            norm = np.linalg.norm(feat, ord=2, axis=0)
            feat = feat / max(norm, 1e-8)

            # get position for vis and seg
            x, y = f.split('.')[0].split('_')
            x, y = int(x), int(y)
            if 'patch_labels' in v:
                patch_label.append(mask[y, x])

            names.append(name)
            feats.append(feat)
        
        if len(names) == 0:
            continue

        # save patch features, name, patch_labels and wsi_labels for eval
        info = {'features': np.stack(feats, 0), 'patch_names': names, \
            'patch_labels': np.array(patch_label), 'wsi_label': wsi_label}
        if 'wsi_labels' in v:
            info['wsi_labels'] = v['wsi_labels']
        np.save(os.path.join(args.dump_features, k + '.npy'), info)
    
    print('finish feature processing and saving!')
    timer.add('io', time.perf_counter() - io_start)
    timer.report()


# ====================== some util functions ======================

def macro_value(l, n):
    out = []
    for i in range(len(l) // n):
        v = sum(l[i * n: i * n + n]) / n
        out.append(v)
    return out


def get_example_names_at_same_num(all_names, dataset_info, example_num, check_num=False, expected_labels=None):
    record = {}
    for n in all_names:
        labels = get_wsi_label_ids(dataset_info[n])
        if not labels:
            labels = [0]
        for lb in labels:
            if lb not in record:
                record[lb] = []
            record[lb].append(n)

    labels = list(expected_labels) if expected_labels is not None else list(record.keys())
    if check_num:
        shortages = []
        for k in labels:
            candidate_num = len(record.get(k, []))
            if candidate_num < example_num:
                shortages.append(f'{k}: need {example_num}, found {candidate_num}')
        if shortages:
            counts = ', '.join(f'{k}:{len(record[k])}' for k in sorted(record))
            raise ValueError(
                'Insufficient example WSIs for balanced multiclass sampling. '
                f'Missing/short classes: {", ".join(shortages)}. '
                f'Candidate label counts: {counts}. '
                'For slideLabel multiclass runs, make sure every class has at least '
                'example_num non-fixed-test WSI candidates.'
            )

    names = []
    seen = set()
    for k in labels:
        for name in record.get(k, [])[:example_num]:
            if name not in seen:
                names.append(name)
                seen.add(name)

    return names


def label_counts(names, dataset_info):
    counts = {}
    for n in names:
        labels = get_wsi_label_ids(dataset_info[n])
        if not labels:
            labels = [0]
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def format_label_counts(counts):
    return ', '.join(f'{k}:{counts[k]}' for k in sorted(counts))


def patch_labels_for_class(patch_labels, cls, class_num, multilabel=False):
    patch_labels = np.asarray(patch_labels)
    if multilabel:
        if patch_labels.ndim == 2:
            out = np.zeros(patch_labels.shape[0], dtype=np.int64)
            col = int(cls) - 1
            if 0 <= col < patch_labels.shape[1]:
                out[patch_labels[:, col] > 0] = 1
            return out
        return (patch_labels == int(cls)).astype(np.int64)

    if class_num > 1:
        out = np.full(patch_labels.shape[0], 255, dtype=np.int64)
        out[patch_labels == int(cls)] = 1
        out[(patch_labels > 0) & (patch_labels != int(cls))] = 0
        if np.any(patch_labels == 1) and not np.any(patch_labels == int(cls)):
            out[patch_labels == 1] = 0
        return out

    return patch_labels.astype(np.int64, copy=True)


def safe_binary_auc(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))


def select_binary_threshold(val_labels, val_preds, prefer_f1=False):
    labels = np.asarray(val_labels).astype(int)
    preds = np.asarray(val_preds, dtype=float)
    if labels.size == 0:
        return 0.0, float('nan'), float('nan')
    unique = np.unique(labels)
    if unique.size < 2:
        if unique[0] == 1:
            thresh = float(preds.min() - 1e-6)
        else:
            thresh = float(preds.max() + 1e-6)
        acc = float(((preds > thresh).astype(int) == labels).mean())
        f1 = float(f1_score(labels, (preds > thresh).astype(int), zero_division=0))
        return thresh, acc, f1

    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    if thresholds.size == 0:
        thresh = float(np.median(preds))
        pred_labels = (preds > thresh).astype(int)
        return thresh, float((pred_labels == labels).mean()), float(f1_score(labels, pred_labels, zero_division=0))

    accs = np.array([((preds > _).astype(int) == labels).mean() for _ in thresholds])
    if prefer_f1:
        f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        idx = int(np.nanargmax(f1_scores))
    else:
        idx = int(np.nanargmax(accs))
    thresh = float(thresholds[idx])
    pred_labels = (preds > thresh).astype(int)
    return thresh, float(accs[idx]), float(f1_score(labels, pred_labels, zero_division=0))


def multilabel_metrics(y_true, y_score, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)
    per_class_auc = []
    for cls_idx in range(y_true.shape[1]):
        per_class_auc.append(safe_binary_auc(y_true[:, cls_idx], y_score[:, cls_idx]))
    valid_auc = [v for v in per_class_auc if np.isfinite(v)]
    flat_auc = safe_binary_auc(y_true.reshape(-1), y_score.reshape(-1))
    return {
        'acc_exact_match': float(accuracy_score(y_true, y_pred)),
        'acc_hamming': float(1.0 - hamming_loss(y_true, y_pred)),
        'auc_micro': flat_auc,
        'auc_macro': float(np.mean(valid_auc)) if valid_auc else float('nan'),
        'auc_per_class': [float(v) if np.isfinite(v) else None for v in per_class_auc],
        'f1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_samples': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
    }


def format_metric_value(value):
    if value is None:
        return 'nan'
    try:
        if not np.isfinite(value):
            return 'nan'
    except TypeError:
        pass
    return str(round(float(value), 4))


def select_validation_names(rest_names, dataset_info, val_num, balanced=False):
    if val_num <= 0 or len(rest_names) == 0:
        return []

    val_num = min(val_num, len(rest_names))
    if not balanced:
        return rest_names[:val_num]

    grouped = {}
    for n in rest_names:
        labels = get_wsi_label_ids(dataset_info[n])
        label = tuple(labels) if len(labels) > 1 else (labels[0] if labels else 0)
        grouped.setdefault(label, []).append(n)

    if len(grouped) <= 1:
        return rest_names[:val_num]

    labels = list(grouped.keys())
    random.shuffle(labels)
    base = val_num // len(labels)
    quotas = {label: min(base, len(grouped[label])) for label in labels}
    selected = sum(quotas.values())

    while selected < val_num:
        progressed = False
        labels_by_remaining = sorted(
            labels,
            key=lambda label: len(grouped[label]) - quotas[label],
            reverse=True
        )
        for label in labels_by_remaining:
            if selected >= val_num:
                break
            if quotas[label] < len(grouped[label]):
                quotas[label] += 1
                selected += 1
                progressed = True
        if not progressed:
            break

    val_names = []
    for label in labels:
        val_names.extend(grouped[label][:quotas[label]])
    random.shuffle(val_names)
    return val_names


def check_different_patient(example_names, query_candidates, mode='TCGA'):
    out = []
    if mode == 'TCGA':
        for q in query_candidates:
            inside = False
            for g in example_names:
                if mode == 'TCGA':
                    if q[:12] == g[:12]:
                        inside = True
            if not inside:
                out.append(q)

    return out


# post processing via gussain blur
class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(GaussianBlur, self).__init__()
        kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 + (y-kernel_size//2)**2)/(2*sigma**2)),
        (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)
        kernel = np.reshape(kernel, (1, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(torch.from_numpy(kernel).float(), requires_grad=False).cuda()

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.weight.shape[-1]//2)


# ====================== evaluation for multiple tasks, prompts, and shots ======================

def evaluate(args, val_only=False):
    auc_list, f1_list, acc_list, example_list = [], [], [], []
    dataset_info = load_dataset_info(args)
    multilabel = dataset_is_multilabel(dataset_info, args)
    all_names = list(dataset_info.keys())
    multilabel_repeat_metrics = []

    records = {}
    txt_rec = []
    # ====================== repeat experimets n=args.runs ======================

    for i in range(args.runs):
        records['repeat_' + str(i)] = {}
        
        repeat_timer = StageTimer(f'evaluate repeat={i}')
        repeat_start = time.perf_counter()
        split_start = time.perf_counter()
        # ====================== data split ======================

        labeled_names, neg_names, test_names, rest_names = [], [], [], []

        for n in all_names:
            # splitdata, if there is fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)

            else:
                # pick pos from labeled wsi
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']
                    
                    # prompt samplinging (camelyon only)
                    if args.c == 1 and 'CAMELYON' in args.wsi_path:
                        if pn >= 1000 and pn < 3000:
                            labeled_names.append(n)
                    
                    else:
                        labeled_names.append(n)
                
                if args.prompt_type == 'slideLabel':
                    # For multiclass slide-level prompts, every labeled WSI can be an example.
                    if args.c > 1 and n not in labeled_names:
                        labeled_names.append(n)
                
                    # Keep original binary WSI behavior: only add negatives here.
                    # h5-only slide-label datasets have no patch labels, so they need both classes.
                    if args.c == 1 and (
                        dataset_info[n].get('wsi_label') == 0 or
                        (dataset_info[n].get('h5_input', False) and 'pos_patch_num' not in dataset_info[n])
                    ):
                        labeled_names.append(n)
                
                # record neg names to exclude from seg val /test
                if len(get_wsi_label_ids(dataset_info[n])) == 0:
                    neg_names.append(n)

        # shuffle example till each run is different
        while True:
            random.shuffle(labeled_names)

            # randomly select "args.example_num" examples for each class
            # note: for binary tasks 'slideLabel' use N // 2 pos and N // 2 neg
            if args.c > 1 or args.prompt_type == 'slideLabel':
                expected_labels = range(1, args.c + 1) if args.c > 1 else None
                example_i = get_example_names_at_same_num(
                    labeled_names, dataset_info, args.example_num, args.c > 1, expected_labels
                )

            # randomly select "args.example_num" positive examples for binary tasks
            else:
                example_i = labeled_names[:args.example_num]

            # avoid repeat example
            example_i.sort()
            if example_i not in example_list:
                example_list.append(example_i)
                example_names = example_i
                break
        if args.c > 1:
            print('[split] repeat=' + str(i) + ' example label counts: ' + format_label_counts(label_counts(example_names, dataset_info)))

        # split val set out of example and test set
        for n in all_names:
            if n not in example_names and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)

        if args.seg:
            rest_names = []
            for ln in labeled_names:
                if ln not in example_names and ln not in neg_names:
                    rest_names.append(ln)

        # avoid same patients in different split, in-house data is cleaned
        if 'TCGA' in args.wsi_path:
            rest_names = check_different_patient(example_names, rest_names, 'TCGA')

        random.shuffle(rest_names)
        val_num = args.val_num if args.val_ratio < 0 else int(len(rest_names) * args.val_ratio)
        use_balanced_val = getattr(args, 'balanced_val_split', False)
        use_disjoint_split = getattr(args, 'disjoint_val_test_split', False)
        val_names = select_validation_names(rest_names, dataset_info, val_num, balanced=use_balanced_val)
        if use_disjoint_split:
            val_name_set = set(val_names)
            remaining_names = [n for n in rest_names if n not in val_name_set]
        else:
            remaining_names = rest_names
        if args.c > 1 and use_balanced_val:
            print('[split] repeat=' + str(i) + ' balanced val label counts: ' + format_label_counts(label_counts(val_names, dataset_info)))

        # split test set by ratio, if no fixed test set
        if len(test_names) == 0:
            if args.val_ratio < 0:
                test_names = remaining_names[-args.test_num:] if args.test_num > 0 else remaining_names
            else:
                test_names = remaining_names if use_disjoint_split else rest_names[val_num:]
            if len(val_names) + len(test_names) > len(rest_names):
                print('wrong split size !!!')
        else: # take partial test slides for tcga cross races
            random.shuffle(test_names)
            if args.test_num > 0:
                test_names = test_names[:args.test_num]
        
        records['repeat_' + str(i)]['split'] = {'example_names': example_names, 'val_names': val_names, 'test_names': test_names}
        repeat_timer.add('split', time.perf_counter() - split_start)

        # ====================== run for each class ======================

        # for subtyping, use different example for each cls and apply marco metics, other tasks have one class
        multilabel_repeat = {}
        for cls in range(1, args.c + 1):

            # ====================== process example and prompts ======================
            class_timer = StageTimer(f'evaluate repeat={i} class={cls}')

            # load example
            example_feats, example_patch_names, example_labels = [], [], []
            example_feature_names = []
            io_start = time.perf_counter()
            for n in example_names:
                example_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                if multilabel and n in dataset_info:
                    example_wsi_label = 1 if has_wsi_label(dataset_info[n], cls) else 0
                else:
                    example_wsi_label = dataset_info[n].get('wsi_label', example_n.get('wsi_label', 0)) if n in dataset_info else example_n.get('wsi_label', 0)
                example_patch_names = example_patch_names + example_n['patch_names']
                example_feats.append(example_n['features'])
                example_feature_names.append(n)

                # empty patch label for image label or sparse label where there is no offline gt
                if args.prompt_type == 'mask':
                    raw_pl = np.asarray(example_n['patch_labels'])

                    # binary use 0 normal, 1 tumor, while subtyping use 0 other cls, 1 this cls, 255 normal
                    if args.c > 1:
                        if multilabel:
                            pl = patch_labels_for_class(raw_pl, cls, args.c, multilabel=True)
                        elif raw_pl.ndim == 1 and set(np.unique(raw_pl).tolist()).issubset({0, 1}):
                            pl = raw_pl.astype(np.int64, copy=True)
                            pl[pl == 0] = 255
                            if example_wsi_label != cls:
                                pl[pl == 1] = 0
                            else:
                                pl[pl == 1] = 1
                        else:
                            pl = patch_labels_for_class(raw_pl, cls, args.c, multilabel=False)
                    else:
                        pl = raw_pl.astype(np.int64, copy=True)
                    
                else:
                    pl = np.zeros(example_n['features'].shape[0]) - 1
                
                # load weak prompts
                # slideLabel + subtyping is uniqe in pseudo label generation
                if args.prompt_type == "slideLabel" and args.c > 1:
                    if multilabel:
                        pl[:] = 1 if has_wsi_label(dataset_info[n], cls) else 0
                    elif example_wsi_label != cls:
                        pl[:] = 0
                    else:
                        pl[:] = 1

                # for box, RoughMask and binary + slideLabel, -1 is uncertain pos, 0 is normal
                elif args.prompt_type != 'mask' :
                    pl = load_weak_prompts(n, example_wsi_label, args.wsi_path, pl, \
                        example_n['patch_names'], args.prompt_path, args.prompt_type, side=args.patch_scale)
                    
                    #  record wsi label for each patch for later label convert
                    if args.c > 1:
                        pl[pl == 0] = 255
                        if multilabel:
                            pl[pl == -1] = 1 if has_wsi_label(dataset_info[n], cls) else 0
                        else:
                            pl[pl == -1] = 1 if example_wsi_label == cls else 0

                example_labels.append(pl)
            
            example_feature_dim = get_min_feature_dim(example_feats, f'evaluate repeat={i} class={cls} examples')
            example_feats = [
                align_numpy_feature_dim(feat, example_feature_dim, f'example slide {name}')
                for feat, name in zip(example_feats, example_feature_names)
            ]
            example_feats = torch.from_numpy(np.concatenate(example_feats, 0).astype(np.float32, copy=False)).cuda()
            example_labels = torch.from_numpy(np.concatenate(example_labels, 0)).cuda().long()
            class_timer.add('io.examples', time.perf_counter() - io_start)

            if args.dump_pseudo != '':
                vis_info = {'wsi_dir': args.wsi_path, 'vis_dir': os.path.join(args.dump_pseudo, 'vis') + str(args.example_num) + '/' + str(i) + '/' + str(cls), \
                        'mask_dir': os.path.join(args.dump_pseudo, 'pseudo') + str(args.example_num) + '/' + str(i) + '/' + str(cls)}

                split_dir = os.path.join(args.dump_pseudo, 'split') + str(args.example_num)
                split = {'example_names': example_names, 'test_names': test_names, 'val_names': val_names}
                os.makedirs(split_dir, exist_ok=True)
                open(os.path.join(split_dir, str(i) + '.json'), 'w').write(json.dumps(split, indent=4))

            else:
                vis_info = None

            # ====================== apply in-context tagger ======================

            tagger_start = time.perf_counter()
            # assign in-context tags for weak prompts (binary tasks: 1 pos, 0 neg, -1 unknown)
            if args.prompt_type != 'mask' and args.c == 1:
                example_labels = execute_tagger(example_feats, example_labels, example_patch_names, example_names, \
                    vis_info=vis_info, uncertain=args.ignore, topk=args.topk)

            # assign in-context tags for subtyping from slideLabel (255 normal, 254 uncertain, 1 this class, 0 other classes)
            if args.prompt_type == 'slideLabel' and args.c > 1 and not multilabel:
                example_labels = execute_subtyping_tagger(example_feats, example_labels, example_patch_names, \
                    example_names, vis_info=vis_info, uncertain=args.ignore, topk=args.topk)
            
            # subtyping + box / roughMask. Need to process "execute_tagger" twice. 
            # Once for shared bg and this class, another for shared bg and other classes
            if args.prompt_type != 'slideLabel' and args.c > 1 and not multilabel:
                
                if args.prompt_type != 'mask':
                    example_labels_this = example_labels.clone()
                    example_labels_this[example_labels_this == 0] = 254 # ignore other fg
                    example_labels_this[example_labels_this == 255] = 0 # subtyping bg label to binary neg label
                    example_labels_this[example_labels_this == 1] = -1  # this class to undertain to relabel
                    example_labels_this = execute_tagger(example_feats, example_labels_this, example_patch_names, example_names, \
                        vis_info=vis_info, uncertain=args.ignore, topk=args.topk)

                    vis_info = None
                    example_labels_others = example_labels.clone()
                    example_labels_others[example_labels_others == 1] = 254 # ignore this fg
                    example_labels_others[example_labels_others == 0] = -1  # other class to undertain to relabel
                    example_labels_others[example_labels_others == 255] = 0 # subtyping bg label to binary neg label
                    example_labels_others = execute_tagger(example_feats, example_labels_others, example_patch_names, example_names, \
                        vis_info=vis_info, uncertain=args.ignore, topk=args.topk)

                    example_labels[:] = 255 # default bg
                    example_labels[example_labels_this == 1] = 1     # this class
                    example_labels[example_labels_others == 1] = 0   # other class
                    example_labels[example_labels_this == -1] = 254  # ignore in the last
                    example_labels[example_labels_others == -1] = 254# ignore in the last
                    if (example_labels == 255).sum() == 0:
                        example_labels[example_labels_others == 0] = 255
                        example_labels[example_labels_this == 0] = 255
            class_timer.add('tagger', time.perf_counter() - tagger_start)

            sparse_start = time.perf_counter()
            sparsify_strategy = args.reference_sparsify_strategy
            if sparsify_strategy == 'auto':
                sparsify_strategy = 'legacy' if args.c == 1 else 'quality'
            example_feats, example_labels = sparsify_reference_tokens(
                example_feats,
                example_labels,
                args.reference_token_budget,
                args.reference_anchor_ratio,
                strategy=sparsify_strategy,
                random_ratio=args.reference_random_ratio
            )
            class_timer.add('reference_sparse', time.perf_counter() - sparse_start)

            # ====================== predict for test slides (queries)======================

            # predict for test slides, name a test slide as query to avoid confusion with test set
            val_preds, test_preds, val_labels, test_labels = [], [], [], []
            wsi_suffix = get_wsi_suffix(args.wsi_path)
            all_query_names = val_names if val_only else val_names + test_names
            for n in all_query_names:
                io_start = time.perf_counter()
                query_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                query_feats = torch.from_numpy(query_n['features'].astype(np.float32, copy=False)).cuda()
                query_patch_names = query_n['patch_names']
                if multilabel and n in dataset_info:
                    label = int(has_wsi_label(dataset_info[n], cls))
                else:
                    label = dataset_info[n].get('wsi_label', query_n.get('wsi_label', 0)) if n in dataset_info else query_n.get('wsi_label', 0)
                    if args.c > 1:
                        label = int(label == cls)
                
                class_timer.add('io.query', time.perf_counter() - io_start)
                example_feats_for_query, query_feats = align_torch_feature_pair(
                    example_feats, query_feats, f'evaluate repeat={i} class={cls} query {n}'
                )
                example_feats_for_query, query_feats = apply_context_feature_centering(
                    example_feats_for_query, query_feats, args.context_centering
                )
                # ====================== discriminative instance miner for subtyping ======================

                # use fg patches for subtyping
                #if args.c > 1 and not args.seg and args.vis_path == '': # vis wo fg
                if args.c > 1 and not args.seg and not multilabel:
                    miner_start = time.perf_counter()
                    query_feats, query_patch_names = execute_miner(example_feats_for_query[example_labels == 255], \
                        query_feats, query_patch_names, uncertain=args.ignore_query)
                    class_timer.add('miner', time.perf_counter() - miner_start)

                # ====================== inference, including classifier, aggregator, post processer ======================

                infer_start = time.perf_counter()
                size = get_wsi_size(args.wsi_path, n, wsi_suffix, args.patch_scale)
                if size is None and 'h5_grid_size' in query_n:
                    size = tuple(int(v) for v in query_n['h5_grid_size'])
                vis_info = None
                
                sm = GaussianBlur(7, 3) if args.seg else None #  seg pred
                wsi_pred, patch_pred, patch_pred_list = inference(args, example_feats_for_query, example_labels, example_patch_names, \
                    query_feats, query_patch_names, size, args.top_instance, vis_info, smooth=sm)
                class_timer.add('infer', time.perf_counter() - infer_start)

                if patch_pred != None and args.vis_path != '' and n in test_names:
                    os.makedirs(args.vis_path, exist_ok=True)
                    np.save(os.path.join(args.vis_path, n + '_' + str(cls) + '.npy'), patch_pred.cpu().numpy())
                
                if args.seg:
                    pred = torch.as_tensor(patch_pred_list)
                    label = torch.tensor(query_n['patch_labels'])
                else:
                    pred = torch.tensor([wsi_pred])
                    label = torch.tensor([label])

                if n in val_names:
                    val_preds.append(pred)
                    val_labels.append(label)
                else:
                    test_preds.append(pred)
                    test_labels.append(label)
                del query_feats, query_n, pred, label
                torch.cuda.empty_cache()

            # ====================== process validation set and assign label ======================

            # Evaluate on the val set to make sure qualified results for application
            # Val set also guidances to select prediction threshod, f1 for seg. acc for others
            validation_start = time.perf_counter()
            val_preds = torch.cat(val_preds).cpu()
            val_labels = torch.cat(val_labels)
            val_auc = safe_binary_auc(val_labels.numpy(), val_preds.numpy())
            if not val_only:
                test_preds = torch.cat(test_preds).cpu()
                test_labels = torch.cat(test_labels)
            
            thresh, best_acc_score, _ = select_binary_threshold(
                val_labels.numpy(), val_preds.numpy(), prefer_f1=args.seg
            )

            if val_only:
                preds = val_preds
                thresh_preds = (val_preds > thresh).float()
                labels = val_labels
            else:
                preds = test_preds
                thresh_preds = (test_preds > thresh).float()
                labels = test_labels

            acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
            label_pos = labels.sum().clamp_min(1)
            pred_pos = thresh_preds.sum().clamp_min(1)
            rec = ((thresh_preds * labels).sum() / label_pos).cpu().item()
            pre = ((thresh_preds * labels).sum() / pred_pos).cpu().item()
            auc = safe_binary_auc(labels.numpy(), preds.numpy())
            if (rec + pre) != 0:
                f1 = rec * pre * 2 / (rec + pre)
            else:
                f1 = 0
            auc_list.append(auc)
            f1_list.append(f1)
            acc_list.append(acc)
            if not val_only:
                conformal = binary_conformal_summary(
                    val_preds, val_labels, preds, labels, thresh, args.conformal_alpha
                )
                s = 'class:' + str(cls) + ' val auc:' + str(round(val_auc, 4)) + ', test auc:' + str(round(auc, 4)) + \
                    ', val acc: ' + str(round(best_acc_score, 4)) + ', test f1: ' + str(round(f1, 4)) + \
                    ', test acc: ' + str(round(acc, 4))
                print(s)
                txt_rec.append(s)
                if conformal is not None:
                    conformal_s = 'class:' + str(cls) + ' conformal coverage:' + str(conformal['coverage']) + \
                        ', avg set size:' + str(conformal['avg_set_size'])
                    print(conformal_s)
                    txt_rec.append(conformal_s)
                records['repeat_' + str(i)]['results_cls' + str(cls)] = {'val_auc': round(val_auc, 4), 'test_auc': round(auc, 4), \
                        'val_acc': round(best_acc_score, 4), 'test_f1': round(f1, 4), 'test_acc': round(acc, 4)}
                records['repeat_' + str(i)]['pred_cls' + str(cls)] = {'labels': labels.cpu().tolist(), \
                        'logits': preds.cpu().tolist(), 'preds': thresh_preds.cpu().tolist()}
                if conformal is not None:
                    records['repeat_' + str(i)]['conformal_cls' + str(cls)] = conformal
                if multilabel and not args.seg:
                    multilabel_repeat[cls] = {
                        'threshold': float(thresh),
                        'names': list(val_names if val_only else test_names),
                        'logits': preds.cpu().numpy().reshape(-1),
                        'labels': labels.cpu().numpy().astype(int).reshape(-1),
                        'preds': thresh_preds.cpu().numpy().astype(int).reshape(-1),
                    }

            class_timer.add('validation', time.perf_counter() - validation_start)
            repeat_timer.merge(class_timer)
        if multilabel and not args.seg and multilabel_repeat:
            names = multilabel_repeat[1]['names']
            y_score = np.stack([multilabel_repeat[cls]['logits'] for cls in range(1, args.c + 1)], axis=1)
            y_true = np.stack([multilabel_repeat[cls]['labels'] for cls in range(1, args.c + 1)], axis=1)
            thresholds = np.array([multilabel_repeat[cls]['threshold'] for cls in range(1, args.c + 1)])
            y_pred = (y_score > thresholds.reshape(1, -1)).astype(int)
            metrics = multilabel_metrics(y_true, y_score, y_pred)
            multilabel_repeat_metrics.append(metrics)
            preview = [
                {'name': names[idx], 'pred': y_pred[idx].astype(int).tolist(), 'label': y_true[idx].astype(int).tolist()}
                for idx in range(min(10, len(names)))
            ]
            records['repeat_' + str(i)]['multilabel'] = {
                'class_thresholds': {str(cls): float(multilabel_repeat[cls]['threshold']) for cls in range(1, args.c + 1)},
                'names': names,
                'labels': y_true.astype(int).tolist(),
                'logits': y_score.tolist(),
                'preds': y_pred.astype(int).tolist(),
                'metrics': metrics,
                'preview': preview,
            }
            s = (
                'multilabel test acc_exact_match: ' + format_metric_value(metrics['acc_exact_match']) +
                ', acc_hamming: ' + format_metric_value(metrics['acc_hamming']) +
                ', auc_micro: ' + format_metric_value(metrics['auc_micro']) +
                ', auc_macro: ' + format_metric_value(metrics['auc_macro']) +
                ', f1_micro: ' + format_metric_value(metrics['f1_micro']) +
                ', f1_macro: ' + format_metric_value(metrics['f1_macro']) +
                ', f1_samples: ' + format_metric_value(metrics['f1_samples'])
            )
            print(s)
            txt_rec.append(s)
            for item in preview:
                p = f"multilabel preview {item['name']}: pred={item['pred']}, label={item['label']}"
                print(p)
                txt_rec.append(p)
        del example_feats
        torch.cuda.empty_cache()
        repeat_timer.report(total_elapsed=time.perf_counter() - repeat_start)

    # ====================== count and record results ======================

    if multilabel and multilabel_repeat_metrics:
        metric_keys = ['acc_exact_match', 'acc_hamming', 'auc_micro', 'auc_macro', 'f1_micro', 'f1_macro', 'f1_samples']
        mean_metrics = {}
        for key in metric_keys:
            values = np.array([m[key] for m in multilabel_repeat_metrics], dtype=float)
            mean_metrics[key + '_mean'] = float(np.nanmean(values))
            mean_metrics[key + '_std'] = float(np.nanstd(values))
        s = (
            'multilabel mean acc_exact_match: ' + format_metric_value(mean_metrics['acc_exact_match_mean']) +
            ', acc_hamming: ' + format_metric_value(mean_metrics['acc_hamming_mean']) +
            ', auc_micro: ' + format_metric_value(mean_metrics['auc_micro_mean']) +
            ', auc_macro: ' + format_metric_value(mean_metrics['auc_macro_mean']) +
            ', f1_micro: ' + format_metric_value(mean_metrics['f1_micro_mean']) +
            ', f1_macro: ' + format_metric_value(mean_metrics['f1_macro_mean']) +
            ', f1_samples: ' + format_metric_value(mean_metrics['f1_samples_mean'])
        )
        print(s)
        txt_rec.append(s)
        records['mean'] = mean_metrics
        records['text_records'] = txt_rec
        return round(mean_metrics['auc_macro_mean'], 4), records

    auc_mean = np.array(auc_list).mean()
    macro_auc = macro_value(auc_list, args.c)
    auc_std = np.array(macro_auc).std()
    f1_mean = np.array(f1_list).mean()
    macro_f1 = macro_value(f1_list, args.c)
    f1_std = np.array(macro_f1).std()
    acc_mean = np.array(acc_list).mean()
    macro_acc = macro_value(acc_list, args.c)
    acc_std = np.array(macro_acc).std()
    s = 'auc mean: ' + str(round(auc_mean, 4)) + ', auc std: ' + str(round(auc_std, 4)) + \
        ', f1 mean: ' + str(round(f1_mean, 4)) + ', f1 std: ' + str(round(f1_std, 4)) + \
        ', acc mean: ' + str(round(acc_mean, 4)) + ', acc std: ' + str(round(acc_std, 4))
    print(s)
    txt_rec.append(s)

    records['mean'] = {'auc_mean': round(auc_mean, 4), 'auc_std': round(auc_std, 4), 'auc_values': macro_auc, \
            'f1_mean': round(f1_mean, 4), 'f1_std': round(f1_std, 4), 'f1_values': macro_f1, \
            'acc_mean': round(acc_mean, 4), 'acc_std': round(acc_std, 4), 'acc_values': macro_acc}
    records['text_records'] = txt_rec
     
    return round(auc_mean, 4), records


# ====================== evaluation for baseline methods ======================

def evaluate_baseline(args, mode):
    auc_list, f1_list, acc_list, example_list = [], [], [], []
    aucroc = torchmetrics.AUROC(task='binary', num_classes=1)
    dataset_info = load_dataset_info(args)
    multilabel = dataset_is_multilabel(dataset_info, args)
    all_names = dataset_info.keys()

    # skip invalid wsis
    temp = []
    for _ in all_names:
        if os.path.exists(os.path.join(args.dump_features, _ + '.npy')):
            temp.append(_)
    all_names = temp

    # ====================== run for each class ======================

    records = {}
    txt_rec = []
    for i in range(args.runs):
        records['repeat_' + str(i)] = {}

        repeat_timer = StageTimer(f'baseline {mode} repeat={i}')
        split_start = time.perf_counter()
        # ====================== data split ======================

        # data split
        labeled_names, neg_names, test_names, rest_names = [], [], [], []

        for n in all_names:
            # splitdata, if there is fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)

            else:
                # pick pos from labeled wsi
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']

                    # prompt samplinging (camelyon only)
                    if args.c == 1 and 'CAMELYON' in args.wsi_path:
                        if pn >= 1000 and pn < 3000:
                            labeled_names.append(n)

                    else:
                        labeled_names.append(n)

                if args.prompt_type == 'slideLabel':
                    # For multiclass slide-level prompts, every labeled WSI can be an example.
                    if args.c > 1 and n not in labeled_names:
                        labeled_names.append(n)

                    # Keep original binary WSI behavior: only add negatives here.
                    # h5-only slide-label datasets have no patch labels, so they need both classes.
                    if args.c == 1 and (
                        dataset_info[n].get('wsi_label') == 0 or
                        (dataset_info[n].get('h5_input', False) and 'pos_patch_num' not in dataset_info[n])
                    ):
                        labeled_names.append(n)

                # record neg names to exclude from seg val /test
                if len(get_wsi_label_ids(dataset_info[n])) == 0:
                    neg_names.append(n)

        # shuffle example till each run is different
        while True:
            random.shuffle(labeled_names)

            # randomly select "args.example_num" examples for each class
            # note: for binary tasks 'slideLabel' use N // 2 pos and N // 2 neg
            if args.c > 1 or args.prompt_type == 'slideLabel':
                expected_labels = range(1, args.c + 1) if args.c > 1 else None
                example_i = get_example_names_at_same_num(
                    labeled_names, dataset_info, args.example_num, args.c > 1, expected_labels
                )

            # randomly select "args.example_num" positive examples for binary tasks
            else:
                example_i = labeled_names[:args.example_num]

            # avoid repeat example
            example_i.sort()
            if example_i not in example_list:
                example_list.append(example_i)
                example_names = example_i
                break
        if args.c > 1:
            print('[split] repeat=' + str(i) + ' example label counts: ' + format_label_counts(label_counts(example_names, dataset_info)))

        # split val set out of example and test set
        for n in all_names:
            if n not in example_names and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)

        if args.seg:
            rest_names = []
            for ln in labeled_names:
                if ln not in example_names and ln not in neg_names:
                    rest_names.append(ln)

        if 'TCGA' in args.wsi_path:
            rest_names = check_different_patient(example_names, rest_names, 'TCGA')
        if 'LN' in args.wsi_path:
            rest_names = check_different_patient(example_names, rest_names, 'LN')

        random.shuffle(rest_names)
        val_num = args.val_num if args.val_ratio < 0 else int(len(rest_names) * args.val_ratio)
        use_balanced_val = getattr(args, 'balanced_val_split', False)
        use_disjoint_split = getattr(args, 'disjoint_val_test_split', False)
        val_names = select_validation_names(rest_names, dataset_info, val_num, balanced=use_balanced_val)
        if use_disjoint_split:
            val_name_set = set(val_names)
            remaining_names = [n for n in rest_names if n not in val_name_set]
        else:
            remaining_names = rest_names
        if args.c > 1 and use_balanced_val:
            print('[split] baseline ' + mode + ' repeat=' + str(i) + ' balanced val label counts: ' + format_label_counts(label_counts(val_names, dataset_info)))

        # split test set by ratio, if no fixed test set
        if len(test_names) == 0:
            if args.val_ratio < 0:
                test_names = remaining_names[-args.test_num:] if args.test_num > 0 else remaining_names
            else:
                test_names = remaining_names if use_disjoint_split else rest_names[val_num:]
            if len(val_names) + len(test_names) > len(rest_names):
                print('wrong split size !!!')
        else: # take partial test slides for tcga cross races
            random.shuffle(test_names)
            if args.test_num > 0:
                test_names = test_names[:args.test_num]

        records['repeat_' + str(i)]['split'] = {'example_names': example_names, 'val_names': val_names, 'test_names': test_names}
        repeat_timer.add('split', time.perf_counter() - split_start)
        repeat_timer.report()

        # ====================== run for each class ======================

        # for subtyping, use different example for each cls and apply marco metics
        for cls in range(1, args.c + 1):

            # load example
            example_feats, example_labels = [], []
            pos_feats, neg_feats = [], []

            # ====================== process example ======================

            for n in example_names:
                example_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                if multilabel and n in dataset_info:
                    example_wsi_label = 1 if has_wsi_label(dataset_info[n], cls) else 0
                else:
                    example_wsi_label = dataset_info[n].get('wsi_label', example_n.get('wsi_label', 0)) if n in dataset_info else example_n.get('wsi_label', 0)

                # empty patch label for image label or sparse label where there is no offline gt
                if args.prompt_type == 'mask':
                    raw_pl = np.asarray(example_n['patch_labels'])

                    # binary use 0 normal, 1 tumor, while subtyping use 0 other cls, 1 this cls, 255 normal
                    if args.c > 1:
                        if multilabel:
                            pl = patch_labels_for_class(raw_pl, cls, args.c, multilabel=True)
                        elif raw_pl.ndim == 1 and set(np.unique(raw_pl).tolist()).issubset({0, 1}):
                            pl = raw_pl.astype(np.int64, copy=True)
                            pl[pl == 0] = 255
                            if example_wsi_label != cls:
                                pl[pl == 1] = 0
                            else:
                                pl[pl == 1] = 1
                        else:
                            pl = patch_labels_for_class(raw_pl, cls, args.c, multilabel=False)
                    else:
                        pl = raw_pl.astype(np.int64, copy=True)

                else:
                    pl = np.zeros(example_n['features'].shape[0]) - 1

                # load sparse label
                # slideLabel + subtyping is uniqe in pseudo label generation
                if args.prompt_type == "slideLabel" and args.c > 1:
                    if multilabel:
                        pl[:] = 1 if has_wsi_label(dataset_info[n], cls) else 0
                    elif example_wsi_label != cls:
                        pl[:] = 0
                    else:
                        pl[:] = 1

                # for box, RoughMask and binary + slideLabel, -1 is uncertain pos, 0 is normal
                elif args.prompt_type != 'mask' :
                    pl = load_weak_prompts(n, example_wsi_label, args.wsi_path, pl, \
                        example_n['patch_names'], args.prompt_path, args.prompt_type, side=args.patch_scale)

                    #  record wsi label for each patch for later label convert
                    if args.c > 1:
                        pl[pl == 0] = 255
                        if multilabel:
                            pl[pl == -1] = 1 if has_wsi_label(dataset_info[n], cls) else 0
                        else:
                            pl[pl == -1] = 1 if example_wsi_label == cls else 0
                
                if 'prototype' in mode:
                    pos_feats.append(example_n['features'][(pl != 0) * (pl != 255)])
                    neg_feats.append(example_n['features'][pl == 0])

                if 'knn' in mode:

                    if args.prompt_type != 'slideLabel':
                        feat_fg = example_n['features'][(pl != 0) * (pl != 255)]
                        if feat_fg.shape[0] != 0:
                            if 'mean' in mode:
                                example_feats.append(feat_fg.mean(0, keepdims=True))
                            elif 'max' in mode:
                                example_feats.append(feat_fg.max(0, keepdims=True))
                            example_labels.append(1)
                        
                        feat_bg = example_n['features'][pl == 0]
                        if feat_bg.shape[0] != 0:
                            if 'mean' in mode:
                                example_feats.append(feat_bg.mean(0, keepdims=True))
                            elif 'max' in mode:
                                example_feats.append(feat_bg.max(0, keepdims=True))
                            example_labels.append(0)
                    else:
                        feat = example_n['features']
                        if 'mean' in mode:
                            example_feats.append(feat.mean(0, keepdims=True))
                        elif 'max' in mode:
                            example_feats.append(feat.max(0, keepdims=True))
                        example_labels.append(1 if example_wsi_label == cls else 0)

            if 'prototype' in mode:
                example_labels = [1, 0]
                prototype_feature_dim = get_min_feature_dim(pos_feats + neg_feats, f'baseline {mode} repeat={i} class={cls} examples')
                pos_feats = [
                    align_numpy_feature_dim(feat, prototype_feature_dim, 'baseline positive prototype examples')
                    for feat in pos_feats if feat.shape[0] > 0
                ]
                neg_feats = [
                    align_numpy_feature_dim(feat, prototype_feature_dim, 'baseline negative prototype examples')
                    for feat in neg_feats if feat.shape[0] > 0
                ]
                pos_feats = np.concatenate(pos_feats, 0)
                neg_feats = np.concatenate(neg_feats, 0)

                if 'simple_shot' in mode:
                    mean_feat = np.concatenate([pos_feats, neg_feats], 0).mean(0)
                    pos_feats -= mean_feat
                    pos_feats = pos_feats.mean(0, keepdims=True)
                    pos_feats = pos_feats / np.linalg.norm(pos_feats, 2, 1, keepdims=True)
                    neg_feats -= mean_feat
                    neg_feats = neg_feats.mean(0, keepdims=True)
                    neg_feats = neg_feats / np.linalg.norm(neg_feats, 2, 1, keepdims=True)
                    example_feats = [pos_feats, neg_feats]
                else:
                    example_feats = [pos_feats.mean(0, keepdims=True), neg_feats.mean(0, keepdims=True)]

            example_feature_dim = get_min_feature_dim(example_feats, f'baseline {mode} repeat={i} class={cls} examples')
            example_feats = [
                align_numpy_feature_dim(feat, example_feature_dim, f'baseline {mode} example vector')
                for feat in example_feats
            ]
            example_feats = torch.tensor(np.concatenate(example_feats, 0)).cuda()
            example_labels = torch.tensor(example_labels).cuda()

            # ====================== inference for test slides ======================

            # predict query
            val_preds, test_preds, val_labels, test_labels = [], [], [], []
            all_query_names = val_names + test_names
            for n in all_query_names:
                query_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                query_feats = torch.tensor(query_n['features'].astype(np.float32, copy=False)).cuda()
                query_patch_names = query_n['patch_names']
                if multilabel and n in dataset_info:
                    label = int(has_wsi_label(dataset_info[n], cls))
                else:
                    query_wsi_label = dataset_info[n].get('wsi_label', query_n.get('wsi_label', 0)) if n in dataset_info else query_n.get('wsi_label', 0)
                    if args.c > 1:
                        label = query_wsi_label == cls
                    else:
                        label = query_wsi_label

                example_feats_for_query, query_feats = align_torch_feature_pair(
                    example_feats, query_feats, f'baseline {mode} repeat={i} class={cls} query {n}'
                )
                example_feats_for_query, query_feats = apply_context_feature_centering(
                    example_feats_for_query, query_feats, args.context_centering
                )
                if 'prototype' in mode:
                    if 'simple_shot' in mode:
                        mean_feat_for_query = align_numpy_vector_dim(
                            mean_feat, query_feats.shape[1], 'baseline simple_shot mean feature'
                        )
                        query_feats -= torch.tensor(mean_feat_for_query).cuda()
                        query_feats = query_feats / torch.linalg.norm(query_feats, 2, 1, keepdims=True)

                    topk = min(args.top_instance, query_feats.shape[0])
                    prob = query_feats @ example_feats_for_query[0]
                    wsi_pred = prob.topk(topk)[0].mean()

                    if args.vis_path != '' or args.seg:
                        wsi_suffix = get_wsi_suffix(args.wsi_path)
                        size = get_wsi_size(args.wsi_path, n, wsi_suffix, args.patch_scale)
                        if size is None and 'h5_grid_size' in query_n:
                            size = tuple(int(v) for v in query_n['h5_grid_size'])
                        if size is None:
                            print('skip visualization or segmentation without WSI files')
                            patch_pred = None
                            patch_pred_list = None
                            continue
                        patch_pred = torch.zeros(size).cuda() + 255
                        idx_in_map = []
                        for pi, pn in enumerate(query_patch_names):
                            x, y = pn.split('/')[-1].split('.')[0].split('_')
                            try:
                                patch_pred[int(y), int(x)] = prob[pi]
                                idx_in_map.append(int(y) * patch_pred.shape[1] + int(x))
                            except:
                                if len(idx_in_map) != 0:
                                    idx_in_map.append(idx_in_map[-1])
                                else:
                                    idx_in_map.append(0)
                                continue

                        if args.vis_path != '' and n in test_names:
                            os.makedirs(args.vis_path, exist_ok=True)
                            np.save(os.path.join(args.vis_path, n + '_' + str(cls) + '.npy'), patch_pred.cpu().numpy())

                        if args.seg:
                            smooth = GaussianBlur(7, 3)
                            fg = patch_pred != 255
                            bg = fg == False
                            smooth_pred = patch_pred.clone()
                            smooth_pred[bg] = smooth_pred[fg].mean() # replace 255 to mean value before smoothing
                            smooth_pred = smooth(smooth_pred.reshape(1, 1, smooth_pred.shape[0], smooth_pred.shape[1]))[0,0]
                            patch_pred[fg] = smooth_pred[fg]
                            patch_pred_list = patch_pred.reshape(-1)[idx_in_map]

                elif 'knn' in mode:
                    if 'mean' in mode:
                        query_feats = query_feats.mean(0)
                    elif 'max' in mode:
                        query_feats = query_feats.max(0)[0]
                    else:
                        print('false eval mode')
                   
                    pos_example_feats = example_feats_for_query[example_labels == 1]
                    neg_example_feats = example_feats_for_query[example_labels == 0]
                    wsi_pred = (pos_example_feats @ query_feats).topk(min(5, pos_example_feats.shape[0]))[0].mean() - \
                            (neg_example_feats @ query_feats).topk(min(5, neg_example_feats.shape[0]))[0].mean()
                    
                else:
                    print('false eval mode')

                if args.seg:
                    pred = torch.as_tensor(patch_pred_list)
                    label = torch.tensor(query_n['patch_labels'])
                else:
                    pred = torch.tensor([wsi_pred])
                    label = torch.tensor([label])

                if n in val_names:
                    val_preds.append(pred)
                    val_labels.append(label)
                else:
                    test_preds.append(pred)
                    test_labels.append(label)

            # ====================== process validation set and assign label ======================

            # search a threshold to predict label on val set for fair comparisions
            val_preds = torch.cat(val_preds).cpu()
            val_labels = torch.cat(val_labels)
            val_auc = aucroc(val_preds, val_labels).item()
            test_preds = torch.cat(test_preds).cpu()
            test_labels = torch.cat(test_labels)
            
            precisions, recalls, thresholds = precision_recall_curve(val_labels.numpy(), val_preds.numpy())
            accs = np.array([((val_preds > _).float() == val_labels).sum() / val_labels.shape[0] for _ in thresholds])
            if args.seg:
                f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_acc_score = accs[best_f1_score_index]
                thresh = thresholds[best_f1_score_index]
            else:
                best_acc_score = np.max(accs[np.isfinite(accs)])
                best_acc_score_index = np.argmax(accs[np.isfinite(accs)])
                thresh = thresholds[best_acc_score_index]

            preds = test_preds
            thresh_preds = (test_preds > thresh).float()
            labels = test_labels
            acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
            label_pos = labels.sum().clamp_min(1)
            pred_pos = thresh_preds.sum().clamp_min(1)
            rec = ((thresh_preds * labels).sum() / label_pos).cpu().item()
            pre = ((thresh_preds * labels).sum() / pred_pos).cpu().item()
            auc = aucroc(preds, labels).item()
            f1 = rec * pre * 2 / (rec + pre) if rec + pre > 0 else 0

            auc_list.append(auc)
            f1_list.append(f1)
            acc_list.append(acc)

            conformal = binary_conformal_summary(
                val_preds, val_labels, preds, labels, thresh, args.conformal_alpha
            )
            s = 'class:' + str(cls) + ' val auc:' + str(round(val_auc, 4)) + ', test auc:' + str(round(auc, 4)) + ', val acc: ' \
                 + str(round(best_acc_score, 4)) + ', test f1: ' + str(round(f1, 4)) + ', test acc: ' + str(round(acc, 4))
            print(s)
            txt_rec.append(s)
            if conformal is not None:
                conformal_s = 'class:' + str(cls) + ' conformal coverage:' + str(conformal['coverage']) + \
                    ', avg set size:' + str(conformal['avg_set_size'])
                print(conformal_s)
                txt_rec.append(conformal_s)
            records['repeat_' + str(i)]['results_cls' + str(cls)] = {'val_auc': round(val_auc, 4), 'test_auc': round(auc, 4), \
                    'val_acc': round(best_acc_score, 4), 'test_f1': round(f1, 4), 'test_acc': round(acc, 4)}
            records['repeat_' + str(i)]['pred_cls' + str(cls)] = {'labels': labels.cpu().tolist(), \
                    'logits': preds.cpu().tolist(), 'preds': thresh_preds.cpu().tolist()}
            if conformal is not None:
                records['repeat_' + str(i)]['conformal_cls' + str(cls)] = conformal

    # ====================== count and record results ======================

    auc_mean = np.array(auc_list).mean()
    macro_auc = macro_value(auc_list, args.c)
    auc_std = np.array(macro_auc).std()
    f1_mean = np.array(f1_list).mean()
    macro_f1 = macro_value(f1_list, args.c)
    f1_std = np.array(macro_f1).std()
    acc_mean = np.array(acc_list).mean()
    macro_acc = macro_value(acc_list, args.c)
    acc_std = np.array(macro_acc).std()

    s = 'auc mean: ' + str(round(auc_mean, 4)) + ', auc std: ' + str(round(auc_std, 4)) + \
        ', f1 mean: ' + str(round(f1_mean, 4)) + ', f1 std: ' + str(round(f1_std, 4)) + \
        ', acc mean: ' + str(round(acc_mean, 4)) + ', acc std: ' + str(round(acc_std, 4))
    print(s)
    txt_rec.append(s)

    records['mean'] = {'auc_mean': round(auc_mean, 4), 'auc_std': round(auc_std, 4), 'auc_values': macro_auc, \
            'f1_mean': round(f1_mean, 4), 'f1_std': round(f1_std, 4), 'f1_values': macro_f1, \
            'acc_mean': round(acc_mean, 4), 'acc_std': round(acc_std, 4), 'acc_values': macro_acc}
    records['text_records'] = txt_rec

    return records


# ====================== the main function ======================

if __name__ == '__main__':

    # ====================== arg parser ======================

    parser = argparse.ArgumentParser('Multiple Instance Prompting')
    parser.add_argument('--mode', default='search', type=str, help="update: update features, inference: process query only, \
        eval: load processed features for evaluate, default: update and test")

    # hyper-params
    parser.add_argument('--topk', default=40, type=int, help='Number of top patchs to take')
    parser.add_argument('--top_instance', default=1, type=int, help='Number of top patchs to take')
    parser.add_argument('--temperature', default=10, type=float, help='Temperature for sample reweights')
    parser.add_argument('--related_thresh', default=0.88, type=float, help='cosine similarity threshold to select related patchs')
    parser.add_argument('--example_num', default=3, type=int, help='number of wsi for init example')
    parser.add_argument('--multiple_num', type=int, nargs='+', default=None, help='multi example num')
    parser.add_argument('--reference_token_budget', default=0, type=int,
        help='maximum number of reference tokens kept after tagger; 0 keeps all tokens')
    parser.add_argument('--reference_sparsify_strategy', default='auto',
        choices=['auto', 'quality', 'legacy', 'hierarchical'],
        help='reference token sparsification strategy; auto uses legacy for binary and quality for multi-class')
    parser.add_argument('--reference_anchor_ratio', default=0.25, type=float,
        help='fraction of sparse reference budget reserved for strongest anchor tokens before diversity selection')
    parser.add_argument('--reference_random_ratio', default=0.1, type=float,
        help='fraction of legacy sparse reference budget reserved for random diversity')
    parser.add_argument('--similarity_aggregation', default='mean', choices=['mean', 'softmax', 'adaptive'],
        help='training-free query/reference similarity reducer; mean reproduces original PRET')
    parser.add_argument('--similarity_temperature', default=10.0, type=float,
        help='temperature for softmax/adaptive top-k similarity aggregation')
    parser.add_argument('--adaptive_min_topk', default=1, type=int,
        help='minimum number of top-k neighbors retained by adaptive similarity aggregation')
    parser.add_argument('--adaptive_window', default=0.6, type=float,
        help='adaptive similarity keeps neighbors within this fraction of the top-k score spread')
    parser.add_argument('--context_centering', default='none', choices=['none', 'example', 'query', 'joint'],
        help='training-free feature centering before query inference to reduce feature-domain shift')
    parser.add_argument('--spatial_smooth_strength', default=0.0, type=float,
        help='blend strength for coordinate-neighborhood smoothing of query patch logits')
    parser.add_argument('--spatial_smooth_radius', default=1, type=int,
        help='Manhattan radius for coordinate-neighborhood smoothing of query patch logits')
    parser.add_argument('--spatial_feature_weight', default=0.0, type=float,
        help='optional feature-similarity weight for spatial smoothing neighbors')
    parser.add_argument('--conformal_alpha', default=0.0, type=float,
        help='if >0, report validation-calibrated conformal prediction-set statistics at this alpha')

    # dataset information and settings
    parser.add_argument('--raw_feature_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--wsi_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--dump_features', default=None, help='Path where to save features')
    parser.add_argument('--dump_pseudo', default='', help='Path where to save pseudo, vis and data split')
    parser.add_argument('--dump_records', default='', help='Path to save records (json file)')
    parser.add_argument('--vis_path', default='', help='Path where to save heatmap')
    parser.add_argument('--dataset_info', default='/path/to/data_list_gt_and_split', type=str, help='json file recording dataset info')
    parser.add_argument('--patch_scale', default=512, type=int, help='patch size in 40x for anno loading')
    parser.add_argument('--h5_coordinate_mode', default='auto', choices=['auto', 'grid', 'pixel'],
        help='interpret h5 coordinates as patch-grid indices, level-0 pixel top-left coordinates, or auto-detect from coordinate step')
    parser.add_argument('--h5_pixel_step_threshold', default=DEFAULT_H5_PIXEL_STEP_THRESHOLD, type=int,
        help='auto mode treats h5 coordinate step >= this value as pixel coordinates; smaller steps are patch-grid coordinates')
    parser.add_argument('--h5_patch_size', default=0, type=int,
        help='level-0 patch size for h5 pixel coordinates; 0 infers from h5 coordinates when possible')
    parser.add_argument('--file_min_size', default=5000, type=int, help='skip background and patches with a few content')
    parser.add_argument('--c', '--class_num', dest='c', default=1, type=int, help='number of classes; use 1 for binary screening and >1 for multi-class subtyping')
    parser.add_argument('--multilabel', default=False, action='store_true',
        help='evaluate WSI labels as multi-hot vectors; enabled automatically when dataset_info has wsi_labels')
    parser.add_argument('--seg', default=False, action='store_true', help='True to evaluate segmentation task (f1 = dice)')

    # for weak prompts
    parser.add_argument('--prompt_type', default='mask', help='prompttation type')
    parser.add_argument('--prompt_path', default='', help='path of prompttation xml file')
    parser.add_argument('--ignore', default=0, type=float, help='degree to ignore uncertain example (during generating example)')
    parser.add_argument('--ignore_query', default=0.3, type=float, help='degree to ignore uncertain foreground query (subtyping only)')

    # test settings
    parser.add_argument('--seed', default=1024, type=int, help='for the reproduce of data split')
    parser.add_argument('--runs', default=5, type=int, help='number of test times')
    parser.add_argument('--val_num', default=100, type=int, help='number of validation WSIs')
    parser.add_argument('--test_num', default=129, type=int, help='number of test WSIs')
    parser.add_argument('--val_ratio', default=-1, type=float, help='split val test via ratio to replace specific number')
    parser.add_argument('--balanced_val_split', default=False, action='store_true',
        help='balance validation slides by class label; default off to preserve original PRET split semantics')
    parser.add_argument('--disjoint_val_test_split', default=False, action='store_true',
        help='remove validation slides before selecting test slides; default off to preserve original PRET split semantics')
    parser.add_argument('--seed_torch_sampling', default=False, action='store_true',
        help='also seed torch random sampling; default off to preserve original PRET subtyping sampler behavior')
    args = parser.parse_args()
    if args.h5_pixel_step_threshold <= 0:
        parser.error('--h5_pixel_step_threshold must be positive')

    random.seed(args.seed)
    if args.seed_torch_sampling:
        torch.manual_seed(args.seed)
    os.makedirs(args.dump_features, exist_ok=True)
    print_memory_usage('startup')

    # collect features and information
    feature_processor(args)

    # ====================== Execute different modes ======================
    
    # evaluat with given hyper-parameters (in deployment)
    if args.mode == 'eval':
        print(args)
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            random.seed(args.seed)
            args.example_num = p
            res, rec = evaluate(args)
            records[str(p) + '-shot'] = rec

        save_numpy_records(args.dump_records, records)
    
    # run baselines
    if args.mode == 'baselines':
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            args.example_num = p
            records[str(p) + '-shot'] = {}

            # segmentation need patch predictions, knn is conducted on wsi-level
            if not args.seg and args.vis_path == '':
                print('mode: knn_mean, example ' + str(args.example_num))
                random.seed(args.seed)
                rec_knn_mean = evaluate_baseline(args, 'knn_mean')
                records[str(p) + '-shot']['knn_mean'] = rec_knn_mean
            
                print('mode: knn_max, example ' + str(args.example_num))
                random.seed(args.seed)
                rec_knn_max = evaluate_baseline(args, 'knn_max')
                records[str(p) + '-shot']['knn_max'] = rec_knn_max
            
            print('mode: prototype, example ' + str(args.example_num))
            random.seed(args.seed)
            rec_proto = evaluate_baseline(args, 'prototype')
            records[str(p) + '-shot']['prototype'] = rec_proto

            print('mode: prototype_simple_shot, example ' + str(args.example_num))
            random.seed(args.seed)
            rec_simp = evaluate_baseline(args, 'prototype_simple_shot')
            records[str(p) + '-shot']['simple_Shot'] = rec_simp

        save_numpy_records(args.dump_records, records)

    # run in val-test set with hyperparameter search
    if args.mode == 'default': 

        pseudo = args.dump_pseudo
        args.dump_pseudo = ''

        # speed up param search
        if args.c > 1:
            ori_runs = args.runs
            ori_val_num = args.val_num
            args.runs=3
            args.val_num=50
        
        # search for parameters (in extended data figure 10)
        if args.c > 1:
            v, t = 0, 0
            for p in [1000, 2000, 3000, 4000, 5000]:
                print('searching top_instance, param: ' + str(p))
                random.seed(args.seed) # validate params without influence from sampling
                args.top_instance = p
                res, _ = evaluate(args, val_only=True)
                if res > v:
                    v = res
                    t = p
            args.top_instance = t
            print('params: top_instance, searched threshold: ' + str(t) + ', mean:' + str(v))

        v, t = 0, 0
        for p in [0, 0.02, 0.04, 0.06, 0.08]:
            print('searching ignore, param: ' + str(p))
            random.seed(args.seed)
            args.ignore = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.ignore = t
        print('params: ignore, searched threshold: ' + str(t) + ', mean:' + str(v))

        if args.c > 1:
            v, t = 0, 0
            for p in [0.1, 0.15, 0.2, 0.25, 0.3]:
                print('searching ignore-query, param: ' + str(p))
                random.seed(args.seed)
                args.ignore_query = p
                res, _ = evaluate(args, val_only=True)
                if res > v:
                    v = res
                    t = p
            args.ignore_query = t
            print('params: ignore-query, searched threshold: ' + str(t) + ', mean:' + str(v))

        v, t = 0, 0
        for p in [20, 30, 40, 50, 60]:
            print('searching topk, param: ' + str(p))
            random.seed(args.seed)
            args.topk = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.topk = t
        print('params: topk, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [0.86, 0.87, 0.88, 0.89, 0.9]:
            print('searching related_thresh, param: ' + str(p))
            random.seed(args.seed)
            args.related_thresh = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.related_thresh = t
        print('params: related_thresh, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [1, 5, 10, 20, 30]:
            print('searching temperature, param: ' + str(p))
            random.seed(args.seed)
            args.temperature = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.temperature = t
        print('params: temperature, searched threshold: ' + str(t) + ', mean:' + str(v))
        
        # eval with searched params and test influence of example number
        if args.c > 1:
            args.runs = ori_runs
            args.val_num = ori_val_num
        args.dump_pseudo = pseudo

        print(args)
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            random.seed(args.seed)
            args.example_num = p
            res, rec = evaluate(args)
            records[str(p) + '-shot'] = rec
        
        # save results
        save_numpy_records(args.dump_records, records)
