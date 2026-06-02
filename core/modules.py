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

import os, math, copy, math
import numpy as np
import openslide
import torch
import cv2

SIMILARITY_EXAMPLE_CHUNK = int(os.environ.get('PRET_EXAMPLE_CHUNK', 4096))
SIMILARITY_QUERY_CHUNK = int(os.environ.get('PRET_QUERY_CHUNK', 4096))
ATTENTION_QUERY_CHUNK = int(os.environ.get('PRET_ATTENTION_QUERY_CHUNK', 512))


# ====================== prompt loader ======================

def load_weak_prompts(fn, wsi_label, wsi_dir, patch_labels, patch_names, anno_dir, anno_type, side=512):

    # img label assign 0 for all neg
    if anno_type == "slideLabel":
        if wsi_label == 0:
            patch_labels[:] = 0
        else:
            patch_labels[:] = -1
    
    else:
        # load positions
        pos = []
        for pn in patch_names:
            x, y = pn.split('/')[-1].split('.')[0].split('_')
            x, y = int(x), int(y)
            pos.append([y, x])
        pos = np.array(pos)
   
        # load xml anno
        s = open(os.path.join(anno_dir, fn + '.xml')).read()
        tks = s.split('<Annotation Id="')[1:]
     
        if anno_type == "box":
            boxes = []
            for tk in tks:
                if tk[0] == '2':
                    for b in tk.split('<Region Id=')[1:]:
                        ps = b.split('<Vertex X="')
                        x_list, y_list = [], []
                        for p in ps[1:]:
                            x_list.append(int(float(p.split('"')[0]) / side + 0.5))
                            y_list.append(int(float(p.split('"')[2]) / side + 0.5))
                        x1, x2 = min(x_list), max(x_list)
                        y1, y2 = min(y_list), max(y_list)
                        boxes.append([x1, y1, x2, y2])
        
            for i in range(pos.shape[0]):
                p = pos[i]
                in_box = False
                for b in boxes:
                    if p[0] >= b[1] and p[0] <= b[3] and p[1] >= b[0] and p[1] <= b[2]:
                        in_box = True
                if not in_box:
                    patch_labels[i] = 0

        elif anno_type == 'roughMask':
            slide = openslide.OpenSlide(os.path.join(wsi_dir, fn + '.svs'))
            w, h = slide.level_dimensions[0]
            if (w % side) != 0 or (h % side) != 0:
                w += side - w % side
                h += side - h % side

            mid_scale = side // 32 # keep anno details, original img is too large to process
            resize_scale = side // mid_scale
            out = np.zeros((h // mid_scale, w // mid_scale, 3)).astype('uint8') # patch level gt

            roi_contours = []
            for tk in tks:
                if tk[0] == '3':
                    for roi in tk.split('<Region Id="')[1:]:
                        points = []
                        for p in roi.split(' X="')[1:]:
                            _ = p.split('" Y="')
                            points.append([int(float(_[0]) / mid_scale + 0.5), int(float(_[1].split('"')[0]) / mid_scale + 0.5)])
                        roi_contours.append(np.array(points))
            
            # resize by keep max value (label=1 if 1 in the window)
            out = cv2.fillPoly(out, roi_contours, [0, 0, 1])
            out = out[:, :, -1]
            out = out.reshape(out.shape[0] // resize_scale, resize_scale, out.shape[1] // resize_scale, resize_scale)
            out = out.max(1)
            out = out.max(2)

            for i in range(pos.shape[0]):
                if out[pos[i][0], pos[i][1]] == 0:
                    patch_labels[i] = 0
             
        else :
             print('wrong anno_type')
              
    return patch_labels
             

# ====================== some util functions ======================

def _reduce_topk_scores(best, aggregation='mean', softmax_temperature=10.0,
    adaptive_min_k=1, adaptive_window=0.6):
    aggregation = (aggregation or 'mean').lower()
    if aggregation == 'mean':
        return best.mean(0)

    if aggregation == 'softmax':
        weights = (best * softmax_temperature).softmax(0)
        return (weights * best).sum(0)

    if aggregation == 'adaptive':
        if best.shape[0] == 1:
            return best[0]
        adaptive_min_k = max(1, min(int(adaptive_min_k), best.shape[0]))
        spread = (best[0] - best[-1]).clamp_min(1e-8).reshape(1, -1)
        keep = (best[0:1] - best) <= (spread * adaptive_window)
        keep[:adaptive_min_k] = True
        masked = best.masked_fill(~keep, -1e9)
        weights = (masked * softmax_temperature).softmax(0).masked_fill(~keep, 0)
        weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-8)
        return (weights * best).sum(0)

    raise ValueError(f'unsupported similarity aggregation: {aggregation}')


def compute_similarity(query, example, topk=40, aggregation='mean', softmax_temperature=10.0,
    adaptive_min_k=1, adaptive_window=0.6):
    """Return per-query similarity without materializing example x query.

    The default aggregation matches the original PRET behavior: average the top-k
    reference similarities for each query token. Optional softmax/adaptive modes are
    training-free research extensions that emphasize the most relevant references.
    """
    if query.shape[0] == 0 or example.shape[0] == 0:
        return torch.zeros(query.shape[0], device=query.device)

    k = min(topk, example.shape[0]) if topk > 0 else -1
    scores_out = []
    with torch.no_grad():
        for q_start in range(0, query.shape[0], SIMILARITY_QUERY_CHUNK):
            query_chunk = query[q_start: q_start + SIMILARITY_QUERY_CHUNK]

            if k > 0:
                best = None
                for e_start in range(0, example.shape[0], SIMILARITY_EXAMPLE_CHUNK):
                    example_chunk = example[e_start: e_start + SIMILARITY_EXAMPLE_CHUNK]
                    scores = example_chunk @ query_chunk.t()
                    scores = scores.topk(min(k, scores.shape[0]), dim=0)[0]
                    best = scores if best is None else torch.cat([best, scores], 0).topk(k, dim=0)[0]
                    del scores
                scores_out.append(_reduce_topk_scores(
                    best, aggregation=aggregation, softmax_temperature=softmax_temperature,
                    adaptive_min_k=adaptive_min_k, adaptive_window=adaptive_window
                ))
            else:
                summed = torch.zeros(query_chunk.shape[0], device=query.device)
                count = 0
                for e_start in range(0, example.shape[0], SIMILARITY_EXAMPLE_CHUNK):
                    example_chunk = example[e_start: e_start + SIMILARITY_EXAMPLE_CHUNK]
                    scores = example_chunk @ query_chunk.t()
                    summed += scores.sum(0)
                    count += example_chunk.shape[0]
                    del scores
                scores_out.append(summed / max(count, 1))

    return torch.cat(scores_out, 0)


def parse_patch_xy(name):
    try:
        x, y = os.path.basename(name).split('.')[0].split('_')[:2]
        return int(x), int(y)
    except (TypeError, ValueError):
        return None


def spatially_smooth_logits(query_logits, query_patch_names, radius=1, strength=0.0,
    query_feats=None, feature_weight=0.0):
    if strength <= 0 or radius <= 0 or len(query_patch_names) <= 1:
        return query_logits

    coords = [parse_patch_xy(name) for name in query_patch_names]
    if any(coord is None for coord in coords):
        return query_logits

    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}
    out = query_logits.clone()
    strength = max(0.0, min(float(strength), 1.0))
    radius = int(radius)

    for idx, (x, y) in enumerate(coords):
        neighbor_idxs, weights = [], []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist = abs(dx) + abs(dy)
                if dist > radius:
                    continue
                n_idx = coord_to_idx.get((x + dx, y + dy))
                if n_idx is None:
                    continue
                weight = math.exp(-dist / max(radius, 1))
                if query_feats is not None and feature_weight > 0:
                    sim = torch.clamp(query_feats[idx] @ query_feats[n_idx], min=0).item()
                    weight *= math.exp(feature_weight * sim)
                neighbor_idxs.append(n_idx)
                weights.append(weight)
        if len(neighbor_idxs) <= 1:
            continue
        idx_tensor = torch.tensor(neighbor_idxs, device=query_logits.device, dtype=torch.long)
        weight_tensor = torch.tensor(weights, device=query_logits.device, dtype=query_logits.dtype)
        local = (query_logits[idx_tensor] * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)
        out[idx] = (1 - strength) * query_logits[idx] + strength * local
    return out


def aggregate_query_logits(query_feats, query_logits, top_instance, related_thresh, temperature):
    """Vectorized attention aggregation over top query patches."""
    top_query_num = min(top_instance, query_logits.shape[0])
    top_query_idxs = query_logits.topk(top_query_num)[1]

    query_feats_t = query_feats.t()
    wsi_pred_list = []
    for start in range(0, top_query_num, ATTENTION_QUERY_CHUNK):
        idxs = top_query_idxs[start: start + ATTENTION_QUERY_CHUNK]
        sim_scores = query_feats[idxs] @ query_feats_t
        related_mask = sim_scores > related_thresh
        empty_rows = related_mask.sum(1) == 0
        if empty_rows.any():
            # Match original PRET slicing behavior: sim_idxs[-0:] selects all patches.
            related_mask[empty_rows] = True
        masked_scores = sim_scores.masked_fill(~related_mask, -1e9)
        weights = (masked_scores * temperature).softmax(1)
        wsi_pred_list.append((weights * query_logits.reshape(1, -1)).sum(1))
    return torch.cat(wsi_pred_list, 0).mean()


# Kept for compatibility with older callers; large paths should use compute_similarity.
def low_memory_matrix_multiply(A, B, max_size=20000):
    na,  nb = A.shape[0], B.shape[-1]
    sim = torch.FloatTensor(na, nb)

    for i in range(math.ceil(na / max_size)):
        for j in range(math.ceil(nb / max_size)):
            i1, i2 = i * max_size, (i + 1) * max_size
            j1, j2 = j * max_size, (j + 1) * max_size
            sim[i1: i2, j1: j2] = (A[i1: i2, :] @ B[:, j1: j2]).cpu()

    return sim


def topk_low_memory(inp, n, dim):
    scores, idxs = [], []
    offset = 0
    chunk_num = max(1, inp.shape[dim] // 10000)
    for i in inp.chunk(chunk_num, dim):
        score, idx = i.cuda().topk(min(n, i.shape[dim]), dim)
        scores.append(score)
        idxs.append(idx + offset)
        offset += i.shape[dim]

    scores = torch.cat(scores, dim)
    scores, idxs2 = scores.topk(min(n, scores.shape[dim]), dim)
    idxs = torch.cat(idxs, dim).transpose(0, dim) # ori index
    idxs2 = idxs2.transpose(0, dim) # idx after reduction
    out_idxs = []
    for i in range(idxs2.shape[-1]):
        out_idxs.append(idxs[idxs2[:, i], i])
    out_idxs = torch.stack(out_idxs, -1).transpose(0, dim)

    return scores, out_idxs


# large query number lead to "CUDA error: an illegal memory access was encountered"
# rare WSI (patch > 20000) topk == topk * math.ceil(num / 20000)
def topk_low_memory_(inp, n):
    scores, idxs = [], []
    for i in range(math.ceil(inp.shape[1] / 20000)):
        a, b = topk_low_memory(inp[:, i * 20000: i * 20000 + 20000], n, 0)
        scores.append(a)
        idxs.append(b)

    return torch.cat(scores, 1), torch.cat(idxs, 1)


# ====================== tagging visualization ======================

# vis_dir visualizes heatmap and pseudo_label with ori image, mask_dir saves the pseudo_label
def vis_heat(score, label, pos, f, wsi_dir, vis_dir, mask_dir, side=512):
    f = f if '.svs' in f else f.replace('.xml', '.svs')
    if not os.path.exists(os.path.join(wsi_dir, f)):
        f = f.replace('.svs', '.tif')
    wsi = openslide.OpenSlide(os.path.join(wsi_dir, f))
    w, h = wsi.level_dimensions[0]
    w = math.ceil(w / 512)
    h = math.ceil(h / 512)
    heat = np.zeros((h, w))
    label_map = np.zeros((h, w)) + 253 # 255 normal, 254 uncertain, 253 defualt, subtyping (0 neg, 1 pos), binary(0 noraml, 1 pos)

    for i in range(score.shape[0]):
        y, x = pos[i]
        heat[y, x] = score[i]
        lb = label[i]
        if lb == -1:
            lb = 254
        label_map[y, x] = lb

    # only vis wsi with low resolution to avoid out of memory
    if len(wsi.level_dimensions) > 1:
        scale = len(wsi.level_dimensions) - 1 #3
        img = np.array(wsi.read_region((0, 0), scale, wsi.level_dimensions[scale]))[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fg = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC) > 0
        heat = cv2.applyColorMap((heat * 255).astype('uint8'), cv2.COLORMAP_JET) * 0.4
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        vis_heat = img.copy()
        vis_heat[fg] = img[fg] * 0.6 + heat[fg]

        vis_label = copy.deepcopy(label_map)
        if 255 in vis_label: # subtyping
            vis_label[vis_label == 255] = -1 # normal to temp label
            vis_label[(vis_label < 253) * (vis_label >= 0)] = 255 # all pos (0 ,1, ...)
            vis_label[vis_label == -1] = 0
            vis_label[vis_label == 253] = 0 # default
            vis_label[vis_label == 254] = 128 # uncertain
        else:
            vis_label[vis_label == 253] = 0
            vis_label[vis_label == 254] = 128
            vis_label[vis_label == 1] = 255
        vis_label = vis_label.astype('uint8')
        vis_label = cv2.applyColorMap((vis_label), cv2.COLORMAP_JET) * 0.4
        vis_label = cv2.resize(vis_label, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img[fg] = img[fg] * 0.6 + vis_label[fg]
        vis_label = img

        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(vis_dir, f.split('.')[0] + '_score.jpg'), vis_heat)
        cv2.imwrite(os.path.join(vis_dir, f.split('.')[0] + '_label.jpg'), vis_label)

    label_map = label_map.astype('uint8')
    cv2.imwrite(os.path.join(mask_dir, f.split('.')[0] + '.png'), label_map)


# ====================== instance miner for subtyping ======================

def execute_miner(neg_example_feats, feats, names, topk=40, uncertain=0.2):
    sim = compute_similarity(feats, neg_example_feats, topk=topk)
    ukn = torch.ones(len(names)) == 1
    fg = torch.zeros(len(names)).long()
    fg = basic_tagger(sim, ukn, fg, uncertain, positive=False)
    fg = fg == 1
    
    if fg.sum() == 0:
        return feats, names

    out_names = []
    for i in fg.nonzero()[:, 0].cpu().numpy():
        out_names.append(names[i])
    return feats[fg], out_names


# ====================== basic tagger (algorithm 1) ======================

def basic_tagger(ukn_sim, ukn_mask, label, uncertain, positive=False):
    thresh, _ = cv2.threshold((ukn_sim.cpu().numpy() * 255).astype('uint8'), \
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff = (ukn_sim.max() - ukn_sim.min()) * uncertain
    low_t = float(thresh) / 255 - diff
    high_t = float(thresh) / 255 + diff

    ukn_label = label[ukn_mask]
    if positive:
        ukn_label[ukn_sim > low_t] = 1
        ukn_label[ukn_sim < high_t] = 0
    else:
        ukn_label[ukn_sim < low_t] = 1
        ukn_label[ukn_sim > high_t] = 0
    ukn_label[(ukn_sim <= high_t) * (ukn_sim >= low_t)] = -1
    label[ukn_mask] = ukn_label

    return label


def normalize_similarity_values(values):
    if values.numel() == 0:
        return values
    return (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)


# ====================== in-context tagger (algorithm 2) ======================

# binary classification via sparse annotations (slideLabel, box, roughMask)
def execute_tagger(feats, labels, patch_names, wsi_names, \
        vis_info=None, uncertain=0.1, topk=40, sampling_size=-1):

    # assign init label for each wsi
    # record uncertain positive and normal patch idx
    info_dic = {}
    pos_idx, neg_idx = [], []
    for n in wsi_names:
        pos, idx = [], []
        for i, pn in enumerate(patch_names):
            if n in pn:
                x, y = pn.split('/')[-1].split('.')[0].split('_')
                pos.append([int(y), int(x)])
                idx.append(i)
        pos = np.array(pos)
        info_dic[n] = {'idx': idx, 'pos': pos}

        ukn_mask = labels[idx] == -1
        if True in ukn_mask:
            sim_ukn = compute_similarity(feats[idx][ukn_mask], feats[labels == 0], topk=topk)
            sim_ukn = (sim_ukn - sim_ukn.min()) / (sim_ukn.max() - sim_ukn.min()).clamp_min(1e-8)
            labels_ukn = torch.zeros(sim_ukn.shape[0]).cuda().long()
            labels_ukn = basic_tagger(sim_ukn, labels_ukn==0, labels_ukn, uncertain, positive=False)
            idx_ukn = [idx[_] for _ in ukn_mask.nonzero()[:, 0]]
            pos_idx.extend([idx_ukn[_] for _ in (labels_ukn == 1).nonzero()[:, 0]])
            neg_idx.extend([idx_ukn[_] for _ in (labels_ukn == 0).nonzero()[:, 0]])
            neg_idx.extend([idx[_] for _ in (labels[idx] == 0).nonzero()[:, 0]])
        else:
            neg_idx.extend(idx)

    # assign label via dataset-level pos and neg
    for n in wsi_names:
        idx_n = info_dic[n]['idx']
        if -1 in labels[idx_n]:
            feats_n = feats[idx_n]
            sim_pos = compute_similarity(feats_n, feats[pos_idx], topk=topk)
            sim_neg = compute_similarity(feats_n, feats[neg_idx], topk=topk)
            score = sim_pos - sim_neg
            score = torch.clamp(score, -0.5, 0.5) + 0.5 # norm to 0-1 for vis via clamp

            labels_n = torch.zeros(feats_n.shape[0]).cuda().long()
            labels_n = basic_tagger(score, labels_n==0, labels_n, uncertain, True)
            labels[idx_n] = labels_n

            if vis_info != None:
                vis_heat(score, labels_n, info_dic[n]['pos'], n + '.svs', vis_info['wsi_dir'], \
                    vis_info['vis_dir'], vis_info['mask_dir'], side=512)
    
    return labels


# ====================== in-context tagger for subtyping (algorithm 2) ======================

# 1.dataset similarity 2.init pseudo in each wsi 3.refine via dataset pseduo
def execute_subtyping_tagger(feats, labels, patch_names, wsi_names, \
        vis_info=None, uncertain=0.1, topk=40, sampling_size=40000):
    
    # step1 similarity cross wsi-level label
    pos, neg = labels == 1, labels == 0
    pos_count = int(pos.sum().item())
    neg_count = int(neg.sum().item())
    if pos_count == 0 or neg_count == 0:
        unique_labels, unique_counts = torch.unique(labels.detach().cpu(), return_counts=True)
        label_counts = ', '.join(
            f'{int(label.item())}:{int(count.item())}'
            for label, count in zip(unique_labels, unique_counts)
        )
        raise ValueError(
            'Subtyping tagger needs both positive (1) and negative (0) example tokens, '
            f'but got pos={pos_count}, neg={neg_count}. Label counts: {label_counts}. '
            'This usually means the selected example WSIs do not contain at least one '
            'slide from the current class and one slide from other classes.'
        )

    if sampling_size > 0:
        sampled_idx = torch.randint(0, feats[neg].shape[0], (1, sampling_size))[0]
        sim_pos = compute_similarity(feats[pos], feats[neg][sampled_idx[:sampling_size], :], topk=topk)
        sampled_idx = torch.randint(0, feats[pos].shape[0], (1, sampling_size))[0]
        sim_neg = compute_similarity(feats[neg], feats[pos][sampled_idx[:sampling_size], :], topk=topk)
    else:
        sim_pos = compute_similarity(feats[pos], feats[neg], topk=topk)
        sim_neg = compute_similarity(feats[neg], feats[pos], topk=topk)
    sim = torch.zeros(feats.shape[0]).cuda()
    sim[pos] = sim_pos.cuda()
    sim[neg] = sim_neg.cuda()

    # step2 assign init label for each wsi
    # record certain pos and normal patches
    info_dic = {}
    labeled_idx_dic = {255: [], 0: [], 1: []}
    for n in wsi_names:
        pos, idx = [], []
        for i, pn in enumerate(patch_names):
            if n in pn:
                x, y = pn.split('/')[-1].split('.')[0].split('_')
                pos.append([int(y), int(x)])
                idx.append(i)
        pos = np.array(pos)
        info_dic[n] = {'idx': idx, 'pos': pos}

        sim_n = sim[idx]
        sim_n = (sim_n - sim_n.min()) / (sim_n.max() - sim_n.min()).clamp_min(1e-8)
        labels_n = torch.zeros(sim_n.shape[0]).cuda().long()
        labels_n = basic_tagger(sim_n, labels_n ==0, labels_n, uncertain, positive=False)

        wsi_label = labels[idx[0]].item()
        labeled_idx_dic[wsi_label].extend([idx[_] for _ in (labels_n == 1).nonzero()[:, 0]])
        labeled_idx_dic[255].extend([idx[_] for _ in (labels_n == 0).nonzero()[:, 0]])

    # step3 assign label via dataset pos and neg
    for n in wsi_names:
        idx_n = info_dic[n]['idx']
        feats_n = feats[idx_n]
        wsi_label = labels[idx_n[0]].item()
        sim_pos = compute_similarity(feats_n, feats[labeled_idx_dic[wsi_label]], topk=-1)
        sim_neg = compute_similarity(feats_n, feats[labeled_idx_dic[255]], topk=-1)
        score = sim_pos - sim_neg
        score = torch.clamp(score, -0.5, 0.5) + 0.5 # norm to 0-1 for vis via clamp

        labels_n = torch.zeros(feats_n.shape[0]).cuda().long()
        labels_n = basic_tagger(score, labels_n==0, labels_n, uncertain, True)
        labels_n[labels_n == 0] = 255 # normal cells
        labels_n[labels_n == -1] = 254 # uncertain
        labels_n[labels_n == 1] = wsi_label
        labels[idx_n] = labels_n

        if vis_info != None:
            vis_heat(score, labels_n, info_dic[n]['pos'], n + '.svs', vis_info['wsi_dir'], \
                vis_info['vis_dir'], vis_info['mask_dir'], side=512)

    return labels


def execute_mask_subtyping_tagger(feats, labels, patch_names, wsi_names, wsi_binary_labels, \
        vis_info=None, uncertain=0.1, topk=40):
    pos_anchor = labels == 1
    neg_anchor = labels == 0
    pos_count = int(pos_anchor.sum().item())
    neg_count = int(neg_anchor.sum().item())
    if pos_count == 0 or neg_count == 0:
        unique_labels, unique_counts = torch.unique(labels.detach().cpu(), return_counts=True)
        label_counts = ', '.join(
            f'{int(label.item())}:{int(count.item())}'
            for label, count in zip(unique_labels, unique_counts)
        )
        raise ValueError(
            'Mask subtyping tagger needs explicit positive (1) and negative (0) anchor tokens, '
            f'but got pos={pos_count}, neg={neg_count}. Label counts: {label_counts}. '
            'This usually means the selected example WSIs do not contain annotated patches '
            'from both the current class and other classes.'
        )

    label_by_wsi = dict(wsi_binary_labels)
    assigned_pos, assigned_neg, kept_bg, kept_uncertain = 0, 0, 0, 0
    total_unknown = 0

    for n in wsi_names:
        idx = []
        for i, pn in enumerate(patch_names):
            if n in pn:
                idx.append(i)
        if len(idx) == 0:
            continue
        if n not in label_by_wsi:
            raise ValueError(f'Missing one-vs-rest WSI label for mask subtyping tagger: {n}')

        wsi_label = int(label_by_wsi[n])
        if wsi_label not in [0, 1]:
            raise ValueError(f'Mask subtyping WSI label must be 0 or 1, got {wsi_label} for {n}')

        idx_t = torch.as_tensor(idx, device=labels.device, dtype=torch.long)
        unknown_mask = (labels[idx_t] == 255) | (labels[idx_t] == 254) | (labels[idx_t] == -1)
        if int(unknown_mask.sum().item()) == 0:
            continue

        unknown_idx = idx_t[unknown_mask]
        unknown_feats = feats[unknown_idx]
        sim_pos = compute_similarity(unknown_feats, feats[pos_anchor], topk=topk)
        sim_neg = compute_similarity(unknown_feats, feats[neg_anchor], topk=topk)
        if wsi_label == 1:
            target_sim = sim_pos
            other_sim = sim_neg
        else:
            target_sim = sim_neg
            other_sim = sim_pos

        fg_score = normalize_similarity_values(torch.maximum(sim_pos, sim_neg))
        fg_labels = torch.zeros(unknown_feats.shape[0], device=labels.device).long()
        fg_labels = basic_tagger(fg_score, fg_labels == 0, fg_labels, uncertain, positive=True)

        updated = torch.full((unknown_feats.shape[0],), 255, device=labels.device, dtype=torch.long)
        updated[fg_labels == -1] = 254
        class_candidate = fg_labels == 1

        if int(class_candidate.sum().item()) > 0:
            class_score = normalize_similarity_values(target_sim[class_candidate] - other_sim[class_candidate])
            class_labels = torch.zeros(int(class_candidate.sum().item()), device=labels.device).long()
            class_labels = basic_tagger(class_score, class_labels == 0, class_labels, uncertain, positive=True)
            class_updated = torch.full_like(class_labels, 255)
            class_updated[class_labels == 1] = wsi_label
            class_updated[class_labels == -1] = 254
            updated[class_candidate] = class_updated

        labels[unknown_idx] = updated
        total_unknown += int(updated.shape[0])
        assigned_pos += int((updated == 1).sum().item())
        assigned_neg += int((updated == 0).sum().item())
        kept_bg += int((updated == 255).sum().item())
        kept_uncertain += int((updated == 254).sum().item())

    print(
        '[tagger] mask subtyping: refined unknown tokens=' + str(total_unknown) +
        ', assigned_pos=' + str(assigned_pos) +
        ', assigned_neg=' + str(assigned_neg) +
        ', background=' + str(kept_bg) +
        ', uncertain=' + str(kept_uncertain) +
        ', anchors pos/neg=' + str(pos_count) + '/' + str(neg_count)
    )
    return labels


# ====================== inference with classifier, aggregator, post processor ======================

def inference(args, example_feats, example_labels, example_patch_names,
    query_feats, query_patch_names, wsi_size, top_instance=1, vis_info=None, smooth=None):

    # ====================== in-context classifier ======================

    # Stream top-k similarities by query chunk to avoid an example x query matrix.
    similarity_kwargs = {
        'aggregation': getattr(args, 'similarity_aggregation', 'mean'),
        'softmax_temperature': getattr(args, 'similarity_temperature', 10.0),
        'adaptive_min_k': getattr(args, 'adaptive_min_topk', 1),
        'adaptive_window': getattr(args, 'adaptive_window', 0.6),
    }
    pos_score = compute_similarity(query_feats, example_feats[example_labels == 1], topk=args.topk, **similarity_kwargs)
    neg_score = compute_similarity(query_feats, example_feats[example_labels == 0], topk=args.topk, **similarity_kwargs)
    query_logits = (pos_score - neg_score).to(query_feats.device)
    query_logits = spatially_smooth_logits(
        query_logits, query_patch_names,
        radius=getattr(args, 'spatial_smooth_radius', 1),
        strength=getattr(args, 'spatial_smooth_strength', 0.0),
        query_feats=query_feats,
        feature_weight=getattr(args, 'spatial_feature_weight', 0.0),
    )

    # ====================== attention aggregator ======================

    wsi_pred = aggregate_query_logits(
        query_feats, query_logits, top_instance, args.related_thresh, args.temperature
    )

    # ====================== patch reshape to wsi for heatmap / seg ======================

    idx_in_map = []
    if wsi_size != None:
        #patch_pred = torch.zeros(wsi_size).cuda() + query_logits.min()
        patch_pred = torch.zeros(wsi_size).cuda() + 255
        patch_pred_list = []
        for i, n in enumerate(query_patch_names):
            patch_pred_list.append(query_logits[i])
            x, y = n.split('/')[-1].split('.')[0].split('_')
            try:
                patch_pred[int(y), int(x)] = query_logits[i]
                idx_in_map.append(int(y) * patch_pred.shape[1] + int(x))
            except:
                if len(idx_in_map) != 0:
                    idx_in_map.append(idx_in_map[-1])
                else:
                    idx_in_map.append(0)
                continue
    else:
        patch_pred, patch_pred_list = None, None

    # ====================== segmentation post processor ======================

    # patch_pred ing
    if smooth != None:
        fg = patch_pred != 255
        bg = fg == False
        smooth_pred = patch_pred.clone()
        smooth_pred[bg] = smooth_pred[fg].mean() # replace 255 to mean value before smoothing
        smooth_pred = smooth(smooth_pred.reshape(1, 1, smooth_pred.shape[0], smooth_pred.shape[1]))[0,0]
        patch_pred[fg] = smooth_pred[fg]
        patch_pred_list = patch_pred.reshape(-1)[idx_in_map]

    return wsi_pred, patch_pred, patch_pred_list
