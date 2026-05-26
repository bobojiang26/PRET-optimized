import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE = os.path.join(ROOT, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from modules import compute_similarity, spatially_smooth_logits, aggregate_query_logits
from main import (
    apply_context_feature_centering,
    apply_require_label_filter,
    binary_conformal_summary,
    chunked_min_cosine_distance,
    get_example_names_at_label_ratio,
    has_real_wsi_label,
    label_counts,
    select_validation_names,
    threshold_source,
    use_test_threshold,
)


def test_compute_similarity_keeps_original_mean_behavior():
    example = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    out = compute_similarity(query, example, topk=2)

    assert torch.allclose(out, torch.tensor([0.9, 0.6]), atol=1e-6)


def test_attention_empty_related_fallback_matches_original_all_patch_behavior():
    query_feats = torch.eye(3)
    query_logits = torch.tensor([2.0, 0.0, -2.0])

    out = aggregate_query_logits(
        query_feats, query_logits, top_instance=1, related_thresh=1.1, temperature=10.0
    )
    expected = (torch.softmax(torch.tensor([1.0, 0.0, 0.0]) * 10.0, 0) * query_logits).sum()

    assert torch.allclose(out, expected, atol=1e-6)


def test_default_validation_selection_preserves_original_order():
    names = ["a", "b", "c", "d"]
    dataset_info = {
        "a": {"wsi_label": 1},
        "b": {"wsi_label": 2},
        "c": {"wsi_label": 1},
        "d": {"wsi_label": 2},
    }

    assert select_validation_names(names, dataset_info, 2, balanced=False) == ["a", "b"]


def test_threshold_source_defaults_to_validation_split():
    class Args:
        pass

    assert threshold_source(Args()) == "val"
    assert not use_test_threshold(Args())


def test_threshold_source_can_use_test_split_for_calibration():
    class Args:
        threshold_source = "test"

    assert threshold_source(Args()) == "test"
    assert use_test_threshold(Args())


def test_example_ratio_selects_per_class_ceiling_targets():
    names = ["a0", "a1", "a2", "b0", "b1", "b2", "b3", "b4"]
    dataset_info = {
        "a0": {"wsi_label": 1},
        "a1": {"wsi_label": 1},
        "a2": {"wsi_label": 1},
        "b0": {"wsi_label": 2},
        "b1": {"wsi_label": 2},
        "b2": {"wsi_label": 2},
        "b3": {"wsi_label": 2},
        "b4": {"wsi_label": 2},
    }

    selected, targets = get_example_names_at_label_ratio(
        names, dataset_info, 0.6, check_num=True, expected_labels=[1, 2]
    )

    assert targets == {1: 2, 2: 3}
    assert label_counts(selected, dataset_info) == targets


def test_example_ratio_respects_per_class_cap():
    names = ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3", "b4"]
    dataset_info = {
        "a0": {"wsi_label": 1},
        "a1": {"wsi_label": 1},
        "a2": {"wsi_label": 1},
        "a3": {"wsi_label": 1},
        "b0": {"wsi_label": 2},
        "b1": {"wsi_label": 2},
        "b2": {"wsi_label": 2},
        "b3": {"wsi_label": 2},
        "b4": {"wsi_label": 2},
    }

    selected, targets = get_example_names_at_label_ratio(
        names, dataset_info, 0.8, max_per_class=2, check_num=True, expected_labels=[1, 2]
    )

    assert targets == {1: 2, 2: 2}
    assert label_counts(selected, dataset_info) == targets


def test_adaptive_similarity_emphasizes_close_references():
    example = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    query = torch.tensor([[1.0, 0.0]])

    mean = compute_similarity(query, example, topk=3, aggregation="mean")
    adaptive = compute_similarity(
        query, example, topk=3, aggregation="adaptive",
        softmax_temperature=20.0, adaptive_min_k=1, adaptive_window=0.2
    )

    assert adaptive.item() > mean.item()
    assert adaptive.item() <= 1.0


def test_spatial_smoothing_uses_coordinate_neighbors():
    logits = torch.tensor([0.0, 1.0, 10.0])
    names = ["slide/0_0.jpeg", "slide/1_0.jpeg", "slide/3_0.jpeg"]

    smoothed = spatially_smooth_logits(logits, names, radius=1, strength=1.0)

    assert 0.0 < smoothed[0].item() < 1.0
    assert 0.0 < smoothed[1].item() < 1.0
    assert smoothed[2].item() == logits[2].item()


def test_context_centering_returns_normalized_rows():
    example = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    query = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    centered_example, centered_query = apply_context_feature_centering(example, query, mode="joint")

    assert torch.allclose(centered_example.norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(centered_query.norm(dim=1), torch.ones(2), atol=1e-6)


def test_chunked_min_cosine_distance_matches_dense_calculation():
    feats = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]), p=2, dim=1)
    refs = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ]), p=2, dim=1)

    chunked = chunked_min_cosine_distance(feats, refs, query_chunk=2, ref_chunk=1)
    dense = (1 - feats @ refs.t()).min(1).values

    assert torch.allclose(chunked, dense, atol=1e-6)


def test_conformal_summary_reports_prediction_set_stats():
    calib_scores = torch.tensor([2.0, -2.0, 1.5, -1.5])
    calib_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    eval_scores = torch.tensor([2.5, -2.5])
    eval_labels = torch.tensor([1.0, 0.0])

    summary = binary_conformal_summary(
        calib_scores, calib_labels, eval_scores, eval_labels, threshold=0.0, alpha=0.1
    )

    assert summary["coverage"] == 1.0
    assert summary["avg_set_size"] >= 1.0
    assert summary["qhat"] >= 0.0


def test_has_real_wsi_label_filters_unlabeled_and_pseudo_labels():
    assert has_real_wsi_label({"wsi_label": 0})
    assert has_real_wsi_label({"wsi_labels": []})
    assert has_real_wsi_label({"wsi_labels": [1, "2"]})
    assert not has_real_wsi_label({})
    assert not has_real_wsi_label({"wsi_label": None})
    assert not has_real_wsi_label({"wsi_label": ""})
    assert not has_real_wsi_label({"wsi_label": 1, "pseudo_label": True})


def test_require_label_filter_keeps_only_real_labels():
    class Args:
        require_label = True

    dataset_info = {
        "labeled_neg": {"wsi_label": 0},
        "labeled_multi_neg": {"wsi_labels": []},
        "unlabeled": {},
        "pseudo": {"wsi_label": 1, "pseudo_label": True},
    }

    filtered = apply_require_label_filter(dataset_info, Args(), context="test")

    assert list(filtered.keys()) == ["labeled_neg", "labeled_multi_neg"]
