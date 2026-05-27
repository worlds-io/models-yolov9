"""
Anti-Forgetting Sampling Strategy (AFSS).

Implements the sampling strategy from "Does YOLO Really Need to See Every
Training Image in Every Epoch?" (Xie et al., arXiv:2603.17684).

Each training image is periodically scored for *learning sufficiency*
(``min(recall, precision)`` of the model's own detections on that image) and
bucketed into easy / medium / hard tiers. Per epoch the sampler then emits:

  - hard images   : every epoch (optionally repeated) - they carry the gradient
                    signal that is still moving the loss;
  - medium images : a fraction, prioritising the longest-unused (anti-forgetting)
                    plus a random remainder for coverage;
  - easy images   : a small fraction, strictly the longest-unused ("continuous
                    review") so the model does not forget what it has mastered.

Sufficiency is recomputed only every ``refresh_period`` epochs to amortise the
inference cost. ``last_used_epoch`` is tracked per image across the whole run.

We use this not to train *faster* but to train *longer on the images that still
matter*: the per-epoch image count shrinks, and the freed budget is reinvested
(by the Go orchestrator) into more epochs / a longer LR schedule, for better
convergence at roughly equal wall-clock. See the project plan for the rationale.

This module is deliberately framework-light. The only YOLO-specific piece is
``score_train_set`` / ``build_scoring_loader``; ``AFSSState`` and ``AFSSSampler``
are reused verbatim by the D-FINE fork.
"""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision.ops import box_iou

from yolo.utils.bounding_box_utils import to_metrics_format
from yolo.utils.logger import logger

# Tier ids
EASY, MEDIUM, HARD = 0, 1, 2


def per_image_recall_precision(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresh: float,
) -> tuple[float, float]:
    """Greedy class-aware IoU matching for a single image.

    Returns ``(recall, precision)``. Predictions are assumed already filtered by
    confidence. Matching is one-to-one, highest-IoU first, same-class only.

    Degenerate cases are defined so that "sufficiency = min(recall, precision)"
    keeps unlearned images in the hard tier:
      - no GT, no pred           -> (1, 1)  nothing to find, nothing found: mastered
      - no GT, some pred         -> (1, 0)  pure false positives: precision 0 -> hard
      - some GT, no pred         -> (0, 1)  everything missed: recall 0 -> hard
    """
    n_gt = int(gt_boxes.shape[0])
    n_pred = int(pred_boxes.shape[0])
    if n_gt == 0:
        return (1.0, 1.0) if n_pred == 0 else (1.0, 0.0)
    if n_pred == 0:
        return 0.0, 1.0

    iou = box_iou(pred_boxes, gt_boxes)  # (n_pred, n_gt)
    same_class = pred_labels.view(-1, 1) == gt_labels.view(1, -1)
    iou = torch.where(same_class, iou, torch.zeros_like(iou))

    # Candidate pairs above threshold, matched greedily by descending IoU.
    mask = iou >= iou_thresh
    if not bool(mask.any()):
        tp = 0
    else:
        pi, gi = torch.where(mask)
        vals = iou[pi, gi]
        order = torch.argsort(vals, descending=True)
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        tp = 0
        for k in order.tolist():
            p, g = int(pi[k]), int(gi[k])
            if p in matched_pred or g in matched_gt:
                continue
            matched_pred.add(p)
            matched_gt.add(g)
            tp += 1

    fp = n_pred - tp
    fn = n_gt - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return recall, precision


class AFSSState:
    """Per-image bookkeeping that persists for the whole run.

    ``sufficiency`` starts at 0.0 so every image is HARD (fully sampled) until it
    has been scored at least once - the safe default. ``last_used_epoch`` starts
    at -1 so never-used images rank oldest for anti-forgetting selection.
    """

    def __init__(self, num_images: int, easy_thresh: float, medium_thresh: float):
        self.num_images = num_images
        self.easy_thresh = easy_thresh
        self.medium_thresh = medium_thresh
        self.sufficiency = np.zeros(num_images, dtype=np.float32)
        self.last_used_epoch = np.full(num_images, -1, dtype=np.int64)
        self.tier = np.full(num_images, HARD, dtype=np.int8)
        self.last_scored_epoch = -1

    def update_sufficiency(self, scores: Dict[int, float], epoch: int) -> None:
        for idx, suff in scores.items():
            self.sufficiency[idx] = suff
        self.tier = np.where(
            self.sufficiency >= self.easy_thresh,
            EASY,
            np.where(self.sufficiency >= self.medium_thresh, MEDIUM, HARD),
        ).astype(np.int8)
        self.last_scored_epoch = epoch

    def tier_counts(self) -> Dict[str, int]:
        return {
            "easy": int((self.tier == EASY).sum()),
            "medium": int((self.tier == MEDIUM).sum()),
            "hard": int((self.tier == HARD).sum()),
        }

    # Persisted with the D-FINE solver checkpoint; unused by the YOLOv9 (Lightning)
    # path but kept so the same class works in both forks.
    def state_dict(self) -> dict:
        return {
            "sufficiency": self.sufficiency,
            "last_used_epoch": self.last_used_epoch,
            "tier": self.tier,
            "last_scored_epoch": self.last_scored_epoch,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.sufficiency = np.asarray(sd["sufficiency"], dtype=np.float32)
        self.last_used_epoch = np.asarray(sd["last_used_epoch"], dtype=np.int64)
        self.tier = np.asarray(sd["tier"], dtype=np.int8)
        self.last_scored_epoch = int(sd["last_scored_epoch"])


class AFSSSampler(Sampler):
    """Emits a per-epoch subset of dataset indices according to AFSS.

    ``__len__`` returns the *current* epoch's selection size, so the DataLoader's
    length (and therefore the per-epoch iteration count and the optimizer's
    per-epoch warmup math) shrinks automatically once sufficiency is known.

    In ``score_only`` mode the full dataset is emitted every epoch (no
    subsampling); sufficiency is still scored/logged so the tier distribution can
    be inspected on real data before committing to subsampling.
    """

    def __init__(
        self,
        num_images: int,
        easy_thresh: float = 0.9,
        medium_thresh: float = 0.5,
        easy_keep_frac: float = 0.15,
        medium_keep_frac: float = 0.5,
        hard_repeat: int = 1,
        refresh_period: int = 5,
        score_only: bool = False,
        seed: int = 0,
    ):
        self.num_images = num_images
        self.easy_keep_frac = easy_keep_frac
        self.medium_keep_frac = medium_keep_frac
        self.hard_repeat = max(1, hard_repeat)
        self.refresh_period = max(1, refresh_period)
        self.score_only = score_only
        self.state = AFSSState(num_images, easy_thresh, medium_thresh)
        self._rng = np.random.default_rng(seed)
        self._epoch = 0
        self._indices: List[int] = list(range(num_images))  # epoch-0 default: all

    def needs_refresh(self, epoch: int) -> bool:
        return self.state.last_scored_epoch < 0 or (epoch - self.state.last_scored_epoch) >= self.refresh_period

    def update_sufficiency(self, scores: Dict[int, float], epoch: int) -> None:
        self.state.update_sufficiency(scores, epoch)
        counts = self.state.tier_counts()
        logger.info(
            f"[AFSS] epoch {epoch}: scored {len(scores)} images -> "
            f"easy={counts['easy']} medium={counts['medium']} hard={counts['hard']}"
        )

    def _select(self, epoch: int) -> List[int]:
        tier = self.state.tier
        last_used = self.state.last_used_epoch
        hard_idx = np.where(tier == HARD)[0]
        medium_idx = np.where(tier == MEDIUM)[0]
        easy_idx = np.where(tier == EASY)[0]

        selected: List[int] = []
        # Hard: all, optionally repeated to concentrate gradient on the unconverged tail.
        selected.extend(np.tile(hard_idx, self.hard_repeat).tolist())

        # Medium: prioritise longest-unused, fill remainder randomly for coverage.
        if medium_idx.size:
            keep = ceil(self.medium_keep_frac * medium_idx.size)
            ordered = medium_idx[np.argsort(last_used[medium_idx], kind="stable")]
            n_priority = keep // 2
            chosen = list(ordered[:n_priority])
            pool = ordered[n_priority:]
            remaining = keep - len(chosen)
            if remaining > 0 and pool.size:
                chosen.extend(self._rng.choice(pool, size=min(remaining, pool.size), replace=False).tolist())
            selected.extend(int(i) for i in chosen)

        # Easy: small fraction, strictly the longest-unused (continuous review).
        if easy_idx.size:
            keep = ceil(self.easy_keep_frac * easy_idx.size)
            ordered = easy_idx[np.argsort(last_used[easy_idx], kind="stable")]
            selected.extend(int(i) for i in ordered[:keep])

        return selected

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if self.score_only or self.state.last_scored_epoch < 0:
            # No (or not-yet-computed) tiers: behave like a plain shuffled full pass.
            idx = np.arange(self.num_images)
        else:
            idx = np.asarray(self._select(epoch), dtype=np.int64)
            if idx.size == 0:  # guard against an all-empty selection
                idx = np.arange(self.num_images)
        self.state.last_used_epoch[idx] = epoch
        self._rng.shuffle(idx)
        self._indices = idx.tolist()

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def build_scoring_loader(data_cfg, dataset_cfg, task: str, max_images: int, seed: int = 0):
    """Build a clean (validation-transform) loader over the *training* images.

    Returns ``(loader, indices)`` where ``indices`` are the train-dataset indices
    each batch item maps to, in loader order. ``data_cfg`` must be the validation
    DataConfig so scoring sees deterministic, un-augmented images; the underlying
    ``YoloDataset`` caches by phase, so index ``i`` here is the same image as index
    ``i`` in the train DataLoader's dataset.
    """
    from torch.utils.data import DataLoader, Subset

    from yolo.tools.data_loader import YoloDataset, collate_fn

    dataset = YoloDataset(data_cfg, dataset_cfg, task)
    n = len(dataset)
    if n <= max_images:
        indices = list(range(n))
    else:
        indices = sorted(np.random.default_rng(seed).choice(n, size=max_images, replace=False).tolist())
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=data_cfg.batch_size,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn,
        shuffle=False,
    )
    return loader, indices


@torch.no_grad()
def score_train_set(
    model,
    post_process,
    loader,
    indices: List[int],
    device,
    conf_thresh: float,
    iou_thresh: float,
) -> Dict[int, float]:
    """Run inference over the scoring loader and return ``{dataset_idx: sufficiency}``.

    Mirrors ``ValidateModel.validation_step``: ``post_process(model(images), ...)``
    yields per-image ``[class, x1, y1, x2, y2, score]`` tensors in pixel coords,
    matched against the (same-coord) ground-truth boxes from the collate output.
    """
    was_training = model.training
    model.eval()
    suff: Dict[int, float] = {}
    pos = 0
    for batch in loader:
        _, images, targets, _, _ = batch
        images = images.to(device)
        h, w = images.shape[2:]
        predicts = post_process(model(images), image_size=[w, h])
        for b in range(len(predicts)):
            idx = indices[pos]
            pos += 1
            pred = to_metrics_format(predicts[b])
            gt = to_metrics_format(targets[b])
            pred_boxes, pred_labels = pred["boxes"], pred["labels"]
            scores = pred.get("scores")
            if scores is not None:
                keep = scores >= conf_thresh
                pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
            recall, precision = per_image_recall_precision(
                pred_boxes.detach().cpu().float(),
                pred_labels.detach().cpu(),
                gt["boxes"].detach().cpu().float(),
                gt["labels"].detach().cpu(),
                iou_thresh,
            )
            suff[idx] = min(recall, precision)
    if was_training:
        model.train()
    return suff
