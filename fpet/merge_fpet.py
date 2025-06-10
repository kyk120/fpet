# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch
import math


def do_nothing(x, mode=None):
    return x


def checkerboard_split(x: torch.Tensor, protected: int):
    
    x_others = x
    
    if protected:
        x_cls = x_others[:, :1, :]
        x_others = x_others[:, 1:, :]
    if protected == 2:
        x_distill = x_others[:, -1:, :]
        x_others = x_others[:, :-1, :]

    B, N, C = x_others.shape

    if N == 196:
        width = 14
        height = int(N/width)
    else:
        width = int(math.floor(math.sqrt(N)))
        height = int(N/width)

    x_others = x_others.view(B, height, width, C)

    x_tl = x_others[..., ::2, :][:, ::2, :, :].reshape(B, -1, C)
    x_br = x_others[..., 1::2, :][:, 1::2, :, :].reshape(B, -1, C)

    x_tr = x_others[..., ::2, :][:, 1::2, :, :].reshape(B, -1, C)
    x_bl = x_others[..., 1::2, :][:, ::2, :, :].reshape(B, -1, C)

    # a = torch.cat([x_cls, x_tl, x_br], dim=1)
    # b = torch.cat([x_tr, x_bl], dim=1)
    a = torch.cat([x_tl, x_br], dim=1)
    b_tensors = [x_tr, x_bl]
    if protected:
        b_tensors = [x_cls] + b_tensors
    if protected == 2:
        b_tensors = b_tensors + [x_distill]
    b = torch.cat(b_tensors, dim=1)
    # a = torch.cat([x_tr, x_bl], dim=1)
    # b = torch.cat([x_cls, x_tl, x_br], dim=1)

    return a, b


def bipartite_diff_matching(
    metric: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Input size is [batch, tokens, channels].

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]

    # with torch.no_grad():
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = checkerboard_split(metric, protected)

    scores = a @ b.transpose(-1, -2)

    if class_token:
        scores[..., :, 0] = -math.inf
    if distill_token:
        scores[..., :, 0] = -math.inf

    v, idx = torch.topk(scores, 2, dim=-1)
    mean12 = v.mean(dim=-1, keepdim=True)
    soft_matrix = torch.sigmoid(scores - mean12)
    hard_matrix = (soft_matrix > 0.5).float()
    matching_matrix = soft_matrix + (hard_matrix - soft_matrix).detach()

    def merge(x: torch.Tensor) -> torch.Tensor:
        x_a, x_b = checkerboard_split(x, protected)
        x_merge = torch.einsum('bik, bij->bkj', matching_matrix, x_a)
        x_merged_sum = x_b + x_merge
        return x_merged_sum
    
    return merge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size)
    size = merge(size)

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
