# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from fpet.merge_fpet import bipartite_diff_matching, merge_source, merge_wavg
from adaptformer import Adapter

def forward_block(self, x):

    attn_size = self._fpet_info["size"] if self._fpet_info["prop_attn"] else None
    x_attn, metric = self.attn(self.norm1(x), attn_size)
    x = x + self.drop_path(x_attn)

    if hasattr(self, 'refinement'):
        metric = metric.detach()
        metric = metric + 1 * self.refinement(metric)
        # Apply FPET here
        merge = bipartite_diff_matching(
            metric,
            self._fpet_info["class_token"],
            self._fpet_info["distill_token"],
        )
        if self._fpet_info["trace_source"]:
            self._fpet_info["source"] = merge_source(
                merge, x, self._fpet_info["source"]
            )
        x, self._fpet_info["size"] = merge_wavg(merge, x, self._fpet_info["size"])
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter_mlp(x) * self.s
    return x

def forward_attn(self, x, size):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    delta_q = self.lora_q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    delta_v = self.lora_v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    q, v = q + delta_q[0], v + delta_v[0]
    attn = (q @ k.transpose(-2, -1)) * self.scale

    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, k.mean(1)

class FPETBlock(Block):
    """
    Modifications:
     - Apply FPET between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape is {x.shape}')
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._fpet_info["size"] if self._fpet_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        if hasattr(self, 'refinement'):
            metric = metric.detach()
            metric = metric + 1 * self.refinement(metric)
            # Apply FPET here
            merge = bipartite_diff_matching(
                metric,
                self._fpet_info["class_token"],
                self._fpet_info["distill_token"],
            )
            if self._fpet_info["trace_source"]:
                self._fpet_info["source"] = merge_source(
                    merge, x, self._fpet_info["source"]
                )
            x, self._fpet_info["size"] = merge_wavg(merge, x, self._fpet_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class FPETAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_fpet_class(transformer_class):
    class FPETVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._fpet_info["size"] = None
            self._fpet_info["source"] = None

            return super().forward(*args, **kwdargs)

    return FPETVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, method = '', r_layer: int = 6, dim_key = 64
):
    """
    Applies FPET to this transformer.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._fpet_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    FPETVisionTransformer = make_fpet_class(model.__class__)

    model.__class__ = FPETVisionTransformer
    model._fpet_info = {
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._fpet_info["distill_token"] = True

    idx = 0
    for module in model.modules():
        if isinstance(module, Block):
            module._fpet_info = model._fpet_info
            if r_layer == idx:
                module.refinement = Adapter(dim=32, bit=1, in_dim=dim_key)
                
            if 'adaptformer' not in method:
                module.__class__ = FPETBlock
            else:
                bound_method = forward_block.__get__(module, module.__class__)
                setattr(module, 'forward', bound_method)
            
            idx += 1
                
        elif isinstance(module, Attention):
            if 'lora' not in method:
                module.__class__ = FPETAttention
            else:
                bound_method = forward_attn.__get__(module, module.__class__)
                setattr(module, 'forward', bound_method)
                
