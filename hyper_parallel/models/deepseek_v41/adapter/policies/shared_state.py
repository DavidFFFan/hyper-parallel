# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model-owned recompute hooks for the DeepSeek-V4.1 shared attention state."""

from __future__ import annotations

from typing import Any


def configure_shared_state(model: Any) -> None:
    """Install the must-run-once regions of the layers that will be replayed.

    Replayed index-producing layers get their discrete selection turned into an
    exclusion region, because a replay must neither choose a different key set
    nor change how many tensors the checkpoint sees in each phase. The published
    shared-state entries need no schedule either: the state is an input of every
    region, so the graph keeps it alive until backward is done with every
    consumer.
    """
    for module in _replayed_attentions(model):
        _pin_regions(module)


def _pin_regions(module: Any) -> None:
    """Install the must-run-once regions of one replayed layer.

    Only the indexer's discrete selection is pinned. A recompute must not choose
    a different Top-K set, and the block contains differentiable ops, so
    skipping it would also break the checkpoint's forward/recompute saved-tensor
    count; an exclusion region runs it once and replays its cached result.

    Producer-side regions are deliberately not pinned: keeping the compressor
    and indexer projections out of the recompute retains their activations
    (+0.21 GB on the 4096-token crop) while the out-of-order producer recompute
    they would guard against does not reproduce on that crop, so the pin buys
    nothing. Add a pin here only with a measurement that shows otherwise.
    """
    if not getattr(module, "is_index_source", False):
        return
    indexer = getattr(module, "indexer", None)
    if indexer is not None:
        indexer.enable_selection_exclude()


def _replayed_attentions(model: Any) -> list[Any]:
    """Return the attention modules whose forward a backward replay revisits.

    The shipped policy makes every decoder layer a region, so the CSA2
    attentions found under those regions are the ones replay reaches. An
    attention that an exclusion keeps out of replay is not pinned, because an
    exclusion region must not nest inside another exclusion region.
    """
    from hyper_parallel.core.activation_memory.wrapper import (  # pylint: disable=C0415
        CheckpointExcludeWrapper,
    )

    excluded_inner_ids = set()
    for module in model.modules():
        if isinstance(module, CheckpointExcludeWrapper):
            excluded_inner_ids.add(id(module._wrapped_module))  # pylint: disable=W0212
    replayed = []
    for module in model.modules():
        if not getattr(module, "compress_ratio", 0):
            continue
        if getattr(module, "layer_idx", None) is None:
            continue
        if id(module) in excluded_inner_ids or isinstance(module, CheckpointExcludeWrapper):
            continue
        replayed.append(module)
    return replayed


__all__ = ["configure_shared_state"]
