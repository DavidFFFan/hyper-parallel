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
"""Model-owned lifetime policy for the DeepSeek-V4.1 shared attention state."""

from __future__ import annotations

from typing import Any


def build_shared_state_release_plan(
        config: Any,
        replayable_layers: Any = (),
) -> dict[int, tuple[tuple[str, int], ...]]:
    """Derive when each published CSA2 entry stops being needed.

    A Reuse/Reindex layer reads compressed K=V and the Top-K selection from the
    most recent producer at or before it, a Reindex layer additionally reads the
    shared indexer K, and hierarchical Reindex layers read the candidate pool of
    the configured source. Producers publish before their own consumers run, so
    the last consumer of every entry follows from the released layer topology
    alone.

    A consumer whose forward is replayed reads its entries a second time during
    backward, so an entry can only be released after its last consumer when that
    consumer is not replayed. Entries nobody reads at all (an encoder Full
    layer's indexer K) are still released at their producer.

    The entry kinds are the contract of
    :meth:`hyper_parallel.components.modules.shared_compressed_dsa_attention.
    SharedCompressedAttentionState.release_consumed`; the topology fields are
    V4.1-only, which is why this schedule lives with the model rather than with
    the generic attention module.

    Args:
        config: V4.1 text config carrying the compression ratios and the
            source-layer lists.
        replayable_layers: Layers whose forward runs again during backward.
            Their consumed entries are kept instead of being released.

    Returns:
        ``{consumer_layer: ((kind, source_layer), ...)}`` where ``kind`` is one
        of ``compressed_kv``, ``index_key``, ``topk`` or ``candidate``.
    """
    replayed = {int(layer) for layer in replayable_layers}
    layer_count = int(config.num_hidden_layers)
    ratios = list(getattr(config, "v41_compress_ratios", ()) or ())
    kv_sources = sorted(int(source) for source in config.v41_kv_source_layer_ids)
    index_sources = sorted(int(source) for source in config.v41_index_source_layer_ids)
    candidate_source = int(getattr(config, "v41_candidate_source_layer_id", -1))
    published: list[tuple[str, int]] = []
    for source in kv_sources:
        published.append(("compressed_kv", source))
        published.append(("index_key", source))
    for source in index_sources:
        published.append(("topk", source))
    if candidate_source >= 0:
        published.append(("candidate", candidate_source))

    last_consumer: dict[tuple[str, int], int] = {}
    unread: set[tuple[str, int]] = set(published)
    for layer_idx in range(layer_count):
        if layer_idx >= len(ratios) or int(ratios[layer_idx]) <= 0:
            continue
        kv_source = max((source for source in kv_sources if source <= layer_idx), default=None)
        index_source = max((source for source in index_sources if source <= layer_idx), default=None)
        if kv_source is not None:
            last_consumer[("compressed_kv", kv_source)] = layer_idx
            unread.discard(("compressed_kv", kv_source))
        if index_source is not None and layer_idx in index_sources and layer_idx not in kv_sources:
            # Only a Reindex layer rescores the shared indexer K; a Reuse layer
            # consumes the Top-K selection alone.
            last_consumer[("index_key", kv_source)] = layer_idx
            unread.discard(("index_key", kv_source))
        if index_source is not None:
            last_consumer[("topk", index_source)] = layer_idx
            unread.discard(("topk", index_source))
        if 0 <= candidate_source < layer_idx:
            last_consumer[("candidate", candidate_source)] = layer_idx
            unread.discard(("candidate", candidate_source))

    # An entry with no consumer at all - an encoder Full layer's indexer K, for
    # example - is dead the moment its producer returns.
    for dependency in published:
        last_consumer.setdefault(dependency, dependency[1])

    plan: dict[int, list[tuple[str, int]]] = {}
    for dependency, layer_idx in last_consumer.items():
        if dependency not in unread and layer_idx in replayed:
            # The last consumer replays, so the entry has to survive into backward.
            continue
        plan.setdefault(layer_idx, []).append(dependency)
    return {layer: tuple(sorted(dependencies)) for layer, dependencies in plan.items()}


def configure_shared_state(model: Any) -> None:
    """Enable the CSA2 release schedule and record the replayable layers.

    The release schedule is only valid when it knows which consumers run again
    during backward: a replayed consumer reads its entries a second time, so
    those entries must be kept. The replayable set is read off the final module
    tree, where an excluded attention is wrapped and a replayed one is not.
    """
    config = getattr(model, "config", None)
    if config is None:
        return
    config.v41_release_consumed_shared_state = True
    config.v41_replayed_csa2_layers = tuple(_replayed_attention_layers(model))


def _replayed_attention_layers(model: Any) -> list[int]:
    """Return the CSA2 layers whose attention runs again during backward."""
    from hyper_parallel.core.activation_memory.wrapper import (  # pylint: disable=C0415
        CheckpointExcludeWrapper,
    )

    excluded_inner_ids = set()
    for module in model.modules():
        if isinstance(module, CheckpointExcludeWrapper):
            excluded_inner_ids.add(id(module._wrapped_module))  # pylint: disable=W0212
    layers = []
    for module in model.modules():
        if not getattr(module, "compress_ratio", 0):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or id(module) in excluded_inner_ids:
            continue
        if isinstance(module, CheckpointExcludeWrapper):
            continue
        layers.append(int(layer_idx))
    return sorted(layers)


__all__ = ["build_shared_state_release_plan", "configure_shared_state"]
