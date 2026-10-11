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
    kv_sources, index_sources, candidate_source = _published_sources(config)
    published = _published_entries(kv_sources, index_sources, candidate_source)
    last_consumer, unread = _last_consumers(
        config, kv_sources, index_sources, candidate_source, published
    )
    replayed = {int(layer) for layer in replayable_layers}
    return _release_plan(last_consumer, unread, replayed)


def _published_sources(config: Any) -> tuple[list[int], list[int], int]:
    """Return the KV-source layers, the index-source layers and the candidate source."""
    kv_sources = sorted(int(source) for source in config.v41_kv_source_layer_ids)
    index_sources = sorted(int(source) for source in config.v41_index_source_layer_ids)
    candidate_source = int(getattr(config, "v41_candidate_source_layer_id", -1))
    return kv_sources, index_sources, candidate_source


def _published_entries(
        kv_sources: list[int],
        index_sources: list[int],
        candidate_source: int,
) -> list[tuple[str, int]]:
    """Return every entry this topology publishes, as ``(kind, source_layer)``."""
    published = [("compressed_kv", source) for source in kv_sources]
    published += [("index_key", source) for source in kv_sources]
    published += [("topk", source) for source in index_sources]
    if candidate_source >= 0:
        published.append(("candidate", candidate_source))
    return published


def _entries_read_by(
        layer_idx: int,
        kv_sources: list[int],
        index_sources: list[int],
        candidate_source: int,
) -> list[tuple[str, int]]:
    """Return the entries one CSA2 layer reads from the most recent producers."""
    kv_source = max((source for source in kv_sources if source <= layer_idx), default=None)
    index_source = max((source for source in index_sources if source <= layer_idx), default=None)
    read: list[tuple[str, int]] = []
    if kv_source is not None:
        read.append(("compressed_kv", kv_source))
        if layer_idx in index_sources and layer_idx not in kv_sources:
            # Only a Reindex layer rescores the shared indexer K; a Reuse layer
            # consumes the Top-K selection alone.
            read.append(("index_key", kv_source))
    if index_source is not None:
        read.append(("topk", index_source))
    if 0 <= candidate_source < layer_idx:
        read.append(("candidate", candidate_source))
    return read


def _last_consumers(
        config: Any,
        kv_sources: list[int],
        index_sources: list[int],
        candidate_source: int,
        published: list[tuple[str, int]],
) -> tuple[dict[tuple[str, int], int], set[tuple[str, int]]]:
    """Return each entry's last consumer layer and the entries nobody reads."""
    ratios = list(getattr(config, "v41_compress_ratios", ()) or ())
    last_consumer: dict[tuple[str, int], int] = {}
    unread = set(published)
    for layer_idx in range(int(config.num_hidden_layers)):
        if layer_idx >= len(ratios) or int(ratios[layer_idx]) <= 0:
            continue
        for dependency in _entries_read_by(
                layer_idx, kv_sources, index_sources, candidate_source
        ):
            last_consumer[dependency] = layer_idx
            unread.discard(dependency)
    for dependency in published:
        # An entry with no consumer at all - an encoder Full layer's indexer K,
        # for example - is dead the moment its producer returns.
        last_consumer.setdefault(dependency, dependency[1])
    return last_consumer, unread


def _release_plan(
        last_consumer: dict[tuple[str, int], int],
        unread: set[tuple[str, int]],
        replayed: set[int],
) -> dict[int, tuple[tuple[str, int], ...]]:
    """Group the entries that may be dropped early by their release layer."""
    plan: dict[int, list[tuple[str, int]]] = {}
    for dependency, layer_idx in last_consumer.items():
        if dependency not in unread and layer_idx in replayed:
            # The last consumer replays, so the entry has to survive into backward.
            continue
        plan.setdefault(layer_idx, []).append(dependency)
    return {layer: tuple(sorted(dependencies)) for layer, dependencies in plan.items()}


def configure_shared_state(model: Any) -> None:
    """Enable the CSA2 release schedule and prepare the replayable layers.

    The release schedule is only valid when it knows which consumers run again
    during backward: a replayed consumer reads its entries a second time, so
    those entries must be kept. Replayed index-producing layers additionally get
    their discrete selection turned into an exclusion region, because a replay
    must neither choose a different key set nor change how many tensors the
    checkpoint sees in each phase.
    """
    config = getattr(model, "config", None)
    if config is None:
        return
    replayed = _replayed_attentions(model)
    for module in replayed:
        _pin_regions(module)
    config.v41_release_consumed_shared_state = True
    config.v41_replayed_csa2_layers = tuple(
        sorted(int(getattr(module, "layer_idx")) for module in replayed)
    )


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
    """Return the attention modules that run again during backward."""
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


__all__ = ["build_shared_state_release_plan", "configure_shared_state"]
