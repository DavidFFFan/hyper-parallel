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
"""Declare the DeepSeek-V4.1 recompute regions used by normal training."""

from typing import Any

from hyper_parallel.models.adapter_spec import RecomputePolicy
from hyper_parallel.models.deepseek_v41.adapter.policies.shared_state import (
    configure_shared_state,
)


def _exclude_state_publishers(module_fqn: str, module: Any) -> bool:
    """Keep every CSA2 layer that publishes cross-layer state outside replay.

    A "Full" or "Reindex" layer publishes compressed K=V, the indexer K or the
    Top-K selection, and replaying it would publish them a second time. A
    "Reuse" layer only reads state, so it can be replayed, and the two pure SWA
    layers do not touch the shared state at all.
    """
    del module_fqn
    return bool(getattr(module, "is_index_source", False))


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer is one region. Only the CSA2 layers that publish the
    per-forward shared state stay outside the replay; the Reuse layers and the
    two pure SWA layers are recomputed, which trades one extra forward pass for
    their saved activations. The model-owned hook then enables the release
    schedule and records which consumers replay, so an entry that a replayed
    consumer still needs is kept until backward is done with it.
    """
    return RecomputePolicy(
        region_patterns=("model.layers.*",),
        exclude_predicate=_exclude_state_publishers,
        on_applied=configure_shared_state,
    )


__all__ = ["build_recompute_policy"]
