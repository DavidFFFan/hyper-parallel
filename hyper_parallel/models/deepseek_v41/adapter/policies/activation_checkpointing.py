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
    """Keep the CSA2 layers that both publish and consume state outside replay.

    A Full layer still publishes compressed K=V and the indexer K, and its own
    attention consumes them, so it stays out of the replay for now. A Reindex
    layer only republishes a Top-K selection that the replay re-reads, which the
    phase aware publish guard already covers, and a Reuse layer never publishes
    at all. The two pure SWA layers do not touch the shared state.
    """
    del module_fqn
    return bool(getattr(module, "is_kv_source", False))


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer is one region. Only Full layers stay outside the replay,
    because they are the ones whose own attention consumes the compressed K=V
    they just published; Reindex and Reuse layers are recomputed, and their
    discrete Top-K selection is re-read from the shared state instead of being
    chosen again. The model-owned hook then enables the release schedule and
    records which consumers replay, so an entry a replayed consumer still needs
    is kept until backward is done with it.
    """
    return RecomputePolicy(
        region_patterns=("model.layers.*",),
        exclude_predicate=_exclude_state_publishers,
        on_applied=configure_shared_state,
    )


__all__ = ["build_recompute_policy"]
