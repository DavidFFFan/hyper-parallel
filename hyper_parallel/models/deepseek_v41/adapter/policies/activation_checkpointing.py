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


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer is one region and every CSA2 layer replays: the shared
    state keeps the discrete selections and the release schedule, so a replayed
    layer neither chooses a different key set nor publishes its state twice. The
    model-owned hook enables both behaviours once the regions are installed, and
    records which consumers replay so their entries survive into backward.
    """
    return RecomputePolicy(
        region_patterns=("model.layers.*",),
        on_applied=configure_shared_state,
    )


__all__ = ["build_recompute_policy"]
