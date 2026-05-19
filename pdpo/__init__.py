# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PD-GDPO: Primal-Dual Group-Decoupled Policy Optimization.

Side-effect import: importing this package registers the ``pd_gdpo``
advantage estimator with verl's ``ADV_ESTIMATOR_REGISTRY``.
"""

from .controller import PrimalDualController, get_controller, reset_controller
from . import advantage  # noqa: F401 -- registers @register_adv_est("pd_gdpo")

__all__ = ["PrimalDualController", "get_controller", "reset_controller"]
