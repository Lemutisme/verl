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
"""PD-GDPO training entry point.

Identical to ``verl.trainer.main_ppo`` except that we use a PD-GDPO
TaskRunner that registers the ``pd_gdpo`` advantage estimator inside
the Ray actor process. The driver import is not enough because Ray
actors run in separate Python processes that do not inherit the
driver's module imports.
"""

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo  # noqa: F401  -- re-exported
from verl.utils.device import auto_set_device

# Side-effect import in the driver -- so things like ``recipe.pdpo`` are
# importable before Hydra resolves the config. The actual estimator
# registration that matters for training runs inside the Ray actor.
import recipe.pdpo  # noqa: F401


class PDGDPOTaskRunner(TaskRunner):
    """TaskRunner that imports ``recipe.pdpo`` inside the Ray actor so
    the ``pd_gdpo`` advantage estimator gets registered in the actor's
    Python process before ``trainer.fit()`` calls ``get_adv_estimator_fn``.
    """

    def run(self, config):
        import recipe.pdpo  # noqa: F401  -- registers pd_gdpo on this actor
        return super().run(config)


@hydra.main(config_path="config", config_name="pd_gdpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PDGDPOTaskRunner))


if __name__ == "__main__":
    main()
