"""
PDAR Registration Entry Point.

Import this module to register the ``"pdar"`` advantage estimator with
verl's ``core_algos`` registry.  The import is triggered from
``custom_reward.py`` when ``combine_mode == "pdar"``, ensuring the estimator
is available before the training loop calls ``compute_advantage()``.
"""

import os
import sys

# Ensure the pd_reward directory is on sys.path so that ``reward_score``
# can be resolved by ``pdar_advantage.py``.
_pd_reward_dir = os.path.dirname(os.path.abspath(__file__))
if _pd_reward_dir not in sys.path:
    sys.path.insert(0, _pd_reward_dir)

# The import triggers @register_adv_est("pdar")
from pdar_advantage import compute_pdar_advantage  # noqa: F401

__all__ = ["compute_pdar_advantage"]
