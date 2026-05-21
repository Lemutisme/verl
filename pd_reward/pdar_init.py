"""
PDAR/PDPO Registration Entry Point.

Import this module to register the local advantage estimators with verl's
``core_algos`` registry before the training loop calls ``compute_advantage()``.
"""

import os
import sys

# Ensure the pd_reward directory is on sys.path so that ``reward_score``
# can be resolved by ``pdar_advantage.py``.
_pd_reward_dir = os.path.dirname(os.path.abspath(__file__))
if _pd_reward_dir not in sys.path:
    sys.path.insert(0, _pd_reward_dir)

# These imports trigger @register_adv_est(...)
from pdar_advantage import compute_pdar_advantage  # noqa: F401
from pdpo_advantage import compute_pdpo_advantage  # noqa: F401

__all__ = ["compute_pdar_advantage", "compute_pdpo_advantage"]
