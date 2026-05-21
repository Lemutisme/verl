"""
PDPO registration entry point.

Import this module before the training loop dispatches to the local ``pdpo``
advantage estimator.
"""

import os
import sys


_pd_reward_dir = os.path.dirname(os.path.abspath(__file__))
if _pd_reward_dir not in sys.path:
    sys.path.insert(0, _pd_reward_dir)


from pdpo_advantage import compute_pdpo_advantage  # noqa: F401


__all__ = ["compute_pdpo_advantage"]
