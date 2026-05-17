"""PD-GDPO: Primal-Dual Group-Decoupled Policy Optimization.

This subpackage implements primal-dual control at the advantage level rather
than the scalar-reward level. Auxiliary reward components are normalized
independently within each prompt group (GDPO-style), then aggregated with
dynamically adapted dual variables maintained by a centralized controller.

Importing this package registers the ``pd_gdpo`` advantage estimator with
verl's advantage estimator registry.
"""

from .controller import PrimalDualController, get_controller  # noqa: F401
from .advantage import compute_pd_gdpo_advantage  # noqa: F401

__all__ = ["PrimalDualController", "get_controller", "compute_pd_gdpo_advantage"]
