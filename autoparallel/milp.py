# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""The MILP backend of the sharding solve: HiGHS when installed, CBC otherwise.

The placement ILP of a large model (a whole training step of a 29B MoE has 1.8M binary
variables and 1M constraints) is beyond CBC in practical time: on one such problem CBC found
its first integer point after 938 s and proved nothing, where HiGHS found a feasible point at
68 s and proved the same optimum at 108 s. Both are reached through PuLP: HiGHS through
``highspy`` (``pulp.HiGHS``, in process, no problem files) and CBC through PuLP's bundled
binary (``pulp.PULP_CBC_CMD``).

The backend is ``highs`` when ``highspy`` is importable and ``cbc`` otherwise, or what the
``AUTOPARALLEL_MILP`` environment variable (``highs`` / ``cbc``) or an explicit
``milp_backend`` argument names.

A solve that reaches its time limit holding an integer-feasible point is accepted as an
*incumbent*: :class:`SolveReport` records the backend, the status, the objective and, where
the backend reports them, the dual bound and the relative MIP gap, so the caller can judge it.
A time limit with no feasible point is still a ``TimeoutError``.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import pulp

__all__ = [
    "MILP_BACKENDS",
    "MILP_BACKEND_ENV",
    "SolveReport",
    "make_milp_solver",
    "resolve_milp_backend",
    "solve_report",
]

logger = logging.getLogger(__name__)

MILP_BACKEND_ENV = "AUTOPARALLEL_MILP"
MILP_BACKENDS = ("highs", "cbc")


def _highs_available() -> bool:
    return pulp.HiGHS(msg=False).available()


def resolve_milp_backend(requested: str | None = None) -> str:
    """The backend to solve with: ``requested``, else ``$AUTOPARALLEL_MILP``, else HiGHS if
    ``highspy`` is importable, else CBC. Naming HiGHS when it is not installed raises."""
    choice = (requested or os.environ.get(MILP_BACKEND_ENV, "")).strip().lower()
    if not choice:
        return "highs" if _highs_available() else "cbc"
    if choice not in MILP_BACKENDS:
        raise ValueError(
            f"unknown MILP backend {choice!r}; expected one of {MILP_BACKENDS}"
        )
    if choice == "highs" and not _highs_available():
        raise RuntimeError(
            "the HiGHS MILP backend was requested but highspy is not importable "
            "(pip install highspy)"
        )
    return choice


def make_milp_solver(
    backend: str, *, msg: bool, time_limit: float | None
) -> pulp.LpSolver:
    """The PuLP solver object of ``backend``, with its time limit."""
    if backend == "highs":
        return pulp.HiGHS(msg=msg, timeLimit=time_limit)
    if backend == "cbc":
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    raise ValueError(
        f"unknown MILP backend {backend!r}; expected one of {MILP_BACKENDS}"
    )


@dataclass(frozen=True)
class SolveReport:
    """How a sharding solve ended.

    ``status`` is ``optimal`` (proven within the solver's gap tolerance) or ``incumbent``
    (the best integer-feasible point when the time limit was reached). ``bound`` and ``gap``
    (relative, ``|objective - bound| / |objective|``) are ``None`` where the backend does not
    report a dual bound (PuLP's CBC interface).
    """

    backend: str
    status: str
    seconds: float
    objective: float | None
    bound: float | None
    gap: float | None

    def describe(self) -> str:
        gap = "unknown" if self.gap is None else f"{self.gap:.3%}"
        return (
            f"{self.backend} {self.status} in {self.seconds:.1f}s, "
            f"objective={self.objective}, bound={self.bound}, gap={gap}"
        )


def _highs_info(problem: pulp.LpProblem) -> tuple[float | None, float | None]:
    """HiGHS's dual bound and relative gap, read off the model PuLP kept."""
    model = getattr(problem, "solverModel", None)
    if model is None or not hasattr(model, "getInfo"):
        return None, None
    info = model.getInfo()
    bound = float(info.mip_dual_bound)
    gap = float(info.mip_gap)
    return (
        bound if math.isfinite(bound) else None,
        gap if math.isfinite(gap) else None,
    )


def solve_report(
    problem: pulp.LpProblem, backend: str, *, incumbent: bool
) -> SolveReport:
    """The report of a finished solve (``incumbent``: accepted at the time limit)."""
    objective = None if problem.objective is None else pulp.value(problem.objective)
    bound, gap = _highs_info(problem) if backend == "highs" else (None, None)
    return SolveReport(
        backend=backend,
        status="incumbent" if incumbent else "optimal",
        seconds=float(problem.solutionTime),
        objective=None if objective is None else float(objective),
        bound=bound,
        gap=gap,
    )
