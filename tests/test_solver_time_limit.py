# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pulp
import pytest

from autoparallel.optimize_sharding import ShardingOptimizer


class _Problem:
    def __init__(self, status, solution_status, solution_time):
        self.status = status
        self.sol_status = solution_status
        self.solutionTime = solution_time
        self.objective = None

    def solve(self, solver):
        assert solver.timeLimit == 7


def _optimizer(problem, backend="cbc"):
    chosen = pulp.LpVariable("chosen", cat="Binary")
    chosen.varValue = 1
    other = pulp.LpVariable("other", cat="Binary")
    other.varValue = 0
    optimizer = SimpleNamespace(
        prob=problem,
        solver_time_limit_seconds=7,
        milp_backend=backend,
        solve_report=None,
        decision_vars={
            (0, 0, 0, 0): SimpleNamespace(var=chosen),
            (0, 0, 1, 0): SimpleNamespace(var=other),
        },
        _root_to_linked={(0, 0, 0, 0): [(1, 0, 0, 0)]},
        _apply_memory_constraint=lambda: None,
        get_violated_constraints_log=lambda: "",
    )
    optimizer._select_keys = lambda: ShardingOptimizer._select_keys(optimizer)
    return optimizer


@pytest.mark.parametrize(
    "status, solution_status",
    [
        (pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound),
        (pulp.LpStatusInfeasible, pulp.LpSolutionInfeasible),
    ],
)
def test_a_deadline_without_a_feasible_solution_is_a_timeout(status, solution_status):
    optimizer = _optimizer(_Problem(status, solution_status, solution_time=7))

    with pytest.raises(TimeoutError, match="found no feasible solution"):
        ShardingOptimizer._solve(optimizer)


@pytest.mark.parametrize("backend", ["cbc", "highs"])
def test_a_deadline_with_a_feasible_solution_accepts_the_incumbent(backend):
    """Both PuLP backends report a time limit holding an integer point as (Optimal,
    IntegerFeasible); the solve keeps that point instead of raising."""
    if backend == "highs" and not pulp.HiGHS(msg=False).available():
        pytest.skip("highspy not installed")
    optimizer = _optimizer(
        _Problem(pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible, 7),
        backend=backend,
    )

    ShardingOptimizer._solve(optimizer)

    assert optimizer.selected_keys == [(0, 0, 0, 0), (1, 0, 0, 0)]
    report = optimizer.solve_report
    assert report.status == "incumbent" and report.backend == backend
    # No solver model to read a bound from: the gap is reported as unknown.
    assert report.gap is None


def test_sharding_optimizer_preserves_fast_infeasibility():
    optimizer = _optimizer(
        _Problem(
            pulp.LpStatusInfeasible,
            pulp.LpSolutionInfeasible,
            solution_time=1,
        )
    )

    with pytest.raises(RuntimeError, match="could not find a feasible solution"):
        ShardingOptimizer._solve(optimizer)


@pytest.mark.parametrize(
    "status, solution_status, message",
    [
        (
            pulp.LpStatusNotSolved,
            pulp.LpSolutionNoSolutionFound,
            "solve failed",
        ),
        (
            pulp.LpStatusOptimal,
            pulp.LpSolutionIntegerFeasible,
            "non-optimal solution",
        ),
    ],
)
def test_sharding_optimizer_preserves_early_solver_failures(
    status, solution_status, message
):
    optimizer = _optimizer(_Problem(status, solution_status, solution_time=1))

    with pytest.raises(RuntimeError, match=message):
        ShardingOptimizer._solve(optimizer)


def test_a_binary_within_the_integrality_tolerance_is_selected():
    """HiGHS returns a selected binary as 0.9999999 or 1.0000001, not exactly 1."""
    optimizer = _optimizer(_Problem(pulp.LpStatusOptimal, pulp.LpSolutionOptimal, 1))
    optimizer.decision_vars[(0, 0, 0, 0)].var.varValue = 0.9999999
    optimizer.decision_vars[(0, 0, 1, 0)].var.varValue = 1e-7
    ShardingOptimizer._select_keys(optimizer)
    assert optimizer.selected_keys == [(0, 0, 0, 0), (1, 0, 0, 0)]


def test_a_node_without_a_selected_strategy_is_an_error():
    nodes = ["a", "b"]
    optimizer = SimpleNamespace(
        nodes=nodes,
        selected_keys=[(0, 0, 0, 0)],
        _resolve_decision_var=lambda key: SimpleNamespace(strategy="S"),
    )
    with pytest.raises(RuntimeError, match="selects no strategy for 1 nodes"):
        ShardingOptimizer._extract_and_validate_solution(optimizer)
