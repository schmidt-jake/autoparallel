# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""The MILP backend: which one is chosen, and that both solve a placement problem alike."""

from types import SimpleNamespace

import pulp
import pytest

from autoparallel import milp
from autoparallel.optimize_sharding import ShardingOptimizer

HIGHS = pulp.HiGHS(msg=False).available()
needs_highs = pytest.mark.skipif(not HIGHS, reason="highspy not installed")


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(milp.MILP_BACKEND_ENV, raising=False)


def test_highs_is_the_default_when_installed(no_env, monkeypatch):
    monkeypatch.setattr(milp, "_highs_available", lambda: True)
    assert milp.resolve_milp_backend() == "highs"


def test_cbc_is_the_fallback_without_highspy(no_env, monkeypatch):
    monkeypatch.setattr(milp, "_highs_available", lambda: False)
    assert milp.resolve_milp_backend() == "cbc"


def test_the_environment_and_the_argument_choose(monkeypatch):
    monkeypatch.setattr(milp, "_highs_available", lambda: True)
    monkeypatch.setenv(milp.MILP_BACKEND_ENV, "CBC")
    assert milp.resolve_milp_backend() == "cbc"
    assert milp.resolve_milp_backend("highs") == "highs"


def test_an_unknown_backend_is_refused(no_env):
    with pytest.raises(ValueError, match="unknown MILP backend"):
        milp.resolve_milp_backend("gurobi")


def test_naming_highs_without_highspy_is_an_error(no_env, monkeypatch):
    monkeypatch.setattr(milp, "_highs_available", lambda: False)
    with pytest.raises(RuntimeError, match="highspy is not importable"):
        milp.resolve_milp_backend("highs")


def test_each_backend_makes_its_solver_with_the_time_limit():
    assert isinstance(
        milp.make_milp_solver("cbc", msg=False, time_limit=5), pulp.PULP_CBC_CMD
    )
    solver = milp.make_milp_solver("highs", msg=False, time_limit=5)
    assert isinstance(solver, pulp.HiGHS) and solver.timeLimit == 5


def _placement_problem():
    """Three nodes with two strategies each (one chosen per node), a transition cost
    between neighbours that disagree, and a memory row that rules out the cheapest
    choices; the optimum is found by enumeration in ``_best_by_enumeration``."""
    prob = pulp.LpProblem("placement", pulp.LpMinimize)
    x = {
        (n, s): pulp.LpVariable(f"x_{n}_{s}", cat="Binary")
        for n in "abc"
        for s in (0, 1)
    }
    cost = {
        ("a", 0): 1,
        ("a", 1): 3,
        ("b", 0): 1,
        ("b", 1): 2,
        ("c", 0): 4,
        ("c", 1): 1,
    }
    memory = {
        ("a", 0): 5,
        ("a", 1): 1,
        ("b", 0): 5,
        ("b", 1): 1,
        ("c", 0): 1,
        ("c", 1): 5,
    }
    switch = {}
    for left, right in (("a", "b"), ("b", "c")):
        for s in (0, 1):
            for t in (0, 1):
                if s != t:
                    y = pulp.LpVariable(f"y_{left}{s}_{right}{t}", lowBound=0)
                    prob += y >= x[left, s] + x[right, t] - 1
                    switch[left, s, right, t] = y
    prob += pulp.lpSum(cost[k] * v for k, v in x.items()) + pulp.lpSum(
        2 * y for y in switch.values()
    )
    for n in "abc":
        prob += pulp.lpSum(x[n, s] for s in (0, 1)) == 1
    prob += pulp.lpSum(memory[k] * v for k, v in x.items()) <= 8, "memory_budget"
    return prob, x, cost, memory


def _best_by_enumeration(cost, memory):
    best = None
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                choice = {"a": a, "b": b, "c": c}
                if sum(memory[n, s] for n, s in choice.items()) > 8:
                    continue
                value = sum(cost[n, s] for n, s in choice.items())
                value += 2 * (a != b) + 2 * (b != c)
                best = value if best is None else min(best, value)
    return best


@pytest.mark.parametrize("backend", ["cbc", pytest.param("highs", marks=needs_highs)])
def test_both_backends_find_the_optimum_through_the_optimizer_solve(backend):
    prob, x, cost, memory = _placement_problem()
    keys = {(i, s, 0, 0): x[n, s] for i, n in enumerate("abc") for s in (0, 1)}
    optimizer = SimpleNamespace(
        prob=prob,
        solver_time_limit_seconds=60,
        milp_backend=backend,
        solve_report=None,
        decision_vars={k: SimpleNamespace(var=v) for k, v in keys.items()},
        _root_to_linked={},
        _apply_memory_constraint=lambda: None,
        get_violated_constraints_log=lambda: "",
    )
    optimizer._select_keys = lambda: ShardingOptimizer._select_keys(optimizer)

    ShardingOptimizer._solve(optimizer)

    report = optimizer.solve_report
    assert report.backend == backend and report.status == "optimal"
    assert report.objective == pytest.approx(_best_by_enumeration(cost, memory))
    assert len(optimizer.selected_keys) == 3  # one strategy per node
    if backend == "highs":
        assert report.gap == pytest.approx(0.0, abs=1e-6)
        assert report.bound == pytest.approx(report.objective)
    else:
        assert report.gap is None  # PuLP's CBC interface reports no bound


@needs_highs
def test_highs_reports_an_infeasible_budget_as_infeasible():
    prob, x, *_ = _placement_problem()
    prob.constraints["memory_budget"].changeRHS(2)

    def no_walk():
        raise AssertionError(
            "HiGHS has no point: the violated-constraint walk is skipped"
        )

    optimizer = SimpleNamespace(
        prob=prob,
        solver_time_limit_seconds=60,
        milp_backend="highs",
        solve_report=None,
        _apply_memory_constraint=lambda: None,
        get_violated_constraints_log=no_walk,
    )
    with pytest.raises(RuntimeError, match="could not find a feasible solution"):
        ShardingOptimizer._solve(optimizer)
