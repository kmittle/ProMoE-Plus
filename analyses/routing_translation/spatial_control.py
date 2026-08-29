"""Exact feasibility diagnostics for spatially matched routing controls."""

from __future__ import annotations

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_array


_MILP_STATUS_NAMES = {
    0: "optimal",
    1: "limit_reached",
    2: "infeasible",
    3: "unbounded",
    4: "solver_error",
}


def _grid_incident_edges(changed_mask, grid_size):
    token_count = int(changed_mask.size)
    if grid_size < 2 or grid_size * grid_size != token_count:
        raise ValueError("Four-neighbor statistics require a square grid")
    token_grid = np.arange(token_count, dtype=np.int64).reshape(
        grid_size,
        grid_size,
    )
    left = np.concatenate((
        token_grid[:, :-1].reshape(-1),
        token_grid[:-1].reshape(-1),
    ))
    right = np.concatenate((
        token_grid[:, 1:].reshape(-1),
        token_grid[1:].reshape(-1),
    ))
    active = changed_mask[left] | changed_mask[right]
    return list(zip(left[active].tolist(), right[active].tolist()))


def _pair_id(left_expert, right_expert, num_routed_experts):
    lower = min(int(left_expert), int(right_expert))
    upper = max(int(left_expert), int(right_expert))
    return lower * num_routed_experts + upper


def _four_neighbor_pair_counts(
    route_ids,
    changed_mask,
    grid_size,
    num_routed_experts,
):
    counts = np.zeros(num_routed_experts * num_routed_experts, dtype=np.int64)
    edges = _grid_incident_edges(changed_mask, grid_size)
    if not edges:
        raise ValueError("Spatial control needs at least one incident grid edge")
    for left, right in edges:
        counts[_pair_id(
            route_ids[left],
            route_ids[right],
            num_routed_experts,
        )] += 1
    return counts, edges


def _validate_exact_inputs(
    native_ids,
    content_ids,
    changed_mask,
    grid_size,
    num_routed_experts,
    minimum_mismatches,
):
    native_ids = np.asarray(native_ids, dtype=np.int64)
    content_ids = np.asarray(content_ids, dtype=np.int64)
    changed_mask = np.asarray(changed_mask)
    if not (
        native_ids.ndim == content_ids.ndim == changed_mask.ndim == 1
    ):
        raise ValueError("Exact spatial-control inputs must be flat vectors")
    if not (
        native_ids.shape == content_ids.shape == changed_mask.shape
    ):
        raise ValueError("Exact spatial-control inputs must align")
    if changed_mask.dtype != np.bool_:
        raise ValueError("Changed mask must be boolean")
    if num_routed_experts < 1:
        raise ValueError("Number of routed experts must be positive")
    if native_ids.size and (
        native_ids.min() < 0
        or content_ids.min() < 0
        or native_ids.max() >= num_routed_experts
        or content_ids.max() >= num_routed_experts
    ):
        raise ValueError("Route IDs are outside the routed expert range")
    if not np.array_equal(changed_mask, native_ids != content_ids):
        raise ValueError("Changed mask must equal the content disagreement support")
    changed_count = int(changed_mask.sum())
    if not 0 <= int(minimum_mismatches) <= changed_count:
        raise ValueError("Minimum mismatches must fit the changed support")
    _grid_incident_edges(changed_mask, grid_size)
    return native_ids, content_ids, changed_mask


def _maximum_mismatch_assignment(native_changed, content_changed):
    changed_count = int(content_changed.size)
    if changed_count == 0:
        return content_changed.copy(), 0

    expert_items = content_changed.copy()
    allowed = expert_items[None, :] != native_changed[:, None]
    mismatch = expert_items[None, :] != content_changed[:, None]
    forbidden_cost = changed_count + 1
    costs = np.where(allowed, -mismatch.astype(np.int64), forbidden_cost)
    rows, columns = linear_sum_assignment(costs)
    if not allowed[rows, columns].all():
        raise RuntimeError("The native-conflict assignment unexpectedly became infeasible")
    assignment = np.empty_like(content_changed)
    assignment[rows] = expert_items[columns]
    mismatch_count = int((assignment != content_changed).sum())
    return assignment, mismatch_count


def _append_constraint(
    coefficients,
    lower,
    upper,
    row_indices,
    column_indices,
    values,
    row_lower,
    row_upper,
):
    row = len(row_lower)
    for column, value in coefficients.items():
        if value:
            row_indices.append(row)
            column_indices.append(column)
            values.append(float(value))
    row_lower.append(float(lower))
    row_upper.append(float(upper))


def _minimum_pair_l1_milp(
    native_ids,
    content_ids,
    changed_mask,
    grid_size,
    num_routed_experts,
    minimum_mismatches,
    time_limit_seconds=None,
):
    changed_indices = np.flatnonzero(changed_mask)
    changed_count = int(changed_indices.size)
    native_changed = native_ids[changed_indices]
    content_changed = content_ids[changed_indices]
    expert_counts = np.bincount(
        content_changed,
        minlength=num_routed_experts,
    )
    active_experts = np.flatnonzero(expert_counts).tolist()
    local_index = {
        int(token_index): local
        for local, token_index in enumerate(changed_indices.tolist())
    }

    variable_count = 0
    x_indices = []
    for local in range(changed_count):
        allowed = {
            expert: variable_count + offset
            for offset, expert in enumerate(
                expert
                for expert in active_experts
                if expert != int(native_changed[local])
            )
        }
        if not allowed:
            raise RuntimeError("A changed token has no support-preserving expert")
        variable_count += len(allowed)
        x_indices.append(allowed)

    reference_counts, edges = _four_neighbor_pair_counts(
        content_ids,
        changed_mask,
        grid_size,
        num_routed_experts,
    )
    pair_expressions = {
        lower * num_routed_experts + upper: {}
        for lower in range(num_routed_experts)
        for upper in range(lower, num_routed_experts)
    }
    joint_edges = []
    for left_token, right_token in edges:
        left_changed = bool(changed_mask[left_token])
        right_changed = bool(changed_mask[right_token])
        if left_changed and right_changed:
            left_local = local_index[left_token]
            right_local = local_index[right_token]
            joint = {}
            for left_expert in x_indices[left_local]:
                for right_expert in x_indices[right_local]:
                    joint[(left_expert, right_expert)] = variable_count
                    pair = _pair_id(
                        left_expert,
                        right_expert,
                        num_routed_experts,
                    )
                    pair_expressions[pair][variable_count] = 1.0
                    variable_count += 1
            joint_edges.append((left_local, right_local, joint))
        else:
            changed_token = left_token if left_changed else right_token
            fixed_token = right_token if left_changed else left_token
            changed_local = local_index[changed_token]
            fixed_expert = int(content_ids[fixed_token])
            for expert, variable in x_indices[changed_local].items():
                pair = _pair_id(expert, fixed_expert, num_routed_experts)
                pair_expressions[pair][variable] = (
                    pair_expressions[pair].get(variable, 0.0) + 1.0
                )

    deviation_indices = {}
    for pair in pair_expressions:
        deviation_indices[pair] = variable_count
        variable_count += 1

    objective = np.zeros(variable_count, dtype=np.float64)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)
    integrality = np.zeros(variable_count, dtype=np.uint8)
    for allowed in x_indices:
        for variable in allowed.values():
            integrality[variable] = 1
    for variable in deviation_indices.values():
        objective[variable] = 1.0
        upper_bounds[variable] = np.inf

    row_indices = []
    column_indices = []
    values = []
    row_lower = []
    row_upper = []

    for allowed in x_indices:
        _append_constraint(
            {variable: 1.0 for variable in allowed.values()},
            1.0,
            1.0,
            row_indices,
            column_indices,
            values,
            row_lower,
            row_upper,
        )
    for expert in active_experts:
        _append_constraint(
            {
                allowed[expert]: 1.0
                for allowed in x_indices
                if expert in allowed
            },
            expert_counts[expert],
            expert_counts[expert],
            row_indices,
            column_indices,
            values,
            row_lower,
            row_upper,
        )
    mismatch_coefficients = {}
    for local, allowed in enumerate(x_indices):
        for expert, variable in allowed.items():
            if expert != int(content_changed[local]):
                mismatch_coefficients[variable] = 1.0
    _append_constraint(
        mismatch_coefficients,
        minimum_mismatches,
        np.inf,
        row_indices,
        column_indices,
        values,
        row_lower,
        row_upper,
    )

    for left_local, right_local, joint in joint_edges:
        for left_expert, x_variable in x_indices[left_local].items():
            coefficients = {x_variable: -1.0}
            coefficients.update({
                variable: 1.0
                for (candidate_left, _), variable in joint.items()
                if candidate_left == left_expert
            })
            _append_constraint(
                coefficients,
                0.0,
                0.0,
                row_indices,
                column_indices,
                values,
                row_lower,
                row_upper,
            )
        for right_expert, x_variable in x_indices[right_local].items():
            coefficients = {x_variable: -1.0}
            coefficients.update({
                variable: 1.0
                for (_, candidate_right), variable in joint.items()
                if candidate_right == right_expert
            })
            _append_constraint(
                coefficients,
                0.0,
                0.0,
                row_indices,
                column_indices,
                values,
                row_lower,
                row_upper,
            )

    for pair, expression in pair_expressions.items():
        deviation = deviation_indices[pair]
        positive = dict(expression)
        positive[deviation] = -1.0
        _append_constraint(
            positive,
            -np.inf,
            reference_counts[pair],
            row_indices,
            column_indices,
            values,
            row_lower,
            row_upper,
        )
        negative = {
            variable: -coefficient
            for variable, coefficient in expression.items()
        }
        negative[deviation] = -1.0
        _append_constraint(
            negative,
            -np.inf,
            -reference_counts[pair],
            row_indices,
            column_indices,
            values,
            row_lower,
            row_upper,
        )

    constraints = coo_array(
        (values, (row_indices, column_indices)),
        shape=(len(row_lower), variable_count),
        dtype=np.float64,
    ).tocsc()
    options = {"disp": False, "presolve": True, "mip_rel_gap": 0.0}
    if time_limit_seconds is not None:
        time_limit_seconds = float(time_limit_seconds)
        if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
            raise ValueError("MILP time limit must be finite and positive")
        options["time_limit"] = time_limit_seconds
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            constraints,
            np.asarray(row_lower, dtype=np.float64),
            np.asarray(row_upper, dtype=np.float64),
        ),
        options=options,
    )
    status = _MILP_STATUS_NAMES.get(int(result.status), "unknown")
    metadata = {
        "status": status,
        "status_code": int(result.status),
        "message": str(result.message),
        "proven_optimal": bool(result.status == 0 and result.success),
        "variables": int(variable_count),
        "binary_variables": int(integrality.sum()),
        "constraints": int(len(row_lower)),
        "time_limit_seconds": time_limit_seconds,
    }
    if not metadata["proven_optimal"]:
        return None, None, metadata

    assignment = np.empty(changed_count, dtype=np.int64)
    for local, allowed in enumerate(x_indices):
        selected = [
            expert
            for expert, variable in allowed.items()
            if result.x[variable] >= 0.5
        ]
        if len(selected) != 1:
            raise RuntimeError("MILP did not return a one-hot expert assignment")
        assignment[local] = selected[0]
    candidate = native_ids.copy()
    candidate[changed_indices] = assignment
    if not np.array_equal(candidate != native_ids, changed_mask):
        raise RuntimeError("MILP solution changed the intervention support")
    if not np.array_equal(
        np.bincount(assignment, minlength=num_routed_experts),
        expert_counts,
    ):
        raise RuntimeError("MILP solution changed the replacement histogram")
    mismatch_count = int((assignment != content_changed).sum())
    if mismatch_count < minimum_mismatches:
        raise RuntimeError("MILP solution violated the derangement constraint")

    candidate_counts, _ = _four_neighbor_pair_counts(
        candidate,
        changed_mask,
        grid_size,
        num_routed_experts,
    )
    pair_l1 = int(np.abs(candidate_counts - reference_counts).sum())
    if not np.isclose(float(result.fun), pair_l1, atol=1e-6, rtol=0.0):
        raise RuntimeError("MILP objective disagrees with reconstructed pair counts")
    adjacency_tv = 0.5 * pair_l1 / len(edges)
    return candidate, float(adjacency_tv), metadata


def exact_spatial_control_diagnostic(
    native_ids,
    content_ids,
    changed_mask,
    grid_size,
    num_routed_experts,
    minimum_mismatches,
    max_adjacency_tv,
    random_adjacency_tv,
    time_limit_seconds=None,
):
    """Prove derangement reachability and the best possible adjacency TV."""
    native_ids, content_ids, changed_mask = _validate_exact_inputs(
        native_ids,
        content_ids,
        changed_mask,
        grid_size,
        num_routed_experts,
        minimum_mismatches,
    )
    changed_indices = np.flatnonzero(changed_mask)
    changed_count = int(changed_indices.size)
    if changed_count == 0:
        return None, {
            "solver": "Hungarian assignment followed by scipy.optimize.milp",
            "scipy_version": scipy.__version__,
            "maximum_mismatch_proven_optimal": True,
            "maximum_mismatches": 0,
            "maximum_derangement": None,
            "minimum_required_mismatches": int(minimum_mismatches),
            "derangement_feasible": False,
            "milp": None,
            "minimum_adjacency_tv": None,
            "meets_maximum_adjacency_tv": None,
            "not_worse_than_random": None,
            "all_acceptance_constraints_feasible": False,
        }

    _, maximum_mismatches = _maximum_mismatch_assignment(
        native_ids[changed_indices],
        content_ids[changed_indices],
    )
    derangement_feasible = maximum_mismatches >= minimum_mismatches
    diagnostics = {
        "solver": "Hungarian assignment followed by scipy.optimize.milp",
        "scipy_version": scipy.__version__,
        "maximum_mismatch_proven_optimal": True,
        "maximum_mismatches": int(maximum_mismatches),
        "maximum_derangement": float(maximum_mismatches / changed_count),
        "minimum_required_mismatches": int(minimum_mismatches),
        "derangement_feasible": bool(derangement_feasible),
        "milp": None,
        "minimum_adjacency_tv": None,
        "meets_maximum_adjacency_tv": None,
        "not_worse_than_random": None,
        "all_acceptance_constraints_feasible": None,
    }
    if not derangement_feasible:
        diagnostics["all_acceptance_constraints_feasible"] = False
        return None, diagnostics

    candidate, minimum_tv, milp_metadata = _minimum_pair_l1_milp(
        native_ids=native_ids,
        content_ids=content_ids,
        changed_mask=changed_mask,
        grid_size=grid_size,
        num_routed_experts=num_routed_experts,
        minimum_mismatches=minimum_mismatches,
        time_limit_seconds=time_limit_seconds,
    )
    diagnostics["milp"] = milp_metadata
    if not milp_metadata["proven_optimal"]:
        return None, diagnostics

    meets_maximum_tv = minimum_tv <= float(max_adjacency_tv) + 1e-12
    not_worse_than_random = (
        minimum_tv <= float(random_adjacency_tv) + 1e-12
    )
    diagnostics.update({
        "minimum_adjacency_tv": float(minimum_tv),
        "meets_maximum_adjacency_tv": bool(meets_maximum_tv),
        "not_worse_than_random": bool(not_worse_than_random),
        "all_acceptance_constraints_feasible": bool(
            meets_maximum_tv and not_worse_than_random
        ),
    })
    return candidate, diagnostics
