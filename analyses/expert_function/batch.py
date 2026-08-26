"""Locked batch protocol for the expert-function transport mechanism gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .consistency_probe import (
    ALL_METRICS,
    PRIMARY_METRIC,
    PROBE_VERSION,
    ROUTER_METRIC,
    summarize_token,
    summarize_tokens,
)


BATCH_VERSION = 7
MANIFEST_NAME = "function_transport_gate_v1"
MODEL_NAME = "ProMoE_TC_B"
CHECKPOINT_STEP = 50000
CHECKPOINT_STATE = "ema_model_state_dict"
BLOCK_INDEX = 3
NUM_ROUTED_EXPERTS = 12
NUM_TOKEN_PROBES = 8
EXACT_BATCH_SIZE = 24
SIGMAS = (0.276, 0.5, 0.724)
SHIFTS = ((0, 2), (0, -2), (2, 0), (-2, 0))
EXPECTED_WEIGHTS_SHA256 = (
    "cecc673a4541facf94595bee5a7090314fdb129672634694bbdfb335780606e8"
)
SELECTION_SALT = "promoe-expert-function-transport-confirmatory-v1-20260826"
EXCLUDED_LABELS = (
    0, 36, 69, 96, 100, 113, 120, 131, 142, 144, 159, 169,
    211, 250, 300, 301, 316, 346, 351, 381, 384, 421, 442,
    445, 451, 465, 489, 500, 527, 532, 545, 604, 683, 724,
    735, 738, 748, 750, 764, 774, 780, 788, 815, 884, 902,
    914, 954, 956, 971, 983, 987, 995, 998, 999,
)
CASE_SPECS = (
    ("class078_n01776313_23498", 78, 1834169401, "n01776313",
     "n01776313/n01776313_23498.latent.npz"),
    ("class853_n04417672_823", 853, 1704155487, "n04417672",
     "n04417672/n04417672_823.latent.npz"),
    ("class594_n03495258_1818", 594, 630088202, "n03495258",
     "n03495258/n03495258_1818.latent.npz"),
    ("class620_n03642806_3195", 620, 1636379026, "n03642806",
     "n03642806/n03642806_3195.latent.npz"),
    ("class992_n12998815_8768", 992, 626966951, "n12998815",
     "n12998815/n12998815_8768.latent.npz"),
    ("class588_n03482405_25006", 588, 901923975, "n03482405",
     "n03482405/n03482405_25006.latent.npz"),
    ("class106_n01883070_4993", 106, 1916276538, "n01883070",
     "n01883070/n01883070_4993.latent.npz"),
    ("class589_n03483316_15544", 589, 1723597068, "n03483316",
     "n03483316/n03483316_15544.latent.npz"),
    ("class599_n03530642_29319", 599, 1587056499, "n03530642",
     "n03530642/n03530642_29319.latent.npz"),
    ("class156_n02086646_4478", 156, 1785568368, "n02086646",
     "n02086646/n02086646_4478.latent.npz"),
    ("class887_n04532106_2486", 887, 230187236, "n04532106",
     "n04532106/n04532106_2486.latent.npz"),
    ("class656_n03770679_9172", 656, 930732017, "n03770679",
     "n03770679/n03770679_9172.latent.npz"),
    ("class725_n03950228_15972", 725, 1084842906, "n03950228",
     "n03950228/n03950228_15972.latent.npz"),
    ("class658_n03775071_14743", 658, 593837238, "n03775071",
     "n03775071/n03775071_14743.latent.npz"),
    ("class416_n02777292_5880", 416, 1580839236, "n02777292",
     "n02777292/n02777292_5880.latent.npz"),
    ("class945_n07720875_8054", 945, 755335173, "n07720875",
     "n07720875/n07720875_8054.latent.npz"),
    ("class870_n04482393_5171", 870, 1653554661, "n04482393",
     "n04482393/n04482393_5171.latent.npz"),
    ("class188_n02095314_3318", 188, 1128192249, "n02095314",
     "n02095314/n02095314_3318.latent.npz"),
    ("class710_n03908714_5270", 710, 1216353048, "n03908714",
     "n03908714/n03908714_5270.latent.npz"),
    ("class150_n02077923_10933", 150, 1013805366, "n02077923",
     "n02077923/n02077923_10933.latent.npz"),
    ("class224_n02105056_570", 224, 582140143, "n02105056",
     "n02105056/n02105056_570.latent.npz"),
    ("class615_n03623198_3727", 615, 116950104, "n03623198",
     "n03623198/n03623198_3727.latent.npz"),
    ("class238_n02107574_1737", 238, 896123997, "n02107574",
     "n02107574/n02107574_1737.latent.npz"),
    ("class833_n04347754_51551", 833, 1192711317, "n04347754",
     "n04347754/n04347754_51551.latent.npz"),
)
GATE_REQUIREMENTS = {
    "expected_case_count": 24,
    "expected_cells_per_case": 12,
    "expected_tokens_per_cell": 8,
    "bootstrap_resamples": 200000,
    "bootstrap_seed": 2026082602,
    "minimum_valid_fraction_per_image": 0.95,
    "minimum_mean_primary_spearman": 0.10,
    "minimum_positive_images": 15,
    "require_every_sigma_positive": True,
    "minimum_mean_primary_minus_router_spearman": 0.05,
    "require_primary_ci_lower_positive": True,
    "require_delta_ci_lower_positive": True,
    "require_positive_native_router_weight": True,
    "maximum_noop_abs_mse_change": 1e-12,
    "maximum_noop_abs_output_change": 0.0,
    "maximum_forced_unforced_abs_mse_change": 5e-8,
    "maximum_forced_unforced_abs_output_change": 5e-6,
}


def _require_int(value, name, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _require_finite_number(value, name, minimum=None, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _validate_requirements(requirements):
    if set(requirements) != set(GATE_REQUIREMENTS):
        raise ValueError("Gate requirements must use the complete locked key set")
    expected_cases = _require_int(
        requirements["expected_case_count"],
        "expected_case_count",
        1,
    )
    expected_cells = _require_int(
        requirements["expected_cells_per_case"],
        "expected_cells_per_case",
        1,
    )
    expected_tokens = _require_int(
        requirements["expected_tokens_per_cell"],
        "expected_tokens_per_cell",
        1,
    )
    if expected_cases != len(CASE_SPECS):
        raise ValueError("expected_case_count must match the locked manifest")
    if expected_cells != len(SIGMAS) * len(SHIFTS):
        raise ValueError("expected_cells_per_case must match sigma-shift grid")
    if expected_tokens != NUM_TOKEN_PROBES:
        raise ValueError("expected_tokens_per_cell must match the locked probe")
    _require_int(requirements["bootstrap_resamples"], "bootstrap_resamples", 1000)
    _require_int(requirements["bootstrap_seed"], "bootstrap_seed", 0)
    _require_finite_number(
        requirements["minimum_valid_fraction_per_image"],
        "minimum_valid_fraction_per_image",
        0.0,
        1.0,
    )
    _require_finite_number(
        requirements["minimum_mean_primary_spearman"],
        "minimum_mean_primary_spearman",
        -1.0,
        1.0,
    )
    _require_int(
        requirements["minimum_positive_images"],
        "minimum_positive_images",
        0,
        expected_cases,
    )
    _require_finite_number(
        requirements["minimum_mean_primary_minus_router_spearman"],
        "minimum_mean_primary_minus_router_spearman",
        -2.0,
        2.0,
    )
    for name in (
        "require_every_sigma_positive",
        "require_primary_ci_lower_positive",
        "require_delta_ci_lower_positive",
        "require_positive_native_router_weight",
    ):
        if not isinstance(requirements[name], bool):
            raise ValueError(f"{name} must be a boolean")
    for name in (
        "maximum_noop_abs_mse_change",
        "maximum_noop_abs_output_change",
        "maximum_forced_unforced_abs_mse_change",
        "maximum_forced_unforced_abs_output_change",
    ):
        _require_finite_number(requirements[name], name, 0.0)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _selection_payload():
    return {
        "locked_before_confirmatory_results": True,
        "salt": SELECTION_SALT,
        "class_rule": (
            "Exclude all labels used by prior probes, then take the first 24 "
            "classes after sorting SHA256(salt|label03|synset)."
        ),
        "latent_rule": (
            "Sort latent basenames and select "
            "int(SHA256(salt|latent|label03|synset)[0:8],16) modulo class count."
        ),
        "seed_rule": (
            "int(SHA256(salt|latent|label03|synset)[8:16],16) modulo "
            "2147483647."
        ),
        "excluded_labels": list(EXCLUDED_LABELS),
    }


def canonical_manifest_payload():
    return {
        "version": 1,
        "name": MANIFEST_NAME,
        "selection": _selection_payload(),
        "cases": [
            {
                "id": case_id,
                "label": label,
                "seed": seed,
                "synset": synset,
                "latent": latent,
            }
            for case_id, label, seed, synset, latent in CASE_SPECS
        ],
    }


def _class_digest(label, synset):
    payload = f"{SELECTION_SALT}|{label:03d}|{synset}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _latent_digest(label, synset):
    payload = f"{SELECTION_SALT}|latent|{label:03d}|{synset}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(manifest_path, latent_root):
    manifest_path = Path(manifest_path).resolve()
    latent_root = Path(latent_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if not latent_root.is_dir():
        raise NotADirectoryError(f"Latent root does not exist: {latent_root}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload != canonical_manifest_payload():
        raise ValueError("Manifest does not match the locked function-transport gate")

    class_names = sorted(
        path.name
        for path in latent_root.iterdir()
        if path.is_dir()
        and len(path.name) == 9
        and path.name.startswith("n")
        and path.name[1:].isdigit()
    )
    if len(class_names) != 1000:
        raise ValueError(
            f"Expected 1000 ImageNet synset directories, found {len(class_names)}"
        )

    excluded = set(EXCLUDED_LABELS)
    selected = sorted(
        (
            _class_digest(label, synset),
            label,
            synset,
        )
        for label, synset in enumerate(class_names)
        if label not in excluded
    )[:len(CASE_SPECS)]
    expected_class_pairs = [(label, synset) for _, label, synset in selected]
    actual_class_pairs = [
        (case["label"], case["synset"])
        for case in payload["cases"]
    ]
    if actual_class_pairs != expected_class_pairs:
        raise ValueError("Manifest classes do not reproduce the locked hash selection")

    cases = []
    for raw_case in payload["cases"]:
        label = raw_case["label"]
        synset = raw_case["synset"]
        if class_names[label] != synset:
            raise ValueError(
                f"Label {label} maps to {class_names[label]}, not {synset}"
            )
        class_dir = latent_root / synset
        latent_names = sorted(
            path.name
            for path in class_dir.iterdir()
            if path.is_file() and path.name.endswith(".latent.npz")
        )
        if not latent_names:
            raise FileNotFoundError(f"No latent files found in {class_dir}")
        digest = _latent_digest(label, synset)
        selected_name = latent_names[int(digest[:8], 16) % len(latent_names)]
        selected_seed = int(digest[8:16], 16) % 2147483647
        if Path(raw_case["latent"]).name != selected_name:
            raise ValueError(
                f"Case {raw_case['id']} does not reproduce latent selection"
            )
        if raw_case["seed"] != selected_seed:
            raise ValueError(f"Case {raw_case['id']} does not reproduce its seed")
        relative = Path(raw_case["latent"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] != synset
        ):
            raise ValueError(f"Case {raw_case['id']} has an unsafe latent path")
        latent_path = (latent_root / relative).resolve()
        try:
            latent_path.relative_to(latent_root)
        except ValueError as error:
            raise ValueError(f"Case {raw_case['id']} escapes latent root") from error
        if not latent_path.is_file():
            raise FileNotFoundError(f"Case latent does not exist: {latent_path}")
        cases.append({
            **raw_case,
            "latent": str(latent_path),
            "latent_relative": relative.as_posix(),
            "latent_key": "latent",
            "latent_sha256": sha256_file(latent_path),
        })

    return {
        "version": payload["version"],
        "name": payload["name"],
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "latent_root": str(latent_root),
        "selection": payload["selection"],
        "cases": cases,
    }


def _assert_nested_close(actual, expected, path="value"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path} keys differ")
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} list shape differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_close(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    if isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError(f"{path} differs: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise ValueError(f"{path} differs: {actual!r} != {expected!r}")


def _recompute_cell(cell, case_id):
    if cell.get("sampled_tokens") != NUM_TOKEN_PROBES:
        raise ValueError(f"{case_id}: cell token count differs")
    if cell.get("valid_tokens", 0) < NUM_TOKEN_PROBES:
        raise ValueError(f"{case_id}: cell has too few valid translated tokens")
    if not math.isfinite(float(cell.get("shifted_native_mse", float("nan")))):
        raise ValueError(f"{case_id}: shifted native MSE is not finite")

    candidate_groups = defaultdict(list)
    for candidate in cell.get("candidates", []):
        key = (
            candidate.get("token_index"),
            candidate.get("content_source_index"),
        )
        candidate_groups[key].append(candidate)
    if len(candidate_groups) != NUM_TOKEN_PROBES:
        raise ValueError(f"{case_id}: candidate token groups differ")

    recomputed = []
    observations = []
    native_exact_changes = []
    for cached_token in cell.get("tokens", []):
        key = (
            cached_token.get("token_index"),
            cached_token.get("content_source_index"),
        )
        candidates = candidate_groups.pop(key, None)
        if candidates is None or len(candidates) != NUM_ROUTED_EXPERTS:
            raise ValueError(f"{case_id}: candidate expert grid is incomplete")
        candidates = sorted(candidates, key=lambda row: row.get("expert", -1))
        if [row.get("expert") for row in candidates] != list(
            range(NUM_ROUTED_EXPERTS)
        ):
            raise ValueError(f"{case_id}: candidate expert IDs are not canonical")
        native_rows = [row for row in candidates if row.get("is_native") is True]
        if len(native_rows) != 1:
            raise ValueError(f"{case_id}: candidate grid must name one native expert")
        native_expert = int(native_rows[0]["expert"])
        exact = np.asarray(
            [row["exact_mse_change"] for row in candidates],
            dtype=np.float64,
        )
        metric_scores = {
            metric: np.asarray(
                [row["scores"][metric] for row in candidates],
                dtype=np.float64,
            )
            for metric in ALL_METRICS
        }
        if any(set(row.get("scores", {})) != set(ALL_METRICS) for row in candidates):
            raise ValueError(f"{case_id}: candidate metrics differ")
        token = summarize_token(metric_scores, exact, native_expert)
        native_exact_changes.append(abs(float(exact[native_expert])))
        token.update({
            "token_index": int(key[0]),
            "content_source_index": int(key[1]),
        })
        _assert_nested_close(cached_token, token, f"{case_id}.token")
        recomputed.append(token)
        primary = token["metrics"][PRIMARY_METRIC]
        router = token["metrics"][ROUTER_METRIC]
        primary_rho = primary["spearman_with_exact_utility"]
        router_rho = router["spearman_with_exact_utility"]
        observations.append({
            "sigma": float(cell["sigma"]),
            "shift": list(cell["shift_latent"]),
            "token_index": int(key[0]),
            "primary_spearman": primary_rho,
            "router_spearman": router_rho,
            "primary_minus_router_spearman": (
                primary_rho - router_rho
                if primary_rho is not None and router_rho is not None
                else None
            ),
            "native_router_weight": token["native_router_weight"],
            "exact_mse_change_range": token["exact_mse_change_range"],
            "primary_selected_beats_native": primary["selected_beats_native"],
            "primary_oracle_top3": primary["oracle_in_top3"],
        })
    if candidate_groups:
        raise ValueError(f"{case_id}: candidate records include unknown tokens")
    if len(recomputed) != NUM_TOKEN_PROBES:
        raise ValueError(f"{case_id}: cached token list is incomplete")
    _assert_nested_close(
        cell.get("summary"),
        summarize_tokens(recomputed),
        f"{case_id}.cell_summary",
    )

    controls = cell.get("numerical_controls", {})
    expected_control_keys = {
        "max_abs_noop_mse_change",
        "max_abs_noop_output_change",
        "max_abs_forced_unforced_output_change",
        "max_abs_forced_unforced_mse_change",
    }
    if set(controls) != expected_control_keys:
        raise ValueError(f"{case_id}: numerical-control keys differ")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in controls.values()
    ):
        raise ValueError(f"{case_id}: numerical controls must be finite and nonnegative")
    recomputed_noop_mse = max(native_exact_changes)
    if not math.isclose(
        recomputed_noop_mse,
        float(controls["max_abs_noop_mse_change"]),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{case_id}: no-op MSE control differs from candidate records"
        )
    return recomputed, observations, {key: float(value) for key, value in controls.items()}


def validate_case_result(result, case, expected_run):
    case_id = case["id"]
    expected_metadata = {
        "expert_function_consistency_probe_version": PROBE_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "checkpoint": expected_run["checkpoint"],
        "weights_checkpoint": expected_run["weights_checkpoint"],
        "checkpoint_sha256": expected_run["checkpoint_sha256"],
        "weights_checkpoint_sha256": expected_run["weights_sha256"],
        "checkpoint_step": CHECKPOINT_STEP,
        "weights_checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_state": CHECKPOINT_STATE,
        "config": expected_run["config"],
        "model_name": MODEL_NAME,
        "latent": case["latent"],
        "latent_key": case["latent_key"],
        "latent_sha256": case["latent_sha256"],
        "label": case["label"],
        "block_index": BLOCK_INDEX,
        "sigmas": list(SIGMAS),
        "shifts_latent": [list(shift) for shift in SHIFTS],
        "patch_size": 2,
        "num_token_probes_per_cell": NUM_TOKEN_PROBES,
        "exact_batch_size": EXACT_BATCH_SIZE,
        "seed": case["seed"],
        "device": expected_run["device"],
        "num_threads": expected_run["num_threads"],
        "protocol_sha256": expected_run["protocol_sha256"],
        "batch_case": case,
    }
    for key, expected in expected_metadata.items():
        if result.get(key) != expected:
            raise ValueError(
                f"{case_id}: {key} differs: {result.get(key)!r} != {expected!r}"
            )

    cells = result.get("cells", [])
    expected_cells = [
        (float(sigma), list(shift))
        for sigma in SIGMAS
        for shift in SHIFTS
    ]
    actual_cells = [
        (float(cell.get("sigma")), cell.get("shift_latent"))
        for cell in cells
    ]
    if actual_cells != expected_cells:
        raise ValueError(f"{case_id}: sigma-shift cells differ from protocol")

    all_tokens = []
    observations = []
    controls = []
    cell_token_lists = []
    for cell in cells:
        tokens, cell_observations, cell_controls = _recompute_cell(cell, case_id)
        all_tokens.extend(tokens)
        cell_token_lists.append((cell, tokens))
        for observation in cell_observations:
            observations.append({"case_id": case_id, **observation})
        controls.append({"case_id": case_id, **cell_controls})

    _assert_nested_close(
        result.get("summary"),
        summarize_tokens(all_tokens),
        f"{case_id}.summary",
    )
    for sigma in SIGMAS:
        sigma_tokens = [
            token
            for cell, tokens in cell_token_lists
            if float(cell["sigma"]) == float(sigma)
            for token in tokens
        ]
        _assert_nested_close(
            result.get("per_sigma", {}).get(str(float(sigma))),
            summarize_tokens(sigma_tokens),
            f"{case_id}.per_sigma.{sigma}",
        )
    for dy, dx in SHIFTS:
        shift_tokens = [
            token
            for cell, tokens in cell_token_lists
            if cell["shift_latent"] == [dy, dx]
            for token in tokens
        ]
        _assert_nested_close(
            result.get("per_shift", {}).get(f"{dy}:{dx}"),
            summarize_tokens(shift_tokens),
            f"{case_id}.per_shift.{dy}:{dx}",
        )
    return observations, controls


def _bootstrap_ci(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Cluster bootstrap needs at least two finite image values")
    if resamples < 1000:
        raise ValueError("Cluster bootstrap requires at least 1000 resamples")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        values.size,
        size=(resamples, values.size),
        endpoint=False,
    )
    bootstrap_means = values[indices].mean(axis=1)
    return [
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    ]


def _check(observed, required, passed):
    return {
        "observed": observed,
        "required": required,
        "passed": bool(passed),
    }


def build_gate_summary(observations, controls, requirements=None):
    requirements = dict(requirements or GATE_REQUIREMENTS)
    _validate_requirements(requirements)
    expected_cases = int(requirements["expected_case_count"])
    expected_cells = int(requirements["expected_cells_per_case"])
    expected_tokens = int(requirements["expected_tokens_per_cell"])
    expected_per_image = expected_cells * expected_tokens
    case_ids = sorted({row["case_id"] for row in observations})
    if len(case_ids) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} cases, found {len(case_ids)}"
        )
    if len(observations) != expected_cases * expected_per_image:
        raise ValueError("Observation count does not match the locked protocol")
    if len(controls) != expected_cases * expected_cells:
        raise ValueError("Numerical-control count does not match the protocol")

    per_image = []
    primary_image_values = []
    delta_image_values = []
    minimum_valid_fraction = 1.0
    for case_id in case_ids:
        rows = [row for row in observations if row["case_id"] == case_id]
        if len(rows) != expected_per_image:
            raise ValueError(f"{case_id}: observation count differs")
        primary = np.asarray([
            row["primary_spearman"]
            for row in rows
            if row["primary_spearman"] is not None
        ], dtype=np.float64)
        delta = np.asarray([
            row["primary_minus_router_spearman"]
            for row in rows
            if row["primary_minus_router_spearman"] is not None
        ], dtype=np.float64)
        if primary.size == 0 or delta.size == 0:
            raise ValueError(f"{case_id}: no valid rank correlations")
        valid_fraction = min(primary.size, delta.size) / expected_per_image
        minimum_valid_fraction = min(minimum_valid_fraction, valid_fraction)
        primary_mean = float(primary.mean())
        delta_mean = float(delta.mean())
        primary_image_values.append(primary_mean)
        delta_image_values.append(delta_mean)
        per_image.append({
            "case_id": case_id,
            "valid_primary_tokens": int(primary.size),
            "valid_paired_tokens": int(delta.size),
            "valid_fraction": float(valid_fraction),
            "mean_primary_spearman": primary_mean,
            "mean_router_spearman": float(np.mean([
                row["router_spearman"]
                for row in rows
                if row["router_spearman"] is not None
            ])),
            "mean_primary_minus_router_spearman": delta_mean,
            "mean_native_router_weight": float(np.mean([
                row["native_router_weight"] for row in rows
            ])),
            "primary_selected_beats_native_rate": float(np.mean([
                row["primary_selected_beats_native"] for row in rows
            ])),
            "primary_oracle_top3_rate": float(np.mean([
                row["primary_oracle_top3"] for row in rows
            ])),
            "mean_exact_mse_change_range": float(np.mean([
                row["exact_mse_change_range"] for row in rows
            ])),
        })

    primary_ci = _bootstrap_ci(
        primary_image_values,
        int(requirements["bootstrap_resamples"]),
        int(requirements["bootstrap_seed"]),
    )
    delta_ci = _bootstrap_ci(
        delta_image_values,
        int(requirements["bootstrap_resamples"]),
        int(requirements["bootstrap_seed"]) + 1,
    )
    mean_primary = float(np.mean(primary_image_values))
    mean_delta = float(np.mean(delta_image_values))
    positive_images = int(np.sum(np.asarray(primary_image_values) > 0))

    per_sigma = {}
    for sigma in SIGMAS:
        sigma_image_values = []
        for case_id in case_ids:
            values = [
                row["primary_spearman"]
                for row in observations
                if row["case_id"] == case_id
                and float(row["sigma"]) == float(sigma)
                and row["primary_spearman"] is not None
            ]
            if not values:
                raise ValueError(f"{case_id}: sigma {sigma} has no valid primary values")
            sigma_image_values.append(float(np.mean(values)))
        per_sigma[str(float(sigma))] = {
            "image_cluster_mean_primary_spearman": float(
                np.mean(sigma_image_values)
            ),
            "positive_images": int(
                np.sum(np.asarray(sigma_image_values) > 0)
            ),
        }
    every_sigma_positive = all(
        row["image_cluster_mean_primary_spearman"] > 0
        for row in per_sigma.values()
    )

    minimum_router_weight = float(min(
        row["native_router_weight"] for row in observations
    ))
    mean_router_weight = float(np.mean([
        row["native_router_weight"] for row in observations
    ]))
    max_noop_mse = float(max(
        row["max_abs_noop_mse_change"] for row in controls
    ))
    max_noop_output = float(max(
        row["max_abs_noop_output_change"] for row in controls
    ))
    max_forced_unforced_mse = float(max(
        row["max_abs_forced_unforced_mse_change"] for row in controls
    ))
    max_forced_unforced_output = float(max(
        row["max_abs_forced_unforced_output_change"] for row in controls
    ))

    safety_checks = {
        "minimum_valid_fraction_per_image": _check(
            minimum_valid_fraction,
            f">={requirements['minimum_valid_fraction_per_image']}",
            minimum_valid_fraction
            >= requirements["minimum_valid_fraction_per_image"],
        ),
        "positive_native_router_weight": _check(
            minimum_router_weight,
            ">0" if requirements["require_positive_native_router_weight"]
            else "disabled",
            not requirements["require_positive_native_router_weight"]
            or minimum_router_weight > 0,
        ),
        "noop_abs_mse_change": _check(
            max_noop_mse,
            f"<={requirements['maximum_noop_abs_mse_change']}",
            max_noop_mse <= requirements["maximum_noop_abs_mse_change"],
        ),
        "noop_abs_output_change": _check(
            max_noop_output,
            f"<={requirements['maximum_noop_abs_output_change']}",
            max_noop_output <= requirements["maximum_noop_abs_output_change"],
        ),
        "forced_unforced_abs_mse_change": _check(
            max_forced_unforced_mse,
            f"<={requirements['maximum_forced_unforced_abs_mse_change']}",
            max_forced_unforced_mse
            <= requirements["maximum_forced_unforced_abs_mse_change"],
        ),
        "forced_unforced_abs_output_change": _check(
            max_forced_unforced_output,
            f"<={requirements['maximum_forced_unforced_abs_output_change']}",
            max_forced_unforced_output
            <= requirements["maximum_forced_unforced_abs_output_change"],
        ),
    }
    mechanism_checks = {
        "mean_primary_spearman": _check(
            mean_primary,
            f">={requirements['minimum_mean_primary_spearman']}",
            mean_primary >= requirements["minimum_mean_primary_spearman"],
        ),
        "primary_cluster_bootstrap_ci95_lower": _check(
            primary_ci[0],
            ">0" if requirements["require_primary_ci_lower_positive"]
            else "disabled",
            not requirements["require_primary_ci_lower_positive"]
            or primary_ci[0] > 0,
        ),
        "positive_images": _check(
            positive_images,
            f">={requirements['minimum_positive_images']}",
            positive_images >= requirements["minimum_positive_images"],
        ),
        "every_sigma_positive": _check(
            every_sigma_positive,
            "true" if requirements["require_every_sigma_positive"]
            else "disabled",
            not requirements["require_every_sigma_positive"]
            or every_sigma_positive,
        ),
        "mean_primary_minus_router_spearman": _check(
            mean_delta,
            f">={requirements['minimum_mean_primary_minus_router_spearman']}",
            mean_delta
            >= requirements["minimum_mean_primary_minus_router_spearman"],
        ),
        "delta_cluster_bootstrap_ci95_lower": _check(
            delta_ci[0],
            ">0" if requirements["require_delta_ci_lower_positive"]
            else "disabled",
            not requirements["require_delta_ci_lower_positive"]
            or delta_ci[0] > 0,
        ),
    }
    safety_passed = all(row["passed"] for row in safety_checks.values())
    mechanism_passed = all(row["passed"] for row in mechanism_checks.values())
    return {
        "passed": bool(safety_passed and mechanism_passed),
        "safety_passed": safety_passed,
        "mechanism_passed": mechanism_passed,
        "requirements": requirements,
        "primary_metric": PRIMARY_METRIC,
        "inference_unit": "image",
        "num_cases": len(case_ids),
        "num_token_observations": len(observations),
        "image_cluster_mean_primary_spearman": mean_primary,
        "image_cluster_primary_bootstrap_ci95": primary_ci,
        "positive_primary_images": positive_images,
        "image_cluster_mean_primary_minus_router_spearman": mean_delta,
        "image_cluster_delta_bootstrap_ci95": delta_ci,
        "minimum_native_router_weight": minimum_router_weight,
        "mean_native_router_weight": mean_router_weight,
        "per_sigma": per_sigma,
        "safety_checks": safety_checks,
        "mechanism_checks": mechanism_checks,
        "per_image": per_image,
    }


def aggregate_case_results(case_results, manifest_cases, expected_runs):
    if len(case_results) != len(manifest_cases):
        raise ValueError("Case result count does not match manifest")
    observations = []
    controls = []
    for result, case in zip(case_results, manifest_cases):
        expected_run = expected_runs[case["id"]]
        case_observations, case_controls = validate_case_result(
            result,
            case,
            expected_run,
        )
        observations.extend(case_observations)
        controls.extend(case_controls)
    return build_gate_summary(observations, controls)
