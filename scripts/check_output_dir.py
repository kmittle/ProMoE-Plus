#!/usr/bin/env python3
"""Deterministic output-directory collision guard for ProMoE experiments.

Output dirs are derived as ``outputs/{model_name}/{custom_cfg_name}`` (train.py:434),
where ``custom_cfg_name`` defaults to the config filename without extension
(train.py:723). Re-running an experiment after a model-code change WITHOUT changing
the config name silently re-uses the previous run's output dir — mixing old (buggy)
checkpoints with new code, or colliding with crashed-run artifacts. This helper makes
the check mechanical so the /new-experiment and /rerun-experiment skills (and humans)
never have to re-derive it by hand.

Usage
-----
  # Single config: is its output dir already claimed / on disk? suggest next _vN.
  python scripts/check_output_dir.py --config configs/004_ProMoE_<size>_<variant>.yaml

  # Whole-repo audit: every config + run script -> output dir, list collisions.
  python scripts/check_output_dir.py --all

  # Just print the next free _vN config name for a given config (no other output).
  python scripts/check_output_dir.py --suggest-version configs/004_ProMoE_<size>_<variant>.yaml

Exit codes: 0 = no collision, 1 = collision found, 2 = usage/IO error.
No third-party dependencies (crude top-level YAML scalar parse) so the guard always runs.
"""
import argparse
import glob
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _yaml_scalar(path, key):
    """Grab a TOP-LEVEL ``key: value`` scalar from a YAML file (no PyYAML needed).

    Anchored at zero indentation on purpose: train.py reads these from the top-level cfg, so a
    nested (indented) ``output_dir:``/``model_name:`` under a model-config block must NOT be picked up.
    """
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"^" + re.escape(key) + r"\s*:\s*(.+?)\s*(#.*)?$", line)
                if m:
                    return m.group(1).strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def output_dir_for(config_path):
    """Return (model_name, output_dir_relpath) for a config, mirroring train.py exactly.

    train.py forces ``custom_cfg_name`` to the config basename (train.py:723), overwriting any
    YAML ``custom_cfg_name`` — so we use the basename unconditionally and never honor an override
    for it. A top-level ``output_dir`` IS honored (the YAML deep-merges into ``cfg.output_dir``
    before the join at train.py:434); it defaults to "outputs".
    """
    base = os.path.splitext(os.path.basename(config_path))[0]
    model_name = _yaml_scalar(config_path, "model_name")
    ccn = base  # train.py:723 always uses the basename, ignoring any YAML custom_cfg_name
    root = _yaml_scalar(config_path, "output_dir") or "outputs"
    # os.path.join (not an f-string) mirrors train.py's osp.join and avoids a double slash when a
    # YAML output_dir carries a trailing "/" (cfg.output_dir defaults to "outputs/").
    return model_name, os.path.join(root, str(model_name), ccn)


def _count_checkpoints(out_rel):
    ckpt = os.path.join(REPO_ROOT, out_rel, "checkpoints")
    if not os.path.isdir(ckpt):
        return None
    return len(glob.glob(os.path.join(ckpt, "*.pth")))


def _all_configs():
    return sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "*.yaml")))


def _run_scripts():
    """Semantic run scripts only (skip _run_times wrappers and template.sh)."""
    out = []
    for p in glob.glob(os.path.join(REPO_ROOT, "scripts", "**", "*.sh"), recursive=True):
        if "/_run_times/" in p or os.path.basename(p) == "template.sh":
            continue
        out.append(p)
    return sorted(out)


def _script_config(script_path):
    try:
        txt = open(script_path, errors="ignore").read()
    except OSError:
        return None
    m = re.search(r'^CONFIG=["\']?(configs/[^"\'\s]+)', txt, re.M)
    return m.group(1) if m else None


def _is_split_pair(script_basenames):
    """Legit train/infer split: the WHOLE group is exactly one train script plus one or more
    infer/sample/eval read scripts, and contains NO standalone ``*_train_sample_eval.sh``. An
    extra all-in-one script or a second train script sharing the dir is a real collision, not a
    pair, so it must fall through to COLLISION rather than be masked as OK."""
    if any("_train_sample_eval" in b for b in script_basenames):
        return False
    train = [b for b in script_basenames if "_train" in b]
    # a read script must NOT also be a train script — keep the two roles disjoint so a name like
    # "foo_train_eval.sh" counts once (as train), never as both.
    read = [b for b in script_basenames
            if ("infer" in b or "sample" in b or "eval" in b) and "_train" not in b]
    accounted = set(train) | set(read)
    # exactly one train + >=1 read, and NOTHING else in the group (an unrelated extra script
    # sharing the dir is a real collision, not a split pair)
    return len(train) == 1 and len(read) >= 1 and accounted == set(script_basenames)


def suggest_version(config_path):
    """Next free ``_vN`` config filename for config_path (v2, v3, ...)."""
    d = os.path.dirname(config_path)
    base = os.path.splitext(os.path.basename(config_path))[0]
    ext = os.path.splitext(config_path)[1] or ".yaml"
    # strip an existing _vN suffix so v2 -> v3, not v2_v2
    stem = re.sub(r"_v\d+$", "", base)
    n = 2
    while True:
        cand = os.path.join(d, f"{stem}_v{n}{ext}")
        if not os.path.exists(cand):
            return cand
        n += 1


def check_one(config_path):
    rel_cfg = os.path.relpath(config_path, REPO_ROOT) if os.path.isabs(config_path) else config_path
    abs_cfg = config_path if os.path.isabs(config_path) else os.path.join(REPO_ROOT, config_path)
    if not os.path.isfile(abs_cfg):
        print(f"ERROR: config not found: {rel_cfg}", file=sys.stderr)
        return 2
    model_name, out_rel = output_dir_for(abs_cfg)
    print(f"config        : {rel_cfg}")
    print(f"model_name    : {model_name}")
    print(f"output dir    : {out_rel}/")

    problems = []

    # (1) another config maps to the same output dir (only possible via overrides)
    for other in _all_configs():
        if os.path.abspath(other) == os.path.abspath(abs_cfg):
            continue
        if output_dir_for(other)[1] == out_rel:
            problems.append(f"another config also maps here: {os.path.relpath(other, REPO_ROOT)}")

    # (2) output dir already exists on local disk (best-effort; real runs may live on a remote server)
    nck = _count_checkpoints(out_rel)
    if os.path.isdir(os.path.join(REPO_ROOT, out_rel)):
        problems.append(f"output dir already exists on local disk (checkpoints={nck if nck is not None else 0})")

    if problems:
        print("RESULT        : COLLISION")
        for p in problems:
            print(f"  - {p}")
        print(f"suggested name: {os.path.relpath(suggest_version(abs_cfg), REPO_ROOT)}")
        print("NOTE          : local disk only reflects THIS machine. If this experiment was already")
        print("                launched on the training server (or its model code changed since the last")
        print("                run), bump to the suggested _vN name so the new run gets a fresh bucket.")
        return 1
    print("RESULT        : OK (no static or local-disk collision)")
    print("REMINDER      : local disk can't see the training server. If you changed model code and are")
    print("                RE-RUNNING an already-launched experiment, still bump to a _vN name.")
    return 0


def audit_all():
    rc = 0
    cfg_out = {}
    for cfg in _all_configs():
        cfg_out[cfg] = output_dir_for(cfg)[1]

    # configs sharing a dir (override-induced)
    by_out = {}
    for cfg, out in cfg_out.items():
        by_out.setdefault(out, []).append(cfg)
    print("########## configs sharing the SAME output dir ##########")
    hit = False
    for out, cfgs in sorted(by_out.items()):
        if len(cfgs) > 1:
            hit = True
            rc = 1
            print(f"  COLLISION {out}")
            for c in cfgs:
                print(f"      <- {os.path.relpath(c, REPO_ROOT)}")
    if not hit:
        print("  none")

    # run scripts sharing a dir (split-pairs are OK)
    print("\n########## run scripts sharing the SAME output dir ##########")
    script_out = {}
    for s in _run_scripts():
        crel = _script_config(s)
        if not crel:
            continue
        cabs = os.path.join(REPO_ROOT, crel)
        out = cfg_out.get(cabs) or output_dir_for(cabs)[1]
        script_out.setdefault(out, []).append(s)
    hit = False
    for out, ss in sorted(script_out.items()):
        if len(ss) > 1:
            basenames = [os.path.basename(x) for x in ss]
            tag = "split-pair (OK)" if _is_split_pair(basenames) else "COLLISION"
            if tag != "split-pair (OK)":
                hit = True
                rc = 1
            print(f"  {tag} {out}")
            for s in ss:
                print(f"      <- {os.path.relpath(s, REPO_ROOT)}")
    if not hit:
        print("  none beyond intentional train/infer split-pairs")

    # configs whose dir exists on local disk — INFORMATIONAL only. These are usually legitimate
    # completed/in-progress runs, so (unlike the single-config guard check_one) they do NOT change
    # the exit code; --all returns non-zero only for the genuine same-dir collisions reported above.
    print("\n########## output dirs already on local disk (informational) ##########")
    hit = False
    for cfg, out in sorted(cfg_out.items()):
        if os.path.isdir(os.path.join(REPO_ROOT, out)):
            hit = True
            nck = _count_checkpoints(out)
            print(f"  EXISTS {out}  (checkpoints={nck if nck is not None else 0})"
                  f"  <- {os.path.relpath(cfg, REPO_ROOT)}")
    if not hit:
        print("  none on this machine")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="check a single config's output dir for collisions")
    g.add_argument("--all", action="store_true", help="audit every config + run script")
    g.add_argument("--suggest-version", metavar="CONFIG",
                   help="print only the next free _vN config name and exit")
    args = ap.parse_args()

    if args.suggest_version:
        abs_cfg = (args.suggest_version if os.path.isabs(args.suggest_version)
                   else os.path.join(REPO_ROOT, args.suggest_version))
        if not os.path.isfile(abs_cfg):
            print(f"ERROR: config not found: {args.suggest_version}", file=sys.stderr)
            return 2
        print(os.path.relpath(suggest_version(abs_cfg), REPO_ROOT))
        return 0
    if args.all:
        return audit_all()
    return check_one(args.config)


if __name__ == "__main__":
    sys.exit(main())
