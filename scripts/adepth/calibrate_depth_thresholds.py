"""Calibrate the adaptive-depth inference cutoffs of one checkpoint.

Training classifies cond tokens with a quota over a per-rank batch that mixes timesteps, so
the quota's boundary scores act as one cutoff pair for every timestep. A sampling forward
shares a single timestep, and the same quota forces exactly q skipped and q deepened tokens
at every step (header of models/models_ProMoE_TC_adepth.py). This script

1. runs training-style batches (per-rank batch size, logit-normal t, label dropout, latent
   flips) through the EMA weights with the quota, and takes the q and 1-q quantiles of each
   MoE block's pooled cond-token scores as that block's cutoffs;
2. runs held-out batches under four policies -- the batch quota (what training did), a
   quota per image (what a sampling batch at one timestep does), the cutoffs, and adaptive
   depth switched off -- and reports per-token agreement with the batch quota, the
   skip/deepen shares and the flow-matching loss by noise level;
3. samples a few images with the cutoffs and records the skip/deepen shares along the
   sampler's own trajectory, which sets the inference compute.

The JSON output carries `depth_infer_thresholds`, ready for MoE_config. Everything runs in
fp32, the precision sample.py uses (val_param_dtype).

    python scripts/adepth/calibrate_depth_thresholds.py \\
        --config configs/004_ProMoE_B_adepth_q0p3.yaml \\
        --checkpoint outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p3/checkpoints/ckpt_step_300000.pth \\
        --output q0p3.calibration.json
"""
import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402

from config import cfg  # noqa: E402
from sample import _load_project_checkpoint, get_sampling_sigmas, model_dict, retrieve_timesteps  # noqa: E402
from train import compute_density_for_timestep_sampling, get_sigmas_timesteps  # noqa: E402
from utils import deep_update  # noqa: E402

SIGMA_BINS = np.linspace(0.0, 1.0, 11)
POLICIES = ("batch_quota", "image_quota", "cutoffs", "depth_off")


class ScoreRecorder:
    """Keeps each MoE block's cond-token scores from the latest forward."""

    def __init__(self, mlps):
        self.scores = [None] * len(mlps)
        for index, mlp in enumerate(mlps):
            mlp.depth_gate.register_forward_hook(self._hook(index))

    def _hook(self, index):
        def hook(module, inputs, output):
            if output.shape[0] > 1:  # the DDP keep-alive touch scores a single row
                self.scores[index] = output.detach().squeeze(-1).float()
        return hook

    def reset(self):
        self.scores = [None] * len(self.scores)


class TrainingBatches:
    """Per-rank batches drawn the way train.py draws them, from one seed."""

    def __init__(self, latent_dir, batch_size, dropout_prob, num_classes, seed, device):
        self.latent_dir = latent_dir
        self.classes = sorted(entry.name for entry in os.scandir(latent_dir) if entry.is_dir())
        if len(self.classes) != num_classes:
            raise ValueError(f"{latent_dir} has {len(self.classes)} class dirs, expected {num_classes}")
        self.files = {}
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.device = device
        self.rng = random.Random(seed)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def _latent(self, label):
        if label not in self.files:
            directory = os.path.join(self.latent_dir, self.classes[label])
            self.files[label] = sorted(name for name in os.listdir(directory) if name.endswith(".latent.npz"))
        name = self.rng.choice(self.files[label])
        key = "latent_flip" if self.rng.random() < 0.5 else "latent"
        with np.load(os.path.join(self.latent_dir, self.classes[label], name)) as data:
            return np.array(data[key], copy=True)

    def next(self):
        labels = [self.rng.randrange(self.num_classes) for _ in range(self.batch_size)]
        moments = torch.from_numpy(np.stack([self._latent(label) for label in labels])).to(self.device)
        z = DiagonalGaussianDistribution(moments).sample(generator=self.generator).mul_(0.18215)
        labels = torch.tensor(labels, device=self.device)
        dropped = torch.rand(self.batch_size, generator=self.generator, device=self.device) < self.dropout_prob
        labels = torch.where(dropped, torch.full_like(labels, self.num_classes), labels)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=cfg.weighting_scheme, batch_size=self.batch_size,
            logit_mean=cfg.logit_mean, logit_std=cfg.logit_std, sigmoid_scale=cfg.sigmoid_scale,
            mode_scale=cfg.mode_scale, generator=self.generator, device=self.device,
        )
        t, sigma = get_sigmas_timesteps(u, cfg.shift, cfg.num_train_timesteps, n_dim=4)
        sigma = sigma.flatten()
        noise = torch.randn(z.shape, generator=self.generator, device=self.device)
        x = (1.0 - sigma).view(-1, 1, 1, 1) * z + sigma.view(-1, 1, 1, 1) * noise
        return x, t, sigma, labels, noise - z


def quota_classes(scores, q):
    """The batch quota of SparseMoeBlock.forward: 0 skip, 1 normal, 2 deepen."""
    n = scores.numel()
    k = min(int(round(q * n)), n // 2)
    classes = torch.ones(n, dtype=torch.long, device=scores.device)
    if k > 0:
        order = torch.argsort(scores)
        classes[order[n - k:]] = 2
        classes[order[:k]] = 0
    return classes


def cutoff_classes(scores, skip_threshold, deepen_threshold):
    classes = torch.ones(scores.numel(), dtype=torch.long, device=scores.device)
    classes[scores > deepen_threshold] = 2
    classes[scores < skip_threshold] = 0
    return classes


def prediction(output):
    # learn_sigma doubles the channels; train.py and sample.py keep the first half.
    return output.chunk(2, dim=1)[0] if output.shape[1] == 8 else output


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def shares_by_sigma(rows, num_blocks):
    """rows: (sigma, per-block skip share, per-block deepen share) per cond image."""
    sigma = np.array([row[0] for row in rows])
    skip = np.array([row[1] for row in rows])
    deepen = np.array([row[2] for row in rows])
    bins = []
    for lo, hi in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:]):
        inside = (sigma >= lo) & (sigma < hi)
        if not inside.any():
            bins.append(dict(sigma=[float(lo), float(hi)], images=0))
            continue
        bins.append(dict(sigma=[float(lo), float(hi)], images=int(inside.sum()),
                         skip=[round(float(v), 4) for v in skip[inside].mean(0)],
                         deepen=[round(float(v), 4) for v in deepen[inside].mean(0)]))
    return dict(overall_skip=[round(float(v), 4) for v in skip.mean(0)],
                overall_deepen=[round(float(v), 4) for v in deepen.mean(0)], by_sigma=bins)


def per_image_shares(classes, num_images, seq_len):
    per_image = classes.view(num_images, seq_len)
    return (per_image == 0).float().mean(1).tolist(), (per_image == 2).float().mean(1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="JSON file to write")
    parser.add_argument("--latent-dir", default=None, help="default: the config's latent_data_path")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: total_train_batch_size / len(gpu_ids), the per-rank batch")
    parser.add_argument("--calib-batches", type=int, default=64)
    parser.add_argument("--check-batches", type=int, default=16)
    parser.add_argument("--sample-images", type=int, default=64, help="0 skips the sampling probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        deep_update(cfg, yaml.safe_load(f))
    device = torch.device(args.device)
    batch_size = args.batch_size or cfg.total_train_batch_size // len(cfg.gpu_ids)
    latent_dir = args.latent_dir or cfg.latent_data_path

    model_class, config_name = model_dict[cfg.model_name]
    model = model_class(**getattr(cfg, config_name))
    checkpoint = _load_project_checkpoint(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"EMA weights do not match the model: missing={missing} unexpected={unexpected}")
    del checkpoint
    model = model.to(device).eval()

    block_ids = [i for i, block in enumerate(model.blocks) if getattr(block, "use_moe", False)]
    mlps = [model.blocks[i].mlp for i in block_ids]
    if not mlps or any(getattr(mlp, "alloc_mode", None) != "fixed_q" for mlp in mlps):
        raise ValueError(f"{cfg.model_name} has no fixed_q adaptive-depth blocks")
    for mlp in mlps:  # calibration starts from the quota whatever the config carries
        mlp.depth_skip_threshold = mlp.depth_deepen_threshold = None
    qs = [mlp.current_depth_q() for mlp in mlps]
    num_classes = model.y_embedder.num_classes
    dropout_prob = model.y_embedder.dropout_prob
    recorder = ScoreRecorder(mlps)
    print(f"{cfg.model_name} blocks={block_ids} q={qs} batch={batch_size} latents={latent_dir}", flush=True)

    def forward(x, t, labels):
        recorder.reset()
        with torch.no_grad():
            return model(x, t, context=labels)

    def set_cutoffs(pairs):
        for mlp, pair in zip(mlps, pairs):
            if pair is None:
                mlp.depth_skip_threshold = mlp.depth_deepen_threshold = None
            else:
                mlp.set_depth_thresholds(*pair)

    # 1. Cutoffs from pooled training-style scores.
    batches = TrainingBatches(latent_dir, batch_size, dropout_prob, num_classes, args.seed, device)
    pooled = [[] for _ in mlps]
    batch_cutoffs = [[] for _ in mlps]
    for b in range(args.calib_batches):
        x, t, _, labels, _ = batches.next()
        forward(x, t, labels)
        for i, scores in enumerate(recorder.scores):
            if scores is None:  # an all-uncond batch
                continue
            pooled[i].append(scores.cpu())
            n = scores.numel()
            k = min(int(round(qs[i] * n)), n // 2)
            if k > 0:
                ordered = torch.sort(scores).values
                batch_cutoffs[i].append((ordered[k - 1].item(), ordered[n - k].item()))
        if (b + 1) % 16 == 0:
            print(f"calibration batch {b + 1}/{args.calib_batches}", flush=True)
    thresholds, blocks = [], []
    for i, block_id in enumerate(block_ids):
        scores = torch.cat(pooled[i]).double().numpy()
        skip, deepen = float(np.quantile(scores, qs[i])), float(np.quantile(scores, 1.0 - qs[i]))
        thresholds.append([skip, deepen])
        per_batch = np.array(batch_cutoffs[i])
        blocks.append(dict(
            block=block_id, q=qs[i], skip_threshold=skip, deepen_threshold=deepen, tokens=int(scores.size),
            score_quantiles={str(p): float(np.quantile(scores, p)) for p in (0.01, 0.1, 0.5, 0.9, 0.99)},
            batch_boundary_scores=dict(
                skip_mean=float(per_batch[:, 0].mean()), skip_std=float(per_batch[:, 0].std()),
                deepen_mean=float(per_batch[:, 1].mean()), deepen_std=float(per_batch[:, 1].std()),
            ),
        ))
    print("thresholds", json.dumps(thresholds), flush=True)

    # 2. Held-out batches under the four policies.
    check = TrainingBatches(latent_dir, batch_size, dropout_prob, num_classes, args.seed + 1, device)
    agreement = {policy: [[] for _ in mlps] for policy in ("image_quota", "cutoffs")}
    shares = {policy: [] for policy in POLICIES[:3]}
    losses = {policy: [] for policy in POLICIES}
    loss_sigma = []
    for b in range(args.check_batches):
        x, t, sigma, labels, target = check.next()
        cond = torch.where(labels != num_classes)[0]
        if cond.numel() == 0:
            continue
        cond_sigma = sigma[cond].tolist()
        loss_sigma.extend(cond_sigma)
        classes, outputs = {}, {}

        set_cutoffs([None] * len(mlps))
        outputs["batch_quota"] = prediction(forward(x, t, labels))
        classes["batch_quota"] = [quota_classes(s, q) for s, q in zip(recorder.scores, qs)]

        outputs["image_quota"] = torch.empty_like(outputs["batch_quota"])
        image_classes = [[] for _ in mlps]
        for j in range(len(labels)):
            outputs["image_quota"][j:j + 1] = prediction(forward(x[j:j + 1], t[j:j + 1], labels[j:j + 1]))
            if labels[j] != num_classes:
                for i, (s, q) in enumerate(zip(recorder.scores, qs)):
                    image_classes[i].append(quota_classes(s, q))
        classes["image_quota"] = [torch.cat(c) for c in image_classes]

        set_cutoffs(thresholds)
        outputs["cutoffs"] = prediction(forward(x, t, labels))
        classes["cutoffs"] = [cutoff_classes(s, *pair) for s, pair in zip(recorder.scores, thresholds)]

        for mlp in mlps:
            mlp.alloc_mode = "none"
        outputs["depth_off"] = prediction(forward(x, t, labels))
        for mlp in mlps:
            mlp.alloc_mode = "fixed_q"
        set_cutoffs([None] * len(mlps))

        seq_len = classes["batch_quota"][0].numel() // cond.numel()
        for policy in shares:
            per_block = [per_image_shares(c, cond.numel(), seq_len) for c in classes[policy]]
            for j in range(cond.numel()):
                shares[policy].append((cond_sigma[j], [p[0][j] for p in per_block], [p[1][j] for p in per_block]))
        for policy in agreement:
            for i in range(len(mlps)):
                agreement[policy][i].append((classes[policy][i] == classes["batch_quota"][i]).float().mean().item())
        for policy in POLICIES:
            per_image = ((outputs[policy][cond] - target[cond]) ** 2).flatten(1).mean(1)
            losses[policy].extend(per_image.tolist())
        print(f"check batch {b + 1}/{args.check_batches}", flush=True)

    loss_sigma = np.array(loss_sigma)
    loss_table = {}
    for policy in POLICIES:
        values = np.array(losses[policy])
        rows = []
        for lo, hi in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:]):
            inside = (loss_sigma >= lo) & (loss_sigma < hi)
            rows.append(round(float(values[inside].mean()), 6) if inside.any() else None)
        loss_table[policy] = dict(mean=round(float(values.mean()), 6), by_sigma=rows)
    loss_table["images_by_sigma"] = [int(((loss_sigma >= lo) & (loss_sigma < hi)).sum())
                                     for lo, hi in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:])]
    check_report = dict(
        batches=args.check_batches, cond_images=len(loss_sigma),
        agreement_with_batch_quota={policy: [round(float(np.mean(v)), 4) for v in values]
                                    for policy, values in agreement.items()},
        shares={policy: shares_by_sigma(rows, len(mlps)) for policy, rows in shares.items()},
        loss=loss_table,
    )

    # 3. The sampler's own trajectory under the cutoffs.
    sampling = {}
    if args.sample_images > 0:
        set_cutoffs(thresholds)
        for guide_scale in (1.0, 1.5):
            generator = torch.Generator(device=device).manual_seed(args.seed + 2)
            n = args.sample_images
            latents = torch.randn(n, 4, 1, cfg.image_size // 8, cfg.image_size // 8, generator=generator, device=device)
            y = torch.randint(0, num_classes, (n,), generator=generator, device=device)
            y_null = torch.full_like(y, num_classes)
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift)
            timesteps, _ = retrieve_timesteps(scheduler, device=device,
                                              sigmas=get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift))
            steps = []
            for step_t in timesteps:
                timestep = step_t.expand(n)
                pred = prediction(forward(latents, timestep, y))
                skip = [float((s < pair[0]).float().mean()) for s, pair in zip(recorder.scores, thresholds)]
                deepen = [float((s > pair[1]).float().mean()) for s, pair in zip(recorder.scores, thresholds)]
                steps.append((float(step_t) / cfg.num_train_timesteps, skip, deepen))
                if guide_scale > 1.0:
                    pred_null = prediction(forward(latents, timestep, y_null))
                    pred = pred_null + guide_scale * (pred - pred_null)
                latents = scheduler.step(pred.unsqueeze(2), step_t, latents, return_dict=False)[0]
            table = shares_by_sigma(steps, len(mlps))
            routed = 1.0 - np.mean([row[1] for row in steps]) + np.mean([row[2] for row in steps])
            sampling[f"cfg{guide_scale}"] = dict(
                images=n, steps=len(steps),
                cond_routed_passes_vs_base=round(float(routed), 4),
                routed_passes_vs_base=round(float(routed if guide_scale == 1.0 else (routed + 1.0) / 2.0), 4),
                **table,
            )
            print(f"sampling probe cfg{guide_scale}: cond routed passes {routed:.4f} of base", flush=True)
        set_cutoffs([None] * len(mlps))

    report = dict(
        created_at=datetime.datetime.now().astimezone().isoformat(), git_head=git_head(),
        config=os.path.abspath(args.config), checkpoint=os.path.abspath(args.checkpoint),
        model_name=cfg.model_name, precision="fp32",
        settings=dict(batch_size=batch_size, calib_batches=args.calib_batches, check_batches=args.check_batches,
                      sample_images=args.sample_images, seed=args.seed, latent_dir=latent_dir,
                      label_dropout=dropout_prob, sample_steps=cfg.sample_steps, sample_shift=cfg.sample_shift),
        depth_infer_thresholds=thresholds, blocks=blocks, check=check_report, sampling=sampling,
    )
    temporary = args.output + ".pending"
    with open(temporary, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    os.replace(temporary, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
