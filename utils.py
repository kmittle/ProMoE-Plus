import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights='DEFAULT')

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break
        

        return outp[0].squeeze(3).squeeze(2)


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    # Skips default weight inititialization if supported by torchvision
    # version. See https://github.com/mseitzer/pytorch-fid/issues/28.
    if version >= (0, 6):
        kwargs['init_weights'] = False

    # Backwards compatibility: `weights` argument was handled by `pretrained`
    # argument prior to version 0.13.
    if version < (0, 13) and 'weights' in kwargs:
        if kwargs['weights'] == 'DEFAULT':
            kwargs['pretrained'] = True
        elif kwargs['weights'] is None:
            kwargs['pretrained'] = False
        else:
            raise ValueError(
                'weights=={} not supported in torchvision {}'.format(
                    kwargs['weights'], torchvision.__version__
                )
            )
        del kwargs['weights']

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008,
                              aux_logits=False,
                              weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def load_vae(hf_repo_id, local_root="pretrained_ckpt/vae", vae_path=None):
    """Load VAE from local cache if available, otherwise download from HuggingFace and save locally.

    Args:
        hf_repo_id: HuggingFace repo ID, used when no vae_path is given.
        local_root: Root directory for auto-downloaded cache.
        vae_path: If specified, load VAE directly from this local path
                  (skip download logic entirely).
    """
    from diffusers.models import AutoencoderKL

    if vae_path is not None:
        logging.info(f"Loading VAE from user-specified path: {vae_path}")
        vae = AutoencoderKL.from_pretrained(vae_path)
        return vae

    local_path = os.path.join(local_root, hf_repo_id.replace("/", "--"))
    if os.path.isdir(local_path) and os.listdir(local_path):
        logging.info(f"Loading VAE from local path: {local_path}")
        vae = AutoencoderKL.from_pretrained(local_path)
    else:
        logging.info(f"Local VAE not found at {local_path}, downloading from HuggingFace: {hf_repo_id}")
        vae = AutoencoderKL.from_pretrained(hf_repo_id)
        os.makedirs(local_path, exist_ok=True)
        vae.save_pretrained(local_path)
        logging.info(f"VAE saved to {local_path}")
    return vae


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def deep_update(target, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in target:
            deep_update(target[k], v)
        else:
            target[k] = v

def str_to_int_list(arg_string):
    if not arg_string:
        return []
    try:
        return [int(s.strip()) for s in arg_string.split(',')]
    except ValueError:
        msg = f"'{arg_string}' contains values ​​that cannot be converted to integers."
        raise argparse.ArgumentTypeError(msg)

def str_to_float_list(arg_string):
    if not arg_string:
        return []
    try:
        return [float(s.strip()) for s in arg_string.split(',')]
    except ValueError:
        msg = f"'{arg_string}' contains values ​​that cannot be converted to floating-point numbers."
        raise argparse.ArgumentTypeError(msg)


# =============================================================================
# TrainingMonitor — non-invasive statistics for diagnosing crashed models.
#
# Observed crashes (plans 02/03/04/08) shared the same symptom — MSE stays
# healthy for thousands of steps, then suddenly spikes and never recovers.
# This monitor is designed to surface the *precursor* signals in the hope that
# a re-run on a fixed model exposes the leading indicator (attention-row
# collapse, exploding projector features, a single group's grad norm running
# away, etc.).
#
# Design goals:
#   * Zero model code changes — uses `forward_hook`s keyed on class name and
#     `named_parameters` iteration for gradient stats.
#   * Minimal training-script changes — construct once after the model is
#     built, call `on_step(step, losses=...)` once per optimizer step (after
#     `backward()`, before `zero_grad()`), optional `close()` at the end.
#   * Cheap — every stat is reduced to a python float on the captured
#     tensor's device, never keeps whole feature maps alive.
#   * Safe — the whole pipeline is wrapped in try/except so a monitoring bug
#     cannot take the training run down with it.
# =============================================================================


class TrainingMonitor:
    """Collect per-step statistics on cross-alignment / REPA modules.

    What it captures (rank-0 only by default):
        * Attention map W from cross-alignment modules (ExpertLocalAttention,
          BlockAlignAttention, GlobalPreAttention) — min/max/mean, row-sum
          min/max, fraction of near-empty rows, NaN/Inf counts.
        * Projector outputs (REPA `projectors` / MoS `mos_projectors`) —
          token-norm min/max/mean, max-abs, NaN/Inf counts.
        * Coefficient-predictor outputs (`CoeffPredictor`,
          `AlignCoefficientPredictor`) — per-sample sigmoid coefficient
          min/max/mean.
        * BlockRouter outputs — routing-weight max/mean and per-token entropy
          (detects collapse onto a single teacher block).
        * `cluster_centers` parameter norm min/max/mean (one per step).
        * Cross-alignment internals (model-cached, no extra hooks):
            - cos_sim_absmax: pre-clamp cosine similarity max absolute value
              (detects bf16 overflow beyond [-1,1]).
            - cos_sim_clamp_count / cos_sim_numel: fraction of values clamped.
            - proto_sim_neg_frac / proto_sim_min: prototype similarity before
              clamp(min=0) — detects routing pathology in proto variants.
            - W_row_sum_min / W_nonzero_frac: cross-weight matrix health
              (detects near-zero denominator in proto normalization).
        * Gradient statistics grouped by parameter-name substring:
            attn_modules, projectors, block_router, cluster_centers,
            coeff_predictor, moe_experts, backbone.
          Each group reports total L2 norm + NaN/Inf param count. `global`
          reports the same across all parameters.
        * Loss values passed via `on_step(losses=...)` — tracked with a simple
          exponential moving average + a relative-jump alert.

    Usage:
        >>> from utils import TrainingMonitor
        >>> monitor = TrainingMonitor(
        ...     model, logger=logging.getLogger(),
        ...     log_every=cfg.log_interval,
        ...     enabled=(cfg.rank == 0),
        ... )
        >>> # inside the training loop, AFTER scaler.unscale_(optimizer)
        >>> # but BEFORE clip_grad_norm_() — this is the only window where
        >>> # gradients are unscaled AND unclipped, so per-group norms
        >>> # reflect the true gradient magnitude:
        >>> monitor.on_step(step, losses=logged_loss_dict)
        >>> grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        >>> # ... (at shutdown)
        >>> monitor.close()

    The monitor logs a compact one-line summary every `log_every` steps and
    always emits a warning line the moment an anomaly threshold is tripped
    (regardless of `log_every`).

    Note on gradient accumulation (``grad_mix > 1``): forward-hook stats
    and ``_cross_align_stats`` are overwritten each micro-batch, so only
    the last micro-batch's activation statistics are captured. Loss values
    passed via ``on_step(losses=...)`` should already be averaged across
    micro-batches by the caller. This is a known design trade-off; for the
    typical ``grad_mix=1`` case there is no discrepancy.
    """

    # -------- class-name-based hook targets (leaf modules) --------
    # NOTE: we deliberately DON'T hook RouterTransformerBlock — it's a shared
    # 2-layer transformer used *inside* attention/router classes, and hooking
    # it would duplicate capture. We also don't hook FFN/MLP helpers.
    _ATTN_CLASS_NAMES = (
        'ExpertLocalAttention',
        'BlockAlignAttention',
        'GlobalPreAttention',
    )
    _COEFF_CLASS_NAMES = (
        'CoeffPredictor',               # models_ProMoE_TC_repa_MoS_naive_choice_fused.py
        'AlignCoefficientPredictor',    # models_ProMoE_TC_repa_multi_align.py
    )
    # Router classes differ in output shape and whether output is softmaxed,
    # so we dispatch per-class in the hook body (see _make_router_hook).
    #   BlockRouter    -> (N,T,m,n), softmax over m (dim=-2)  [MoS naive/choice]
    #   PerBlockRouter -> (N,T,m),   softmax over dim=-1       [choice_per_block]
    #   AdaLNRouter    -> (N,T,K),   raw logits                [MoS.py]
    _ROUTER_CLASS_NAMES = (
        'BlockRouter',
        'PerBlockRouter',
        'AdaLNRouter',
    )

    # -------- parameter-name grouping for grad stats --------
    # Ordered: first match wins.  Keep `backbone` as the implicit catch-all.
    # `shared_expert` must come before `moe_experts` so `blocks.*.mlp.shared_expert.*`
    # doesn't fall through to the `.experts.` bucket (note: the singular
    # "shared_expert" name does *not* contain `.experts.`, but be explicit anyway).
    _GRAD_GROUPS = (
        ('attn_modules', ('block_align_attn', 'expert_local_attn',
                           'global_pre_attn')),
        ('projectors', ('mos_projectors', 'align_projectors', 'projectors')),
        ('block_router', ('block_router', 'per_block_routers', 'routers.')),
        ('coeff_predictor', ('coeff_predictor', 'align_coeff_predictor')),
        ('capacity_predictor', ('capacity_predictor',)),  # DiffMoE
        ('cluster_centers', ('cluster_centers',)),
        ('shared_expert', ('shared_expert',)),
        ('moe_experts', ('.experts.',)),
        # backbone: t_embedder, y_embedder, x_embedder, final_layer, DiT attention/ffn
    )

    # -------- alert thresholds (override via kwargs) --------
    _DEFAULT_THRESHOLDS = {
        'grad_norm_warn': 50.0,        # per-group L2 norm
        'feature_norm_warn': 5.0e3,    # projector / attention output abs-max
        'attn_row_sum_min': 0.5,       # active softmax rows that sum below this are suspect
        'attn_row_sum_max': 1.5,       # or above this (shouldn't happen post-softmax)
        'loss_jump_ratio': 3.0,        # total_loss > ratio * EMA => warn
        'cos_sim_absmax_warn': 1.01,   # pre-clamp cos_sim outside [-1,1] (bf16 overflow)
        'proto_neg_frac_warn': 0.5,    # >50% negative proto_sim before clamp
        'W_row_sum_min_warn': 0.01,    # near-zero W row sum (normalization risk)
    }
    # Rows whose sum <= this are treated as "inactive" (masked out by design).
    # Kept well below any legitimate softmax normalisation to avoid mistaking
    # a merely-collapsed row for an inactive one.
    _ACTIVE_ROW_EPS = 1.0e-4

    def __init__(self, model, logger=None, log_every=500, enabled=True,
                 capture_attn=True, capture_projector=True,
                 capture_coeff=True, capture_router=True,
                 capture_grads=True, track_losses=True,
                 thresholds=None, writer=None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.log_every = max(int(log_every), 1)
        self.enabled = bool(enabled)
        self._writer = writer
        self.capture_attn = capture_attn
        self.capture_projector = capture_projector
        self.capture_coeff = capture_coeff
        self.capture_router = capture_router
        self.capture_grads = capture_grads
        self.track_losses = track_losses

        self.thresholds = dict(self._DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

        self._hooks = []
        self._attn_stats = {}         # name -> dict
        self._projector_stats = {}    # name -> dict
        self._coeff_stats = {}        # name -> dict
        self._router_stats = {}       # name -> dict
        self._hooked_names = {
            'attn': [],
            'projector': [],
            'coeff': [],
            'router': [],
        }
        self._loss_ema = {}           # loss_name -> running mean
        self._ema_alpha = 0.05
        self._last_step_logged = -1

        self._model = model
        self._real = self._unwrap(model)

        if self.enabled:
            try:
                self._install_hooks()
                self._log_init_summary()
            except Exception as exc:
                self.logger.warning(
                    "[TrainingMonitor] hook installation failed: %r. "
                    "Monitor disabled.", exc)
                self.enabled = False

    # ----- lifecycle -------------------------------------------------------
    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def _unwrap(model):
        # DDP / DataParallel wrapping.
        inner = getattr(model, 'module', None)
        return inner if inner is not None else model

    @staticmethod
    def _cls_name(module):
        return type(module).__name__

    def _install_hooks(self):
        for name, mod in self._real.named_modules():
            cls = self._cls_name(mod)
            if self.capture_attn and cls in self._ATTN_CLASS_NAMES:
                self._hooks.append(mod.register_forward_hook(
                    self._make_attn_hook(name)))
                self._hooked_names['attn'].append(f"{name}[{cls}]")
            elif self.capture_coeff and cls in self._COEFF_CLASS_NAMES:
                self._hooks.append(mod.register_forward_hook(
                    self._make_coeff_hook(name)))
                self._hooked_names['coeff'].append(f"{name}[{cls}]")
            elif self.capture_router and cls in self._ROUTER_CLASS_NAMES:
                self._hooks.append(mod.register_forward_hook(
                    self._make_router_hook(name, cls)))
                self._hooked_names['router'].append(f"{name}[{cls}]")

        if self.capture_projector:
            # Auto-detect any top-level attribute whose name ends with
            # `projectors` and is an nn.ModuleList — catches `projectors`,
            # `mos_projectors`, `align_projectors`, and any future variant.
            # Also match legacy singular `projector`/`shared_projector`.
            for attr_name, sub in self._real._modules.items():
                name_l = attr_name.lower()
                is_list = isinstance(sub, nn.ModuleList)
                if (name_l.endswith('projectors')
                        or name_l.endswith('projector')):
                    if is_list:
                        for i, child in enumerate(sub):
                            full_name = f"{attr_name}.{i}"
                            self._hooks.append(child.register_forward_hook(
                                self._make_projector_hook(full_name)))
                            self._hooked_names['projector'].append(full_name)
                    elif isinstance(sub, nn.Module):
                        self._hooks.append(sub.register_forward_hook(
                            self._make_projector_hook(attr_name)))
                        self._hooked_names['projector'].append(attr_name)

    def _log_init_summary(self):
        def _fmt(names):
            return ', '.join(names) if names else '(none)'
        self.logger.info(
            "[TrainingMonitor] enabled. hooked: attn=%s | projector=%s | "
            "coeff=%s | router=%s | grads=%s | thresholds=%s",
            _fmt(self._hooked_names['attn']),
            _fmt(self._hooked_names['projector']),
            _fmt(self._hooked_names['coeff']),
            _fmt(self._hooked_names['router']),
            self.capture_grads,
            self.thresholds,
        )

    # ----- hook factories --------------------------------------------------
    def _make_attn_hook(self, name):
        def hook(module, inputs, output):
            try:
                # Some attention modules (ExpertLocalAttention) use masked
                # softmax + nan_to_num(0), so rows belonging to unconditional
                # samples (labels==1000) or otherwise fully-masked-out rows
                # sum to exactly 0. CFG training has ~10% uncond samples, so
                # such zero rows appear EVERY step by design. We therefore
                # separate "active" rows (row_sum > _ACTIVE_ROW_EPS) from
                # inactive ones, compute min/max only over active rows, and
                # report `inactive_frac` as a benign diagnostic instead of an
                # alert signal. Dense-softmax classes (BlockAlignAttention,
                # GlobalPreAttention) have no inactive rows, so behaviour is
                # unchanged for them.
                if not torch.is_tensor(output):
                    return
                with torch.no_grad():
                    W = output.detach()
                    if W.dtype != torch.float32:
                        W = W.float()
                    row_sum = W.sum(dim=-1)
                    active_mask = row_sum > self._ACTIVE_ROW_EPS
                    num_active = int(active_mask.sum().item())
                    num_total = int(row_sum.numel())
                    inactive_frac = 1.0 - (num_active / max(num_total, 1))

                    if num_active > 0:
                        active_sums = row_sum[active_mask]
                        row_sum_min = float(active_sums.min().item())
                        row_sum_max = float(active_sums.max().item())
                        # near-empty *within active rows* — this is the real
                        # collapse signal (e.g. attention distributing its
                        # mass evenly across too many keys wouldn't trigger,
                        # but if an "active" row has sum <<1 it means softmax
                        # itself is broken).
                        near_empty = float(
                            (active_sums < self.thresholds['attn_row_sum_min'])
                            .float().mean().item()
                        )
                    else:
                        # Entire batch is uncond — nothing to check this step.
                        row_sum_min = 0.0
                        row_sum_max = 0.0
                        near_empty = 0.0

                    nan_cnt = int(torch.isnan(W).sum().item())
                    inf_cnt = int(torch.isinf(W).sum().item())
                    self._attn_stats[name] = {
                        'min': float(W.min().item()),
                        'max': float(W.max().item()),
                        'mean': float(W.mean().item()),
                        'row_sum_min': row_sum_min,
                        'row_sum_max': row_sum_max,
                        'row_empty_frac': near_empty,
                        'inactive_frac': inactive_frac,
                        'num_active': num_active,
                        'nan': nan_cnt,
                        'inf': inf_cnt,
                    }
            except Exception as exc:
                self._attn_stats[name] = {'error': repr(exc)}
        return hook

    def _make_projector_hook(self, name):
        def hook(module, inputs, output):
            try:
                if not torch.is_tensor(output):
                    return
                with torch.no_grad():
                    y = output.detach()
                    if y.dtype != torch.float32:
                        y = y.float()
                    # y shape: (*, D_z) — token-level norm
                    norms = y.norm(dim=-1)
                    nan_cnt = torch.isnan(y).sum().item()
                    inf_cnt = torch.isinf(y).sum().item()
                    self._projector_stats[name] = {
                        'norm_min': float(norms.min().item()),
                        'norm_max': float(norms.max().item()),
                        'norm_mean': float(norms.mean().item()),
                        'abs_max': float(y.abs().max().item()),
                        'nan': int(nan_cnt),
                        'inf': int(inf_cnt),
                    }
            except Exception as exc:
                self._projector_stats[name] = {'error': repr(exc)}
        return hook

    def _make_coeff_hook(self, name):
        def hook(module, inputs, output):
            try:
                if not torch.is_tensor(output):
                    return
                with torch.no_grad():
                    c = output.detach()
                    if c.dtype != torch.float32:
                        c = c.float()
                    nan_cnt = torch.isnan(c).sum().item()
                    inf_cnt = torch.isinf(c).sum().item()
                    self._coeff_stats[name] = {
                        'min': float(c.min().item()),
                        'max': float(c.max().item()),
                        'mean': float(c.mean().item()),
                        'nan': int(nan_cnt),
                        'inf': int(inf_cnt),
                    }
            except Exception as exc:
                self._coeff_stats[name] = {'error': repr(exc)}
        return hook

    def _make_router_hook(self, name, cls_name):
        """Class-aware router hook.

        Output shapes across router classes:
          BlockRouter    -> (N, T, m, n); softmaxed along m (dim=-2).
          PerBlockRouter -> (N, T, m);    softmaxed along dim=-1.
          AdaLNRouter    -> (N, T, K);    RAW LOGITS (no softmax applied yet).
        Unknown / other tensor shapes fall back to a row-sum heuristic.
        """
        def hook(module, inputs, output):
            try:
                if not torch.is_tensor(output):
                    return
                with torch.no_grad():
                    r = output.detach()
                    if r.dtype != torch.float32:
                        r = r.float()

                    if cls_name == 'BlockRouter' and r.dim() >= 4:
                        probs = r  # already softmaxed on dim=-2
                        reduce_dim = -2
                    elif cls_name == 'AdaLNRouter':
                        probs = F.softmax(r, dim=-1)
                        reduce_dim = -1
                    elif cls_name == 'PerBlockRouter':
                        probs = r  # already softmaxed on dim=-1
                        reduce_dim = -1
                    else:
                        # Fallback: detect whether output is a probability
                        # distribution on the last axis; otherwise softmax it.
                        row_sum = r.sum(dim=-1)
                        if (float(row_sum.min().item()) > 0.9 and
                                float(row_sum.max().item()) < 1.1):
                            probs = r
                        else:
                            probs = F.softmax(r, dim=-1)
                        reduce_dim = -1

                    p_safe = probs.clamp(min=1e-8)
                    entropy = -(probs * p_safe.log()).sum(dim=reduce_dim)
                    # raw_max/raw_min reports on the unsoftmaxed output so
                    # logit explosion in AdaLNRouter is visible; p_max
                    # reports on the softmaxed distribution so we can detect
                    # routing collapse (p_max -> 1).
                    self._router_stats[name] = {
                        'cls': cls_name,
                        'p_max': float(probs.max().item()),
                        'p_mean': float(probs.mean().item()),
                        'raw_min': float(r.min().item()),
                        'raw_max': float(r.max().item()),
                        'entropy_min': float(entropy.min().item()),
                        'entropy_mean': float(entropy.mean().item()),
                        'nan': int(torch.isnan(r).sum().item()),
                        'inf': int(torch.isinf(r).sum().item()),
                    }
            except Exception as exc:
                self._router_stats[name] = {'error': repr(exc)}
        return hook

    # ----- gradient stats --------------------------------------------------
    def _grad_group_for(self, param_name):
        for group_name, patterns in self._GRAD_GROUPS:
            if any(p in param_name for p in patterns):
                return group_name
        return 'backbone'

    def _collect_grad_stats(self):
        # Accumulate per-group sum-of-squared on-device and do ONE .item()
        # per group at the end. Previously we called .item() per-parameter,
        # which produced O(n_params) GPU->CPU syncs per step — noticeable at
        # log_every=1 on L/XL models with hundreds of parameters.
        groups_sq = {}   # gname -> 0-d float tensor (sum of squared)
        nan_count = None
        inf_count = None
        for pname, p in self._real.named_parameters():
            # Skip frozen params (pos_embed universally, noise_expert EMA
            # params on the noise_expert_ema variant). Without this skip
            # they'd appear as "no grad" and clutter the summary.
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad.detach()
            if nan_count is None:
                nan_count = torch.zeros((), dtype=torch.long, device=g.device)
                inf_count = torch.zeros((), dtype=torch.long, device=g.device)
            nan_count += torch.isnan(g).any().long()
            inf_count += torch.isinf(g).any().long()
            g_f = g.float()
            sq = (g_f * g_f).sum()
            gname = self._grad_group_for(pname)
            if gname in groups_sq:
                groups_sq[gname] = groups_sq[gname] + sq
            else:
                groups_sq[gname] = sq

        if not groups_sq:
            return {
                'global': 0.0,
                '__nan_params__': 0,
                '__inf_params__': 0,
            }
        # Final sync — one .item() per group plus global + nan/inf counters.
        total_sq = sum(groups_sq.values())
        result = {g: float(s.sqrt().item()) for g, s in groups_sq.items()}
        result['global'] = float(total_sq.sqrt().item())
        result['__nan_params__'] = int(nan_count.item())
        result['__inf_params__'] = int(inf_count.item())
        return result

    def _collect_cluster_center_stats(self):
        # `cluster_centers` is defined INSIDE each SparseMoeBlock (not on the
        # top-level DiT), so walk named_parameters and aggregate norms across
        # every MoE block that has it. For dense DiT / non-MoE models this
        # returns None.
        collected = []
        num_blocks = 0
        with torch.no_grad():
            for pname, p in self._real.named_parameters():
                if 'cluster_centers' not in pname:
                    continue
                # p: (num_routed_experts, hidden_size) per block
                norms = p.detach().float().norm(dim=-1)
                collected.append(norms)
                num_blocks += 1
        if not collected:
            return None
        all_norms = torch.cat(collected)
        return {
            'norm_min': float(all_norms.min().item()),
            'norm_max': float(all_norms.max().item()),
            'norm_mean': float(all_norms.mean().item()),
            'num_blocks': num_blocks,
        }

    # ----- cross-alignment stats (model-cached) ----------------------------
    def _collect_cross_align_stats(self):
        """Read ``_cross_align_stats`` cached on model during forward().

        Cross-alignment models (cross_global_block, cross_expert_local,
        cross_proto, and their MoS variants) store a dict of monitoring
        statistics during the loss computation:
          * cos_sim_absmax / cos_sim_clamp_count / cos_sim_numel —
            pre-clamp cosine similarity range; detects bf16 overflow.
          * proto_sim_neg_frac / proto_sim_min / proto_sim_mean —
            prototype similarity before clamp(min=0); detects routing
            pathology in proto variants.
          * W_row_sum_min / W_nonzero_frac — cross-alignment weight
            matrix health; detects near-zero denominator risk in proto
            normalization.
        """
        stats = getattr(self._real, '_cross_align_stats', None)
        if stats is not None:
            self._real._cross_align_stats = None
        return stats

    # ----- loss tracking ---------------------------------------------------
    def _update_loss_ema(self, losses):
        warned = []
        if not losses:
            return warned
        for k, v in losses.items():
            if torch.is_tensor(v):
                v = v.detach()
                if v.numel() != 1:
                    continue
                v = float(v.item())
            if not isinstance(v, (int, float)):
                continue
            if not math.isfinite(v):
                warned.append((k, v, 'nonfinite'))
                continue
            prev = self._loss_ema.get(k)
            if prev is None:
                self._loss_ema[k] = v
                continue
            # Jump detection that works for negative-valued losses too (e.g.
            # REPA / MoS-REPA loss = -cos_sim lives in [-1, 0]). A naive
            # `v > ratio * prev` check fails for prev < 0. Instead:
            #   * Same-sign magnitude blow-up -> jump alert.
            #   * Sign flip with non-trivial magnitude -> distinct alert
            #     (a negative-cosine loss crossing 0 to positive is an
            #     unambiguous divergence signal).
            ratio = self.thresholds['loss_jump_ratio']
            abs_prev, abs_v = abs(prev), abs(v)
            if prev * v < 0 and abs_v > 1e-3:
                warned.append((k, v, f'sign flip from {prev:.4g}'))
            elif abs_prev > 1e-6 and abs_v > ratio * abs_prev:
                warned.append((k, v, f'jump x{abs_v/abs_prev:.1f} (ema={prev:.4g})'))
            # EMA update — alpha=0.05 inherently smooths a single spike
            # (one step contributes only 5% to the running mean).
            self._loss_ema[k] = (1 - self._ema_alpha) * prev + self._ema_alpha * v
        return warned

    # ----- public API ------------------------------------------------------
    def on_step(self, step, losses=None):
        """Capture gradient stats + optionally log a summary for this step.

        Call this after ``scaler.unscale_(optimizer)`` but **before**
        ``clip_grad_norm_()`` — this is the only window where gradients
        are fully assembled, unscaled, and unclipped, so per-group norms
        reflect the true gradient magnitude.
        """
        if not self.enabled:
            return
        try:
            self._on_step_impl(step, losses)
        except Exception as exc:
            self.logger.warning(
                "[TrainingMonitor] on_step(%s) failed: %r", step, exc)

    def _on_step_impl(self, step, losses):
        alerts = []

        # Gradient stats
        grad_info = self._collect_grad_stats() if self.capture_grads else None
        if grad_info is not None:
            if grad_info['__nan_params__'] > 0:
                alerts.append(
                    f"NaN grads in {grad_info['__nan_params__']} params")
            if grad_info['__inf_params__'] > 0:
                alerts.append(
                    f"Inf grads in {grad_info['__inf_params__']} params")
            for g, v in grad_info.items():
                if g.startswith('__'):
                    continue
                if v > self.thresholds['grad_norm_warn']:
                    alerts.append(f"grad[{g}]={v:.3g} > {self.thresholds['grad_norm_warn']}")

        # Feature alerts
        for name, s in self._attn_stats.items():
            if 'error' in s:
                continue
            if s['nan'] > 0 or s['inf'] > 0:
                alerts.append(f"attn[{name}] NaN={s['nan']} Inf={s['inf']}")
            # Only check row_sum bounds when at least one active row exists —
            # a batch of entirely-uncond tokens (rare) would otherwise report
            # 0 and trip a false alert.
            if s.get('num_active', 0) > 0:
                if s['row_sum_min'] < self.thresholds['attn_row_sum_min']:
                    alerts.append(
                        f"attn[{name}] row_sum_min={s['row_sum_min']:.3g} "
                        f"(empty_frac={s['row_empty_frac']:.2%}, "
                        f"inactive={s['inactive_frac']:.2%})")
                if s['row_sum_max'] > self.thresholds['attn_row_sum_max']:
                    alerts.append(
                        f"attn[{name}] row_sum_max={s['row_sum_max']:.3g}")
        for name, s in self._projector_stats.items():
            if 'error' in s:
                continue
            if s['nan'] > 0 or s['inf'] > 0:
                alerts.append(f"proj[{name}] NaN={s['nan']} Inf={s['inf']}")
            if s['abs_max'] > self.thresholds['feature_norm_warn']:
                alerts.append(
                    f"proj[{name}] abs_max={s['abs_max']:.3g} "
                    f">{self.thresholds['feature_norm_warn']}")
        for name, s in self._coeff_stats.items():
            if 'error' in s:
                continue
            if s['nan'] > 0 or s['inf'] > 0:
                alerts.append(f"coeff[{name}] NaN={s['nan']} Inf={s['inf']}")
        for name, s in self._router_stats.items():
            if 'error' in s:
                continue
            if s['nan'] > 0 or s['inf'] > 0:
                alerts.append(f"router[{name}] NaN={s['nan']} Inf={s['inf']}")

        # Loss tracking
        loss_warned = self._update_loss_ema(losses) if self.track_losses else []
        for k, v, reason in loss_warned:
            alerts.append(f"loss[{k}]={v:.4g} {reason}")

        # Cross-alignment stats (cached on model during forward)
        ca_stats = self._collect_cross_align_stats()
        if ca_stats:
            cs_absmax = ca_stats.get('cos_sim_absmax', 0)
            if cs_absmax > self.thresholds['cos_sim_absmax_warn']:
                numel = max(ca_stats.get('cos_sim_numel', 1), 1)
                clamp_frac = ca_stats.get('cos_sim_clamp_count', 0) / numel
                alerts.append(
                    f"cos_sim absmax={cs_absmax:.4g} "
                    f"(clamp_frac={clamp_frac:.2%})")
            neg_frac = ca_stats.get('proto_sim_neg_frac')
            if neg_frac is not None and neg_frac > self.thresholds['proto_neg_frac_warn']:
                alerts.append(
                    f"proto_sim neg_frac={neg_frac:.2%} "
                    f"(min={ca_stats.get('proto_sim_min', 0):.4g})")
            w_min = ca_stats.get('W_row_sum_min')
            if w_min is not None and w_min < self.thresholds['W_row_sum_min_warn']:
                alerts.append(f"W row_sum_min={w_min:.4g}")

        # Emit alerts immediately
        for msg in alerts:
            self.logger.warning("[TrainingMonitor][step=%s] ALERT %s", step, msg)

        # Periodic summary
        if step % self.log_every != 0 or step == self._last_step_logged:
            return
        self._last_step_logged = step

        # cluster_centers params change slowly and _collect_cluster_center_stats
        # walks every named_parameter — sample only at log cadence, not every step.
        cc_stats = self._collect_cluster_center_stats()

        parts = [f"step={step}"]
        if grad_info is not None:
            grad_strs = [
                f"{g}={v:.3g}" for g, v in grad_info.items()
                if not g.startswith('__')
            ]
            parts.append("grad[" + ' '.join(grad_strs) + "]")
        if self._attn_stats:
            attn_strs = []
            for name, s in self._attn_stats.items():
                if 'error' in s:
                    continue
                inactive = s.get('inactive_frac', 0.0)
                piece = (f"{name}: W=[{s['min']:.2g},{s['max']:.2g}] "
                         f"rs=[{s['row_sum_min']:.2g},{s['row_sum_max']:.2g}]")
                if inactive > 0.0:
                    piece += f" inact={inactive:.1%}"
                attn_strs.append(piece)
            if attn_strs:
                parts.append("attn[" + ' | '.join(attn_strs) + "]")
        if self._projector_stats:
            proj_strs = []
            for name, s in self._projector_stats.items():
                if 'error' in s:
                    continue
                proj_strs.append(
                    f"{name}: |z|=[{s['norm_min']:.2g},{s['norm_max']:.2g}]"
                    f" |max|={s['abs_max']:.2g}")
            if proj_strs:
                parts.append("proj[" + ' | '.join(proj_strs) + "]")
        if self._coeff_stats:
            coeff_strs = []
            for name, s in self._coeff_stats.items():
                if 'error' in s:
                    continue
                coeff_strs.append(
                    f"{name}: [{s['min']:.2g},{s['max']:.2g}]"
                    f" mean={s['mean']:.2g}")
            if coeff_strs:
                parts.append("coeff[" + ' | '.join(coeff_strs) + "]")
        if self._router_stats:
            r_strs = []
            for name, s in self._router_stats.items():
                if 'error' in s:
                    continue
                r_strs.append(
                    f"{name}: p_max={s['p_max']:.2g} "
                    f"raw=[{s['raw_min']:.2g},{s['raw_max']:.2g}] "
                    f"H=[{s['entropy_min']:.2g},{s['entropy_mean']:.2g}]")
            if r_strs:
                parts.append("router[" + ' | '.join(r_strs) + "]")
        if cc_stats is not None:
            parts.append(
                f"cc[|w|=({cc_stats['norm_min']:.2g},"
                f"{cc_stats['norm_mean']:.2g},{cc_stats['norm_max']:.2g})]")
        if ca_stats:
            ca_parts = []
            if 'cos_sim_absmax' in ca_stats:
                numel = max(ca_stats.get('cos_sim_numel', 1), 1)
                clamp_frac = ca_stats.get('cos_sim_clamp_count', 0) / numel
                ca_parts.append(
                    f"|cos|={ca_stats['cos_sim_absmax']:.4g} "
                    f"clamp={clamp_frac:.1%}")
            if 'proto_sim_neg_frac' in ca_stats:
                ca_parts.append(
                    f"proto_neg={ca_stats['proto_sim_neg_frac']:.1%} "
                    f"min={ca_stats.get('proto_sim_min', 0):.3g}")
            if 'W_row_sum_min' in ca_stats:
                ca_parts.append(
                    f"W_rs_min={ca_stats['W_row_sum_min']:.3g} "
                    f"W_nz={ca_stats.get('W_nonzero_frac', 0):.1%}")
            if ca_parts:
                parts.append("cross[" + ' '.join(ca_parts) + "]")
        if self._loss_ema:
            loss_strs = [f"{k}={v:.3g}" for k, v in self._loss_ema.items()]
            parts.append("loss_ema[" + ' '.join(loss_strs) + "]")

        self.logger.info("[TrainingMonitor] " + ' '.join(parts))

        self._write_tensorboard(step, grad_info, ca_stats, cc_stats)

    def _write_tensorboard(self, step, grad_info, ca_stats, cc_stats):
        w = self._writer
        if w is None:
            return
        try:
            if grad_info is not None:
                for g, v in grad_info.items():
                    if g.startswith('__'):
                        continue
                    w.add_scalar(f'monitor/grad/{g}', v, step)
            for name, s in self._attn_stats.items():
                if 'error' in s:
                    continue
                w.add_scalar(f'monitor/attn/{name}/row_sum_min', s['row_sum_min'], step)
                w.add_scalar(f'monitor/attn/{name}/row_sum_max', s['row_sum_max'], step)
                if 'inactive_frac' in s:
                    w.add_scalar(f'monitor/attn/{name}/inactive_frac', s['inactive_frac'], step)
                for k in ('min', 'max', 'mean', 'row_empty_frac'):
                    if k in s:
                        w.add_scalar(f'monitor/attn/{name}/{k}', s[k], step)
            for name, s in self._projector_stats.items():
                if 'error' in s:
                    continue
                for k in ('norm_min', 'norm_max', 'norm_mean', 'abs_max'):
                    if k in s:
                        w.add_scalar(f'monitor/proj/{name}/{k}', s[k], step)
            for name, s in self._coeff_stats.items():
                if 'error' in s:
                    continue
                w.add_scalar(f'monitor/coeff/{name}/min', s['min'], step)
                w.add_scalar(f'monitor/coeff/{name}/max', s['max'], step)
                w.add_scalar(f'monitor/coeff/{name}/mean', s['mean'], step)
            for name, s in self._router_stats.items():
                if 'error' in s:
                    continue
                for k in ('p_max', 'entropy_mean', 'entropy_min',
                          'raw_min', 'raw_max'):
                    if k in s:
                        w.add_scalar(f'monitor/router/{name}/{k}', s[k], step)
            if cc_stats is not None:
                for k in ('norm_min', 'norm_max', 'norm_mean'):
                    if k in cc_stats:
                        w.add_scalar(f'monitor/cc/{k}', cc_stats[k], step)
            if ca_stats:
                if 'cos_sim_absmax' in ca_stats:
                    w.add_scalar('monitor/cross/cos_sim_absmax', ca_stats['cos_sim_absmax'], step)
                    numel = max(ca_stats.get('cos_sim_numel', 1), 1)
                    clamp_frac = ca_stats.get('cos_sim_clamp_count', 0) / numel
                    w.add_scalar('monitor/cross/cos_sim_clamp_frac', clamp_frac, step)
                if 'proto_sim_neg_frac' in ca_stats:
                    w.add_scalar('monitor/cross/proto_sim_neg_frac', ca_stats['proto_sim_neg_frac'], step)
                    w.add_scalar('monitor/cross/proto_sim_min', ca_stats.get('proto_sim_min', 0), step)
                if 'W_row_sum_min' in ca_stats:
                    w.add_scalar('monitor/cross/W_row_sum_min', ca_stats['W_row_sum_min'], step)
                    w.add_scalar('monitor/cross/W_nonzero_frac', ca_stats.get('W_nonzero_frac', 0), step)
            for name, v in self._loss_ema.items():
                w.add_scalar(f'monitor/loss_ema/{name}', v, step)
        except Exception as e:
            self.logger.warning("[TrainingMonitor] TensorBoard write failed: %s", e)