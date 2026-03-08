"""Persistent FLOPs counter using thop's counting hooks.

Registers hooks once and accumulates FLOPs across multiple forward passes,
correctly handling dynamic routing where different experts (potentially of
different sizes) are activated per forward pass.
"""

import torch
import torch.nn as nn
from thop.vision.basic_hooks import (
    count_convNd,
    count_linear,
    count_normalization,
    count_relu,
    count_softmax,
)


class FLOPsAccumulator:
    """Accumulates FLOPs across multiple forward passes of a model.

    Unlike thop.profile() which measures a single forward pass,
    this registers hooks once and accumulates total FLOPs.

    thop's hook functions expect each module to have a `total_ops` attribute
    (a torch.Tensor) and add to it during each forward pass.
    """

    def __init__(self, model, custom_ops=None):
        self.model = model
        self.custom_ops = custom_ops or {}
        self._hooks = []
        self.total_flops = 0
        self._started = False

    def start(self):
        """Register FLOPs counting hooks on all modules."""
        if self._started:
            return
        self._started = True
        self._hooks = []

        def _add_hooks(module):
            if len(list(module.children())) > 0:
                return

            # thop hooks accumulate into module.total_ops (use plain attr, not buffer,
            # to avoid polluting state_dict and to match thop's internal convention)
            module.total_ops = torch.zeros(1, dtype=torch.float64)

            fn = None
            if type(module) in self.custom_ops:
                fn = self.custom_ops[type(module)]
            else:
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    fn = count_convNd
                elif isinstance(module, nn.Linear):
                    fn = count_linear
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
                                         nn.InstanceNorm2d, nn.LayerNorm)):
                    fn = count_normalization
                elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                    fn = count_relu
                elif isinstance(module, nn.Softmax):
                    fn = count_softmax

            if fn is not None:
                handle = module.register_forward_hook(fn)
                self._hooks.append(handle)

        self.model.apply(_add_hooks)

    def collect(self):
        """Collect and accumulate FLOPs from all modules since last collect() call."""
        flops_this_pass = 0

        def _collect(module):
            nonlocal flops_this_pass
            if hasattr(module, "total_ops"):
                flops_this_pass += module.total_ops.item()
                module.total_ops.zero_()

        self.model.apply(_collect)
        self.total_flops += flops_this_pass
        return flops_this_pass

    def stop(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._started = False

    def reset(self):
        """Reset accumulated FLOPs to zero."""
        self.total_flops = 0

        def _reset(module):
            if hasattr(module, "total_ops"):
                module.total_ops.zero_()

        self.model.apply(_reset)
